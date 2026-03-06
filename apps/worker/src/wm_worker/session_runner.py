from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import numpy as np

from wm_worker.config import RuntimeConfig, SessionConfig
from wm_worker.coordinator_client import CoordinatorClient
from wm_worker.frame_pipeline import FramePipeline
from wm_worker.livekit_adapter import (
    FakeLiveKitAdapter,
    LiveKitAdapter,
    RealLiveKitAdapter,
)
from wm_worker.models import Assignment, SessionResult
from wm_worker.session_control import SessionControlReducer
from wm_worker.session_status import SessionStatusPublisher
from wm_worker.yume_engine import YumeEngine, YumeEngineError


class SessionRunner:
    HEARTBEAT_INTERVAL_SECS = 2.0

    def __init__(
        self,
        runtime_config: RuntimeConfig,
        session_config: SessionConfig,
        logger: logging.Logger,
        *,
        coordinator: CoordinatorClient | None = None,
        livekit_factory: Callable[[], LiveKitAdapter] | None = None,
        engine: YumeEngine | None = None,
    ) -> None:
        self._runtime_config = runtime_config
        self._session_config = session_config
        self._logger = logger
        self._coordinator = coordinator or CoordinatorClient(
            base_url=session_config.coordinator_base_url,
            worker_internal_token=session_config.worker_internal_token,
        )
        self._engine = engine or YumeEngine(runtime_config, logger)
        self._livekit_factory = livekit_factory
        self._session_stop_event = asyncio.Event()
        self._loaded = False

    async def load(self) -> None:
        if self._loaded:
            return
        await self._engine.load()
        self._loaded = True

    async def close(self) -> None:
        await self._coordinator.close()

    async def run_session(self, assignment: Assignment) -> SessionResult:
        if not self._loaded:
            await self.load()

        self._session_stop_event = asyncio.Event()
        control = SessionControlReducer(self._runtime_config.yume_base_prompt, self._logger)
        frame_pipeline = FramePipeline(self._runtime_config.yume_max_queue_frames)
        livekit = self._build_livekit_adapter()
        status = SessionStatusPublisher(
            livekit=livekit,
            session_id=assignment.session_id,
            logger=self._logger,
        )
        result = SessionResult(error_code=None, ended_by_control=False)

        self._logger.info(
            "starting session session_id=%s room_name=%s",
            assignment.session_id,
            assignment.room_name,
        )
        try:
            await livekit.connect_and_publish(assignment)
            await self._engine.start_session(self._runtime_config.yume_base_prompt)
            await self._engine.update_snapshot(await control.snapshot())
            initial_chunk = await self._engine.generate_chunk()
            self._log_chunk_stats("initial", initial_chunk.frames, initial_chunk.inference_ms)
            for frame in initial_chunk.frames:
                await frame_pipeline.push(frame, inference_ms=initial_chunk.inference_ms)
            await self._coordinator.mark_running(assignment.session_id)
            await status.started(self._session_config.worker_id)

            fallback_frame = np.zeros(
                (
                    self._runtime_config.frame_height,
                    self._runtime_config.frame_width,
                    3,
                ),
                dtype=np.uint8,
            )
            tasks = [
                asyncio.create_task(
                    self._control_loop(
                        control=control,
                        session_id=assignment.session_id,
                        status=status,
                        livekit=livekit,
                        result=result,
                    ),
                    name="control_loop",
                ),
                asyncio.create_task(
                    self._inference_loop(
                        control=control,
                        frame_pipeline=frame_pipeline,
                    ),
                    name="inference_loop",
                ),
                asyncio.create_task(
                    frame_pipeline.publish_loop(
                        livekit.publish_frame,
                        self._session_stop_event,
                        self._runtime_config.target_fps,
                        fallback_frame=fallback_frame,
                    ),
                    name="publish_loop",
                ),
                asyncio.create_task(
                    self._metrics_loop(
                        frame_pipeline=frame_pipeline,
                        status=status,
                    ),
                    name="metrics_loop",
                ),
                asyncio.create_task(
                    self._heartbeat_loop(session_id=assignment.session_id),
                    name="heartbeat_loop",
                ),
            ]
            await self._watch_session_tasks(tasks, result)
        except YumeEngineError as exc:
            result.error_code = "MODEL_RUNTIME_ERROR"
            self._logger.exception("yume engine error: %s", exc)
        except Exception as exc:  # pragma: no cover - integration boundary
            if self._is_modal_input_cancellation(exc):
                self._logger.info("session cancelled by modal input cancellation")
            else:
                result.error_code = "LIVEKIT_DISCONNECT"
                self._logger.exception("session failed: %s", exc)
        finally:
            await self._engine.end_session()
            await self._emit_terminal_status(status, result, frame_pipeline)
            await livekit.disconnect()
            try:
                end_reason = None
                if result.error_code:
                    end_reason = "WORKER_REPORTED_ERROR"
                elif result.ended_by_control:
                    end_reason = "CONTROL_REQUESTED"
                await self._coordinator.mark_ended(
                    assignment.session_id,
                    result.error_code,
                    end_reason,
                )
            except Exception as exc:
                self._logger.warning("failed to mark session ended: %s", exc)
            self._logger.info(
                "session finished session_id=%s error_code=%s",
                assignment.session_id,
                result.error_code,
            )

        return result

    def _build_livekit_adapter(self) -> LiveKitAdapter:
        if self._livekit_factory is not None:
            return self._livekit_factory()
        if self._runtime_config.livekit_mode == "real":
            return RealLiveKitAdapter(
                livekit_url=self._runtime_config.livekit_url,
                frame_width=self._runtime_config.frame_width,
                frame_height=self._runtime_config.frame_height,
                status_topic=self._runtime_config.status_topic,
                logger=self._logger,
            )
        return FakeLiveKitAdapter(self._logger)

    async def _watch_session_tasks(
        self, tasks: list[asyncio.Task[None]], result: SessionResult
    ) -> None:
        stop_task = asyncio.create_task(self._session_stop_event.wait(), name="session_stop")
        try:
            done, _ = await asyncio.wait(
                [*tasks, stop_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                if task is stop_task:
                    continue
                self._consume_task_result(task, result)
        finally:
            self._session_stop_event.set()
            stop_task.cancel()
            for task in tasks:
                task.cancel()
            await asyncio.gather(stop_task, *tasks, return_exceptions=True)

    @staticmethod
    def _is_modal_input_cancellation(exc: BaseException) -> bool:
        return (
            exc.__class__.__module__ == "modal.exception"
            and exc.__class__.__name__ == "InputCancellation"
        )

    async def _control_loop(
        self,
        *,
        control: SessionControlReducer,
        session_id: str,
        status: SessionStatusPublisher,
        livekit: LiveKitAdapter,
        result: SessionResult,
    ) -> None:
        while not self._session_stop_event.is_set():
            raw = await livekit.recv_control(timeout_s=0.5)
            if raw is None:
                continue
            outcome = await control.reduce(raw, session_id=session_id)
            if outcome.stop_requested:
                result.ended_by_control = True
                self._session_stop_event.set()
                continue
            if outcome.pong_payload is not None:
                await status.pong(outcome.pong_payload)

    async def _inference_loop(
        self, *, control: SessionControlReducer, frame_pipeline: FramePipeline
    ) -> None:
        while not self._session_stop_event.is_set():
            snapshot = await control.snapshot()
            await self._engine.update_snapshot(snapshot)
            chunk = await self._engine.generate_chunk()
            self._log_chunk_stats("steady_state", chunk.frames, chunk.inference_ms)
            for frame in chunk.frames:
                if self._session_stop_event.is_set():
                    break
                await frame_pipeline.push(frame, inference_ms=chunk.inference_ms)
            await asyncio.sleep(0)

    def _log_chunk_stats(
        self,
        label: str,
        frames: list[np.ndarray],
        inference_ms: float,
    ) -> None:
        if not frames:
            self._logger.warning("generated empty frame chunk label=%s", label)
            return
        sample = np.asarray(frames[0], dtype=np.uint8)
        self._logger.info(
            (
                "generated yume chunk label=%s frames=%s inference_ms_per_frame=%.2f "
                "first_frame_mean=%.2f first_frame_std=%.2f"
            ),
            label,
            len(frames),
            inference_ms,
            float(sample.mean()),
            float(sample.std()),
        )

    async def _metrics_loop(
        self,
        *,
        frame_pipeline: FramePipeline,
        status: SessionStatusPublisher,
    ) -> None:
        while not self._session_stop_event.is_set():
            await status.frame_metrics(frame_pipeline.metrics())
            await asyncio.sleep(1.0)

    async def _heartbeat_loop(self, *, session_id: str) -> None:
        while not self._session_stop_event.is_set():
            try:
                await self._coordinator.mark_heartbeat(session_id)
            except Exception as exc:
                self._logger.warning("failed sending coordinator heartbeat: %s", exc)
            try:
                await asyncio.wait_for(
                    self._session_stop_event.wait(),
                    timeout=self.HEARTBEAT_INTERVAL_SECS,
                )
            except TimeoutError:
                continue

    async def _emit_terminal_status(
        self,
        status: SessionStatusPublisher,
        result: SessionResult,
        frame_pipeline: FramePipeline,
    ) -> None:
        metrics = frame_pipeline.metrics()
        if result.error_code:
            await status.error(
                result.error_code,
                publish_dropped_frames=metrics.publish_dropped_frames,
            )
        await status.ended(
            ended_by_control=result.ended_by_control,
            publish_dropped_frames=metrics.publish_dropped_frames,
        )

    def _consume_task_result(
        self,
        task: asyncio.Task[None],
        result: SessionResult,
    ) -> None:
        task_name = task.get_name()
        if task.cancelled():
            result.error_code = result.error_code or "WORKER_TASK_ERROR"
            self._logger.error("session task cancelled unexpectedly task=%s", task_name)
            return

        exc = task.exception()
        if exc is None:
            return
        if self._is_modal_input_cancellation(exc):
            self._logger.info(
                "session task cancelled by modal input cancellation task=%s",
                task_name,
            )
            return

        result.error_code = result.error_code or "WORKER_TASK_ERROR"
        self._logger.error(
            "session task failed task=%s",
            task_name,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
