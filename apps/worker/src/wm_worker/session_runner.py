from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import numpy as np

from wm_worker.action_buffer import ActionBuffer
from wm_worker.config import WorkerConfig
from wm_worker.coordinator_client import CoordinatorClient
from wm_worker.frame_pipeline import FramePipeline
from wm_worker.livekit_adapter import (
    FakeLiveKitAdapter,
    LiveKitAdapter,
    RealLiveKitAdapter,
)
from wm_worker.models import Assignment, SessionResult, StatusMessageType
from wm_worker.protocol import ProtocolError, encode_status_message, parse_control_message
from wm_worker.yume_engine import YumeEngine, YumeEngineError


class SessionRunner:
    def __init__(
        self,
        config: WorkerConfig,
        logger: logging.Logger,
        *,
        coordinator: CoordinatorClient | None = None,
        livekit_factory: Callable[[], LiveKitAdapter] | None = None,
        engine: YumeEngine | None = None,
    ) -> None:
        self._config = config
        self._logger = logger
        self._coordinator = coordinator or CoordinatorClient(
            base_url=config.coordinator_base_url,
            worker_internal_token=config.worker_internal_token,
        )
        self._engine = engine or YumeEngine(config, logger)
        self._livekit_factory = livekit_factory
        self._session_stop_event = asyncio.Event()
        self._status_seq = 0
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
        action_buffer = ActionBuffer(self._config.yume_base_prompt)
        frame_pipeline = FramePipeline(self._config.yume_max_queue_frames)
        livekit = self._build_livekit_adapter()
        result = SessionResult(error_code=None, ended_by_control=False)

        self._logger.info(
            "starting session session_id=%s room_name=%s",
            assignment.session_id,
            assignment.room_name,
        )
        try:
            await livekit.connect_and_publish(assignment)
            await self._engine.start_session(self._config.yume_base_prompt)
            await self._engine.update_snapshot(await action_buffer.snapshot())
            initial_chunk = await self._engine.generate_chunk()
            self._log_chunk_stats("initial", initial_chunk.frames, initial_chunk.inference_ms)
            for frame in initial_chunk.frames:
                await frame_pipeline.push(frame, inference_ms=initial_chunk.inference_ms)
            await self._coordinator.mark_running(assignment.session_id)
            await self._send_status(
                livekit,
                StatusMessageType.STARTED,
                assignment.session_id,
                {"worker_id": self._config.worker_id},
            )

            fallback_frame = np.zeros(
                (self._config.frame_height, self._config.frame_width, 3), dtype=np.uint8
            )
            tasks = [
                asyncio.create_task(
                    self._control_loop(
                        livekit=livekit,
                        session_id=assignment.session_id,
                        action_buffer=action_buffer,
                        result=result,
                    ),
                    name="control_loop",
                ),
                asyncio.create_task(
                    self._inference_loop(
                        action_buffer=action_buffer,
                        frame_pipeline=frame_pipeline,
                    ),
                    name="inference_loop",
                ),
                asyncio.create_task(
                    frame_pipeline.publish_loop(
                        livekit.publish_frame,
                        self._session_stop_event,
                        self._config.target_fps,
                        fallback_frame=fallback_frame,
                    ),
                    name="publish_loop",
                ),
                asyncio.create_task(
                    self._metrics_loop(
                        livekit=livekit,
                        session_id=assignment.session_id,
                        frame_pipeline=frame_pipeline,
                    ),
                    name="metrics_loop",
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
            await self._emit_terminal_status(
                livekit, assignment.session_id, result, frame_pipeline
            )
            await livekit.disconnect()
            try:
                await self._coordinator.mark_ended(assignment.session_id, result.error_code)
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
        if self._config.livekit_mode == "real":
            return RealLiveKitAdapter(
                livekit_url=self._config.livekit_url,
                frame_width=self._config.frame_width,
                frame_height=self._config.frame_height,
                status_topic=self._config.status_topic,
                logger=self._logger,
            )
        return FakeLiveKitAdapter(self._logger)

    async def _watch_session_tasks(
        self, tasks: list[asyncio.Task[None]], result: SessionResult
    ) -> None:
        while not self._session_stop_event.is_set():
            for task in tasks:
                if task.done():
                    task_name = task.get_name()
                    if task.cancelled():
                        result.error_code = result.error_code or "WORKER_TASK_ERROR"
                        self._logger.error(
                            "session task cancelled unexpectedly task=%s", task_name
                        )
                        self._session_stop_event.set()
                        break

                    exc = task.exception()
                    if exc is not None:
                        if self._is_modal_input_cancellation(exc):
                            self._logger.info(
                                "session task cancelled by modal input cancellation task=%s",
                                task_name,
                            )
                        else:
                            result.error_code = result.error_code or "WORKER_TASK_ERROR"
                            self._logger.error(
                                "session task failed task=%s",
                                task_name,
                                exc_info=(type(exc), exc, exc.__traceback__),
                            )
                    self._session_stop_event.set()
                    break
            await asyncio.sleep(0.05)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _is_modal_input_cancellation(exc: BaseException) -> bool:
        return (
            exc.__class__.__module__ == "modal.exception"
            and exc.__class__.__name__ == "InputCancellation"
        )

    async def _control_loop(
        self,
        *,
        livekit: LiveKitAdapter,
        session_id: str,
        action_buffer: ActionBuffer,
        result: SessionResult,
    ) -> None:
        while not self._session_stop_event.is_set():
            raw = await livekit.recv_control(timeout_s=0.5)
            if raw is None:
                continue
            try:
                envelope = parse_control_message(raw)
            except ProtocolError as exc:
                self._logger.warning("invalid control payload: %s", exc)
                continue
            if envelope.session_id and envelope.session_id != session_id:
                continue
            if envelope.type.value == "end":
                result.ended_by_control = True
                self._session_stop_event.set()
                continue
            if envelope.type.value == "ping":
                await self._send_status(
                    livekit,
                    StatusMessageType.PONG,
                    session_id,
                    {"client_ts_ms": envelope.payload.get("client_ts_ms")},
                )
                continue
            await action_buffer.apply_control(envelope)

    async def _inference_loop(
        self, *, action_buffer: ActionBuffer, frame_pipeline: FramePipeline
    ) -> None:
        while not self._session_stop_event.is_set():
            snapshot = await action_buffer.snapshot()
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
        livekit: LiveKitAdapter,
        session_id: str,
        frame_pipeline: FramePipeline,
    ) -> None:
        while not self._session_stop_event.is_set():
            metrics = frame_pipeline.metrics()
            await self._send_status(
                livekit,
                StatusMessageType.FRAME_METRICS,
                session_id,
                {
                    "effective_fps": round(metrics.effective_fps, 3),
                    "queue_depth": metrics.queue_depth,
                    "inference_ms_p50": round(metrics.inference_ms_p50, 3),
                    "publish_dropped_frames": metrics.publish_dropped_frames,
                },
            )
            await asyncio.sleep(1.0)

    async def _emit_terminal_status(
        self,
        livekit: LiveKitAdapter,
        session_id: str,
        result: SessionResult,
        frame_pipeline: FramePipeline,
    ) -> None:
        metrics = frame_pipeline.metrics()
        if result.error_code:
            await self._send_status(
                livekit,
                StatusMessageType.ERROR,
                session_id,
                {
                    "error_code": result.error_code,
                    "publish_dropped_frames": metrics.publish_dropped_frames,
                },
            )
        await self._send_status(
            livekit,
            StatusMessageType.ENDED,
            session_id,
            {
                "ended_by_control": result.ended_by_control,
                "publish_dropped_frames": metrics.publish_dropped_frames,
            },
        )

    async def _send_status(
        self,
        livekit: LiveKitAdapter,
        msg_type: StatusMessageType,
        session_id: str | None,
        payload: dict[str, object],
    ) -> None:
        self._status_seq += 1
        encoded = encode_status_message(
            msg_type, session_id=session_id, seq=self._status_seq, payload=payload
        )
        try:
            await livekit.send_status(encoded)
        except Exception as exc:  # pragma: no cover - integration boundary
            self._logger.warning("failed sending status message: %s", exc)
