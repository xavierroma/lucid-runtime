from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from wm_worker.action_buffer import ActionBuffer
from wm_worker.config import WorkerConfig
from wm_worker.coordinator_client import CoordinatorClient, CoordinatorError
from wm_worker.frame_pipeline import FramePipeline
from wm_worker.health_server import HealthServer, HealthState
from wm_worker.livekit_adapter import (
    FakeLiveKitAdapter,
    LiveKitAdapter,
    RealLiveKitAdapter,
)
from wm_worker.models import SessionResult, StatusMessageType
from wm_worker.protocol import ProtocolError, encode_status_message, parse_control_message
from wm_worker.yume_engine import YumeEngine, YumeEngineError


class WorkerRunner:
    def __init__(self, config: WorkerConfig, logger: logging.Logger) -> None:
        self._config = config
        self._logger = logger
        self._health_state = HealthState()
        self._health_server = HealthServer(config.worker_health_port, self._health_state)
        self._coordinator = CoordinatorClient(
            base_url=config.coordinator_base_url,
            worker_id=config.worker_id,
            worker_internal_token=config.worker_internal_token,
        )
        self._engine = YumeEngine(config, logger)
        self._shutdown_event = asyncio.Event()
        self._session_stop_event = asyncio.Event()
        self._active_session_id: str | None = None
        self._status_seq = 0
        self._coordinator_unhealthy_since: float | None = None

    async def run(self) -> None:
        await self._health_server.start()
        self._logger.info("health server listening on :%s", self._config.worker_health_port)
        try:
            await self._engine.load()
            self._health_state.ready = True
            await self._register_with_retry()
            heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="heartbeat_loop")
            try:
                await self._idle_poll_loop()
            finally:
                heartbeat_task.cancel()
                await asyncio.gather(heartbeat_task, return_exceptions=True)
        finally:
            self._health_state.alive = False
            self._health_state.ready = False
            await self._coordinator.close()
            await self._health_server.stop()

    def request_shutdown(self) -> None:
        self._logger.info("shutdown requested")
        self._shutdown_event.set()
        self._session_stop_event.set()

    async def _register_with_retry(self) -> None:
        delay_s = 1.0
        while not self._shutdown_event.is_set():
            try:
                await self._coordinator.register_worker()
                self._logger.info("registered worker_id=%s", self._config.worker_id)
                return
            except Exception as exc:
                self._logger.warning("worker registration failed: %s", exc)
            await asyncio.sleep(delay_s)
            delay_s = min(delay_s * 2, 10.0)

    async def _idle_poll_loop(self) -> None:
        poll_delay_s = self._config.assignment_poll_ms / 1000.0
        while not self._shutdown_event.is_set():
            if self._active_session_id is not None:
                await asyncio.sleep(0.1)
                continue
            try:
                assignment = await self._coordinator.poll_assignment()
            except Exception as exc:
                self._logger.warning("assignment poll failed: %s", exc)
                await asyncio.sleep(poll_delay_s)
                continue
            if assignment is None:
                await asyncio.sleep(poll_delay_s)
                continue
            await self._run_session(assignment)

    async def _heartbeat_loop(self) -> None:
        interval_s = self._config.heartbeat_interval_ms / 1000.0
        while not self._shutdown_event.is_set():
            try:
                heartbeat = await self._coordinator.heartbeat_worker()
                self._coordinator_unhealthy_since = None
                if heartbeat.cancel_active_session and self._active_session_id:
                    self._logger.info(
                        "coordinator requested cancellation for session_id=%s",
                        self._active_session_id,
                    )
                    self._session_stop_event.set()
            except CoordinatorError as exc:
                self._logger.warning("heartbeat error: %s", exc)
                self._track_coordinator_unavailability()
            except Exception as exc:  # pragma: no cover - network/runtime failures
                self._logger.warning("heartbeat failure: %s", exc)
                self._track_coordinator_unavailability()
            await asyncio.sleep(interval_s)

    def _track_coordinator_unavailability(self) -> None:
        now = time.monotonic()
        if self._coordinator_unhealthy_since is None:
            self._coordinator_unhealthy_since = now
            return
        if self._active_session_id is None:
            return
        if now - self._coordinator_unhealthy_since > 30:
            self._logger.error("coordinator unreachable for >30s; ending session")
            self._session_stop_event.set()

    async def _run_session(self, assignment) -> None:
        self._active_session_id = assignment.session_id
        self._session_stop_event = asyncio.Event()
        action_buffer = ActionBuffer(self._config.yume_base_prompt)
        frame_pipeline = FramePipeline(self._config.yume_max_queue_frames)
        livekit = self._build_livekit_adapter()
        result = SessionResult(error_code=None, ended_by_control=False)
        self._logger.info(
            "session assigned session_id=%s room_name=%s",
            assignment.session_id,
            assignment.room_name,
        )
        try:
            await livekit.connect_and_publish(assignment)
            await self._coordinator.mark_running(assignment.session_id)
            await self._engine.start_session(self._config.yume_base_prompt)
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
            if self._coordinator_unhealthy_since and result.error_code is None:
                result.error_code = "COORDINATOR_UNREACHABLE"
        except YumeEngineError as exc:
            result.error_code = "MODEL_RUNTIME_ERROR"
            self._logger.exception("yume engine error: %s", exc)
        except Exception as exc:  # pragma: no cover - integration boundary
            result.error_code = "LIVEKIT_DISCONNECT"
            self._logger.exception("session failed: %s", exc)
        finally:
            await self._engine.end_session()
            await self._emit_terminal_status(livekit, assignment.session_id, result, frame_pipeline)
            await livekit.disconnect()
            try:
                await self._coordinator.mark_ended(assignment.session_id, result.error_code)
            except Exception as exc:
                self._logger.warning("failed to mark session ended: %s", exc)
            self._active_session_id = None
            self._session_stop_event.clear()
            self._logger.info(
                "session finished session_id=%s error_code=%s",
                assignment.session_id,
                result.error_code,
            )

    def _build_livekit_adapter(self) -> LiveKitAdapter:
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
                    exc = task.exception()
                    if exc is not None:
                        result.error_code = result.error_code or "WORKER_TASK_ERROR"
                        self._logger.exception("session task failed: %s", exc)
                    self._session_stop_event.set()
                    break
            await asyncio.sleep(0.05)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

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
            for frame in chunk.frames:
                if self._session_stop_event.is_set():
                    break
                await frame_pipeline.push(frame, inference_ms=chunk.inference_ms)
            await asyncio.sleep(0)

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
