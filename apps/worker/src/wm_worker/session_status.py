from __future__ import annotations

import logging

from wm_worker.livekit_adapter import LiveKitAdapter
from wm_worker.models import FrameMetrics, StatusMessageType
from wm_worker.protocol import encode_status_message


class SessionStatusPublisher:
    def __init__(
        self,
        *,
        livekit: LiveKitAdapter,
        session_id: str | None,
        logger: logging.Logger,
    ) -> None:
        self._livekit = livekit
        self._session_id = session_id
        self._logger = logger
        self._seq = 0

    async def started(self, worker_id: str) -> None:
        await self._send(StatusMessageType.STARTED, {"worker_id": worker_id})

    async def pong(self, payload: dict[str, object]) -> None:
        await self._send(StatusMessageType.PONG, payload)

    async def frame_metrics(self, metrics: FrameMetrics) -> None:
        await self._send(
            StatusMessageType.FRAME_METRICS,
            {
                "effective_fps": round(metrics.effective_fps, 3),
                "queue_depth": metrics.queue_depth,
                "inference_ms_p50": round(metrics.inference_ms_p50, 3),
                "publish_dropped_frames": metrics.publish_dropped_frames,
            },
        )

    async def error(self, error_code: str, *, publish_dropped_frames: int) -> None:
        await self._send(
            StatusMessageType.ERROR,
            {
                "error_code": error_code,
                "publish_dropped_frames": publish_dropped_frames,
            },
        )

    async def ended(
        self,
        *,
        ended_by_control: bool,
        publish_dropped_frames: int,
    ) -> None:
        await self._send(
            StatusMessageType.ENDED,
            {
                "ended_by_control": ended_by_control,
                "publish_dropped_frames": publish_dropped_frames,
            },
        )

    async def _send(
        self,
        msg_type: StatusMessageType,
        payload: dict[str, object],
    ) -> None:
        self._seq += 1
        encoded = encode_status_message(
            msg_type,
            session_id=self._session_id,
            seq=self._seq,
            payload=payload,
        )
        try:
            await self._livekit.send_status(encoded)
        except Exception as exc:  # pragma: no cover - integration boundary
            self._logger.warning("failed sending status message: %s", exc)
