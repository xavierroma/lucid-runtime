from __future__ import annotations

import asyncio
import logging
from typing import Protocol

import numpy as np

from wm_worker.models import Assignment


class LiveKitAdapter(Protocol):
    async def connect_and_publish(self, assignment: Assignment) -> None: ...
    async def disconnect(self) -> None: ...
    async def publish_frame(self, frame: np.ndarray) -> None: ...
    async def recv_control(self, timeout_s: float) -> bytes | None: ...
    async def send_status(self, payload: bytes) -> None: ...


class LiveKitUnavailableError(RuntimeError):
    pass


class FakeLiveKitAdapter:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger
        self._controls: asyncio.Queue[bytes] = asyncio.Queue()
        self._connected = False

    async def connect_and_publish(self, assignment: Assignment) -> None:
        self._logger.info(
            "fake livekit connected room=%s track=%s",
            assignment.room_name,
            assignment.video_track_name,
        )
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False
        self._logger.info("fake livekit disconnected")

    async def publish_frame(self, frame: np.ndarray) -> None:
        if not self._connected:
            raise LiveKitUnavailableError("fake livekit is not connected")
        _ = frame

    async def recv_control(self, timeout_s: float) -> bytes | None:
        timeout_s = max(timeout_s, 0.0)
        if timeout_s == 0:
            try:
                return self._controls.get_nowait()
            except asyncio.QueueEmpty:
                return None

        try:
            return await asyncio.wait_for(self._controls.get(), timeout=timeout_s)
        except TimeoutError:
            return None

    async def send_status(self, payload: bytes) -> None:
        _ = payload

    async def inject_control(self, payload: bytes) -> None:
        await self._controls.put(payload)


class RealLiveKitAdapter:
    """Best-effort wrapper around the LiveKit Python RTC SDK."""

    def __init__(
        self,
        *,
        livekit_url: str,
        frame_width: int,
        frame_height: int,
        status_topic: str,
        logger: logging.Logger,
    ) -> None:
        self._livekit_url = livekit_url
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._logger = logger
        self._room = None
        self._video_source = None
        self._track = None
        self._control_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._status_topic = status_topic
        self._control_topic = "wm.control.v1"
        self._published_frames = 0

    async def connect_and_publish(self, assignment: Assignment) -> None:
        self._control_topic = assignment.control_topic
        try:
            from livekit import rtc  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on runtime package
            raise LiveKitUnavailableError(
                "livekit package is missing; install wm-worker[livekit]"
            ) from exc

        self._room = rtc.Room()

        @self._room.on("data_received")
        def _on_data(data_packet):  # pragma: no cover - callback depends on runtime
            payload = getattr(data_packet, "data", None)
            topic = getattr(data_packet, "topic", "") or ""
            if payload is None:
                return
            if topic and topic != self._control_topic:
                return
            try:
                self._control_queue.put_nowait(bytes(payload))
            except asyncio.QueueFull:
                self._logger.warning("dropping control message because queue is full")

        await self._room.connect(self._livekit_url, assignment.worker_access_token)
        self._video_source = rtc.VideoSource(self._frame_width, self._frame_height)
        self._track = rtc.LocalVideoTrack.create_video_track(
            assignment.video_track_name, self._video_source
        )
        publish_options = None
        try:
            publish_options = rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_CAMERA,
                simulcast=True,
            )
        except Exception:
            publish_options = None
        if publish_options is None:
            await self._room.local_participant.publish_track(self._track)
        else:
            await self._room.local_participant.publish_track(self._track, publish_options)
        self._logger.info("connected to livekit room=%s", assignment.room_name)

    async def disconnect(self) -> None:
        if self._room is not None:
            await self._room.disconnect()
        self._room = None
        self._track = None
        self._video_source = None

    async def publish_frame(self, frame: np.ndarray) -> None:
        if self._video_source is None:
            raise LiveKitUnavailableError("video source is not initialized")
        rgb_frame = np.ascontiguousarray(frame, dtype=np.uint8)
        try:
            from livekit import rtc  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on runtime package
            raise LiveKitUnavailableError("livekit package is missing") from exc
        video_frame = rtc.VideoFrame(
            width=rgb_frame.shape[1],
            height=rgb_frame.shape[0],
            type=rtc.VideoBufferType.RGB24,
            data=memoryview(rgb_frame).cast("B"),
        )
        self._video_source.capture_frame(video_frame)
        self._published_frames += 1
        if self._published_frames == 1:
            self._logger.info(
                "published first livekit frame size=%sx%s mean=%.2f std=%.2f",
                rgb_frame.shape[1],
                rgb_frame.shape[0],
                float(rgb_frame.mean()),
                float(rgb_frame.std()),
            )

    async def recv_control(self, timeout_s: float) -> bytes | None:
        timeout_s = max(timeout_s, 0.0)
        if timeout_s == 0:
            try:
                return self._control_queue.get_nowait()
            except asyncio.QueueEmpty:
                return None

        try:
            return await asyncio.wait_for(self._control_queue.get(), timeout=timeout_s)
        except TimeoutError:
            return None

    async def send_status(self, payload: bytes) -> None:
        if self._room is None:
            return
        await self._room.local_participant.publish_data(
            payload,
            topic=self._status_topic,
            reliable=True,
        )
