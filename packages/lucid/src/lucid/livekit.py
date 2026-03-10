from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

import numpy as np

from .protocol import (
    ControlMessageType,
    ProtocolError,
    StatusMessageType,
    encode_status_message,
    parse_control_message,
)
from .publish import OutputSpec
from .runtime import ActionDispatchError, LucidRuntime, SessionContext
from .types import Assignment, FrameMetrics


def _encode_jwt(payload: dict[str, Any], secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}

    def _b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    header_segment = _b64url(
        json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    payload_segment = _b64url(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header_segment}.{payload_segment}.{_b64url(signature)}"


def mint_access_token(
    *,
    api_key: str,
    api_secret: str,
    identity: str,
    room_name: str,
    ttl_seconds: int = 3600,
) -> str:
    now = int(time.time())
    return _encode_jwt(
        {
            "iss": api_key,
            "sub": identity,
            "nbf": now,
            "exp": now + ttl_seconds,
            "video": {
                "roomJoin": True,
                "room": room_name,
            },
        },
        api_secret,
    )


class LiveKitAdapter(Protocol):
    async def connect(self, assignment: Assignment, outputs: tuple[OutputSpec, ...]) -> None: ...
    async def disconnect(self) -> None: ...
    async def publish_video(self, output_name: str, frame: np.ndarray) -> None: ...
    async def publish_audio(self, output_name: str, samples: np.ndarray) -> None: ...
    async def publish_data(
        self,
        output_name: str,
        payload: bytes,
        *,
        reliable: bool = True,
    ) -> None: ...
    async def recv_control(self, timeout_s: float) -> bytes | None: ...
    async def send_status(self, payload: bytes) -> None: ...


class LiveKitUnavailableError(RuntimeError):
    pass


class FakeLiveKitAdapter:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger
        self._controls: asyncio.Queue[bytes] = asyncio.Queue()
        self._connected = False

    async def connect(self, assignment: Assignment, outputs: tuple[OutputSpec, ...]) -> None:
        self._logger.info(
            "fake livekit connected room=%s outputs=%s",
            assignment.room_name,
            ",".join(output.name for output in outputs) or "<none>",
        )
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False
        self._logger.info("fake livekit disconnected")

    async def publish_video(self, output_name: str, frame: np.ndarray) -> None:
        if not self._connected:
            raise LiveKitUnavailableError("fake livekit is not connected")
        _ = output_name
        _ = frame

    async def publish_audio(self, output_name: str, samples: np.ndarray) -> None:
        if not self._connected:
            raise LiveKitUnavailableError("fake livekit is not connected")
        _ = output_name
        _ = samples

    async def publish_data(
        self,
        output_name: str,
        payload: bytes,
        *,
        reliable: bool = True,
    ) -> None:
        if not self._connected:
            raise LiveKitUnavailableError("fake livekit is not connected")
        _ = output_name
        _ = payload
        _ = reliable

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
        self._video_sources: dict[str, object] = {}
        self._audio_sources: dict[str, object] = {}
        self._control_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._status_topic = status_topic
        self._control_topic = "wm.control"
        self._published_frames = 0
        self._outputs: dict[str, OutputSpec] = {}

    async def connect(self, assignment: Assignment, outputs: tuple[OutputSpec, ...]) -> None:
        self._control_topic = assignment.control_topic
        self._outputs = {output.name: output for output in outputs}
        try:
            from livekit import rtc  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on runtime package
            raise LiveKitUnavailableError(
                "livekit package is missing; install lucid[livekit]"
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
        for output in outputs:
            if output.kind == "video":
                source = rtc.VideoSource(
                    int(output.config["width"]),
                    int(output.config["height"]),
                )
                track = rtc.LocalVideoTrack.create_video_track(output.name, source)
                publish_options = None
                try:
                    publish_options = rtc.TrackPublishOptions(
                        source=rtc.TrackSource.SOURCE_CAMERA,
                        simulcast=True,
                    )
                except Exception:
                    publish_options = None
                if publish_options is None:
                    await self._room.local_participant.publish_track(track)
                else:
                    await self._room.local_participant.publish_track(track, publish_options)
                self._video_sources[output.name] = source
                continue
            if output.kind == "audio":
                try:
                    source = rtc.AudioSource(
                        int(output.config["sample_rate_hz"]),
                        int(output.config["channels"]),
                    )
                    track = rtc.LocalAudioTrack.create_audio_track(output.name, source)
                    await self._room.local_participant.publish_track(track)
                    self._audio_sources[output.name] = source
                except Exception as exc:
                    raise LiveKitUnavailableError(
                        f"failed creating audio track for output {output.name}: {exc}"
                    ) from exc
        self._logger.info("connected to livekit room=%s", assignment.room_name)

    async def disconnect(self) -> None:
        if self._room is not None:
            await self._room.disconnect()
        self._room = None
        self._video_sources = {}
        self._audio_sources = {}

    async def publish_video(self, output_name: str, frame: np.ndarray) -> None:
        video_source = self._video_sources.get(output_name)
        if video_source is None:
            raise LiveKitUnavailableError(f"video output is not initialized: {output_name}")
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
        video_source.capture_frame(video_frame)
        self._published_frames += 1
        if self._published_frames == 1:
            self._logger.info(
                "published first livekit frame size=%sx%s mean=%.2f std=%.2f",
                rgb_frame.shape[1],
                rgb_frame.shape[0],
                float(rgb_frame.mean()),
                float(rgb_frame.std()),
            )

    async def publish_audio(self, output_name: str, samples: np.ndarray) -> None:
        audio_source = self._audio_sources.get(output_name)
        if audio_source is None:
            raise LiveKitUnavailableError(f"audio output is not initialized: {output_name}")
        spec = self._outputs.get(output_name)
        if spec is None:
            raise LiveKitUnavailableError(f"missing output spec for audio output: {output_name}")
        try:
            from livekit import rtc  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on runtime package
            raise LiveKitUnavailableError("livekit package is missing") from exc
        pcm = np.ascontiguousarray(samples)
        channels = int(spec.config["channels"])
        if pcm.ndim == 1:
            samples_per_channel = int(pcm.shape[0])
        else:
            samples_per_channel = int(pcm.shape[0])
        audio_frame = rtc.AudioFrame(
            data=memoryview(pcm).cast("B"),
            sample_rate=int(spec.config["sample_rate_hz"]),
            num_channels=channels,
            samples_per_channel=samples_per_channel,
        )
        await audio_source.capture_frame(audio_frame)

    async def publish_data(
        self,
        output_name: str,
        payload: bytes,
        *,
        reliable: bool = True,
    ) -> None:
        if self._room is None:
            raise LiveKitUnavailableError("livekit room is not connected")
        await self._room.local_participant.publish_data(
            payload,
            topic=f"wm.output.{output_name}",
            reliable=reliable,
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
            raise LiveKitUnavailableError("livekit room is not connected")
        await self._room.local_participant.publish_data(
            payload,
            topic=self._status_topic,
            reliable=True,
        )


@dataclass(slots=True)
class _FrameItem:
    frame: np.ndarray


class FramePipeline:
    def __init__(self, max_frames: int) -> None:
        self._queue: asyncio.Queue[_FrameItem] = asyncio.Queue(maxsize=max_frames)
        self._dropped_frames = 0
        self._inference_ms: deque[float] = deque(maxlen=128)
        self._published_frames = 0
        self._first_publish_ts: float | None = None

    async def push(self, frame: np.ndarray, *, inference_ms: float) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._dropped_frames += 1
            except asyncio.QueueEmpty:
                pass
        await self._queue.put(_FrameItem(frame=frame))
        self._inference_ms.append(inference_ms)

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    async def pop(self, timeout_s: float) -> np.ndarray | None:
        timeout_s = max(timeout_s, 0.0)
        if timeout_s == 0:
            try:
                return self._queue.get_nowait().frame
            except asyncio.QueueEmpty:
                return None

        try:
            return (await asyncio.wait_for(self._queue.get(), timeout=timeout_s)).frame
        except TimeoutError:
            return None

    async def publish_loop(
        self,
        publish_fn: Callable[[np.ndarray], Awaitable[None]],
        stop_event: asyncio.Event,
        target_fps: int,
    ) -> None:
        frame_period_s = 1.0 / max(target_fps, 1)
        last_publish_started: float | None = None
        while not stop_event.is_set():
            next_frame = await self.pop(timeout_s=0.1)
            if next_frame is None:
                continue
            if last_publish_started is not None:
                sleep_for = frame_period_s - (time.monotonic() - last_publish_started)
                if sleep_for > 0:
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=sleep_for)
                        break
                    except TimeoutError:
                        pass
            publish_started = time.monotonic()
            await publish_fn(next_frame)
            if self._first_publish_ts is None:
                self._first_publish_ts = publish_started
            self._published_frames += 1
            last_publish_started = publish_started

    def metrics(self) -> FrameMetrics:
        if self._first_publish_ts is None:
            effective_fps = 0.0
        else:
            elapsed = max(time.monotonic() - self._first_publish_ts, 1e-6)
            effective_fps = self._published_frames / elapsed
        inference_ms_p50 = (
            statistics.median(self._inference_ms) if self._inference_ms else 0.0
        )
        return FrameMetrics(
            effective_fps=effective_fps,
            queue_depth=self._queue.qsize(),
            inference_ms_p50=inference_ms_p50,
            publish_dropped_frames=self._dropped_frames,
        )


@dataclass(slots=True)
class _VideoOutputState:
    spec: OutputSpec
    pipeline: FramePipeline


class OutputRouter:
    def __init__(
        self,
        *,
        outputs: tuple[OutputSpec, ...],
        livekit: LiveKitAdapter,
        target_fps: int,
        max_queue_frames: int,
        frame_width: int,
        frame_height: int,
    ) -> None:
        self._outputs = {output.name: output for output in outputs}
        self._livekit = livekit
        self._target_fps = target_fps
        self._video_outputs: dict[str, _VideoOutputState] = {}

        for output in outputs:
            if output.kind != "video":
                continue
            self._video_outputs[output.name] = _VideoOutputState(
                spec=output,
                pipeline=FramePipeline(max_queue_frames),
            )

    async def publish(self, output_name: str, payload: Any, ts_ms: int | None = None) -> None:
        _ = ts_ms
        spec = self._outputs[output_name]
        if spec.kind == "video":
            await self._video_outputs[output_name].pipeline.push(payload, inference_ms=0.0)
            return
        if spec.kind == "audio":
            await self._livekit.publish_audio(output_name, payload)
            return
        await self._livekit.publish_data(output_name, payload, reliable=True)

    def start(self, stop_event: asyncio.Event) -> list[asyncio.Task[None]]:
        tasks: list[asyncio.Task[None]] = []
        for output_name, state in self._video_outputs.items():
            fps = min(self._target_fps, int(state.spec.config.get("fps", self._target_fps)))
            task = asyncio.create_task(
                state.pipeline.publish_loop(
                    lambda frame, name=output_name: self._livekit.publish_video(name, frame),
                    stop_event,
                    fps,
                ),
                name=f"publish_loop:{output_name}",
            )
            tasks.append(task)
        return tasks

    def snapshot(self) -> dict[str, float | int]:
        if not self._video_outputs:
            return {
                "effective_fps": 0.0,
                "queue_depth": 0,
                "dropped_frames": 0,
            }
        effective_fps = 0.0
        queue_depth = 0
        dropped_frames = 0
        for state in self._video_outputs.values():
            metrics = state.pipeline.metrics()
            effective_fps += metrics.effective_fps
            queue_depth += metrics.queue_depth
            dropped_frames += metrics.publish_dropped_frames
        return {
            "effective_fps": effective_fps,
            "queue_depth": queue_depth,
            "dropped_frames": dropped_frames,
        }

    def metrics(self, *, inference_ms_p50: float) -> FrameMetrics:
        if not self._video_outputs:
            return FrameMetrics(
                effective_fps=0.0,
                queue_depth=0,
                inference_ms_p50=inference_ms_p50,
                publish_dropped_frames=0,
            )
        effective_fps = 0.0
        queue_depth = 0
        dropped_frames = 0
        for state in self._video_outputs.values():
            metrics = state.pipeline.metrics()
            effective_fps += metrics.effective_fps
            queue_depth += metrics.queue_depth
            dropped_frames += metrics.publish_dropped_frames
        return FrameMetrics(
            effective_fps=effective_fps,
            queue_depth=queue_depth,
            inference_ms_p50=inference_ms_p50,
            publish_dropped_frames=dropped_frames,
        )


@dataclass(slots=True)
class ControlOutcome:
    stop_requested: bool = False
    pong_payload: dict[str, object] | None = None


class SessionControlReducer:
    def __init__(
        self,
        runtime: LucidRuntime,
        session_ctx: SessionContext,
        logger: logging.Logger,
    ) -> None:
        self._runtime = runtime
        self._session_ctx = session_ctx
        self._logger = logger

    async def reduce(self, raw: bytes | str, *, session_id: str) -> ControlOutcome:
        try:
            envelope = parse_control_message(raw)
        except ProtocolError as exc:
            self._logger.warning("invalid control payload: %s", exc)
            return ControlOutcome()

        if envelope.session_id and envelope.session_id != session_id:
            return ControlOutcome()
        if envelope.type == ControlMessageType.END:
            return ControlOutcome(stop_requested=True)
        if envelope.type == ControlMessageType.PING:
            return ControlOutcome(
                pong_payload={"client_ts_ms": envelope.payload.get("client_ts_ms")}
            )
        if envelope.type != ControlMessageType.ACTION:
            return ControlOutcome()

        payload = envelope.payload
        name = str(payload.get("name", "")).strip()
        args = payload.get("args", {})
        if not name or not isinstance(args, dict):
            self._logger.warning("invalid action payload name=%s args_type=%s", name, type(args))
            return ControlOutcome()
        try:
            await self._runtime.dispatch_action(self._session_ctx, name, args)
        except ActionDispatchError as exc:
            self._logger.warning("failed dispatching action %s: %s", name, exc)
        return ControlOutcome()


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
