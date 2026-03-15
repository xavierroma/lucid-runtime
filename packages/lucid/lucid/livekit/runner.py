from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import inspect
import json
import logging
import shutil
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Protocol, TypeAlias, TypedDict, TypeVar

import numpy as np

from ..controlplane import SessionLifecycleReporter
from ..core.input_file import InputFile
from ..core.model import LucidError, ManifestDict, MetricsSnapshot, NormalizedOutput
from ..core.runtime import ActionDispatchError, LucidRuntime, ModelConfigInput, RuntimeSession
from ..core.spec import ModelTarget, OutputBinding, OutputSpec, resolve_model_definition
from .config import Assignment, RuntimeConfig, SessionConfig, SessionResult


DEFAULT_CONTROL_TOPIC = "wm.control"
DEFAULT_STATUS_TOPIC = "wm.status"
DEFAULT_INPUT_FILE_TOPIC = "wm.input.file"

LifecycleHook: TypeAlias = Callable[[str], Awaitable[None] | None]
StatusValue: TypeAlias = str | int | float | bool | None
StatusPayload: TypeAlias = dict[str, StatusValue]
StatusSender: TypeAlias = Callable[[str, StatusPayload], Awaitable[None]]
ControlPayload: TypeAlias = dict[str, object]
InputFileSlot: TypeAlias = tuple[str, str]
T = TypeVar("T")


class CapabilitiesPayload(TypedDict):
    control_topic: str
    status_topic: str
    manifest: ManifestDict
    output_bindings: list[OutputBinding]


class LiveKitTransport(Protocol):
    async def connect(self, assignment: Assignment, outputs: tuple[OutputSpec, ...]) -> None: ...
    async def disconnect(self) -> None: ...
    def set_status_sender(self, sender: StatusSender | None) -> None: ...
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
    def resolve_input_file(self, file_id: str) -> InputFile | None: ...


class LiveKitUnavailableError(RuntimeError):
    pass


def capabilities(
    *,
    control_topic: str = DEFAULT_CONTROL_TOPIC,
    status_topic: str = DEFAULT_STATUS_TOPIC,
    model: ModelTarget,
) -> CapabilitiesPayload:
    definition = resolve_model_definition(model)
    return {
        "control_topic": control_topic,
        "status_topic": status_topic,
        "manifest": definition.to_manifest(),
        "output_bindings": definition.output_bindings(),
    }


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
            "video": {"roomJoin": True, "room": room_name},
        },
        api_secret,
    )


@dataclass(slots=True)
class _VideoState:
    fps: int
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    published_frames: int = 0
    first_publish_ts: float | None = None
    last_publish_started: float | None = None

    async def publish(self, frame: np.ndarray, publish_fn: Callable[[np.ndarray], Awaitable[None]]) -> None:
        async with self.lock:
            if self.last_publish_started is not None:
                delay = (1.0 / max(self.fps, 1)) - (time.monotonic() - self.last_publish_started)
                if delay > 0:
                    await asyncio.sleep(delay)
            publish_started = time.monotonic()
            await publish_fn(frame)
            if self.first_publish_ts is None:
                self.first_publish_ts = publish_started
            self.last_publish_started = publish_started
            self.published_frames += 1

    def effective_fps(self) -> float:
        if self.first_publish_ts is None:
            return 0.0
        return self.published_frames / max(time.monotonic() - self.first_publish_ts, 1e-6)


class _OutputSink:
    def __init__(self, outputs: tuple[OutputSpec, ...], livekit: LiveKitTransport) -> None:
        self._outputs = {output.name: output for output in outputs}
        self._livekit = livekit
        self._video = {
            output.name: _VideoState(fps=int(output.config["fps"]))
            for output in outputs
            if output.kind == "video"
        }

    async def publish(
        self,
        output_name: str,
        payload: NormalizedOutput,
        ts_ms: int | None = None,
    ) -> None:
        _ = ts_ms
        spec = self._outputs[output_name]
        if spec.kind == "video":
            await self._video[output_name].publish(
                payload,
                lambda frame, name=output_name: self._livekit.publish_video(name, frame),
            )
            return
        if spec.kind == "audio":
            await self._livekit.publish_audio(output_name, payload)
            return
        await self._livekit.publish_data(output_name, payload, reliable=True)

    def snapshot(self) -> MetricsSnapshot:
        return {"effective_fps": sum(state.effective_fps() for state in self._video.values())}


@dataclass(slots=True)
class _ControlMessage:
    kind: str
    seq: int
    ts_ms: int
    session_id: str | None
    payload: ControlPayload


@dataclass(slots=True)
class _ControlResult:
    stop_requested: bool = False
    pause_requested: bool = False
    resume_requested: bool = False
    pong_payload: dict[str, object] | None = None


class _ProtocolError(ValueError):
    pass


class _StatusChannel:
    def __init__(
        self,
        *,
        livekit: LiveKitTransport,
        session_id: str | None,
        logger: logging.Logger,
    ) -> None:
        self._livekit = livekit
        self._session_id = session_id
        self._logger = logger
        self._seq = 0

    async def started(self, worker_id: str) -> None:
        await self.send("started", {"worker_id": worker_id})

    async def pong(self, payload: StatusPayload) -> None:
        await self.send("pong", payload)

    async def frame_metrics(self, *, effective_fps: float, inference_ms_p50: float) -> None:
        await self.send(
            "frame_metrics",
            {
                "effective_fps": round(effective_fps, 3),
                "inference_ms_p50": round(inference_ms_p50, 3),
            },
        )

    async def error(self, error_code: str) -> None:
        await self.send("error", {"error_code": error_code})

    async def ended(self, *, ended_by_control: bool) -> None:
        await self.send("ended", {"ended_by_control": ended_by_control})

    async def send(self, message_type: str, payload: StatusPayload) -> None:
        self._seq += 1
        try:
            await self._livekit.send_status(
                _encode_status_message(
                    message_type,
                    session_id=self._session_id,
                    seq=self._seq,
                    payload=payload,
                )
            )
        except Exception as exc:  # pragma: no cover - transport boundary
            self._logger.warning("failed sending status message: %s", exc)


class _RealLiveKitTransport:
    def __init__(
        self,
        *,
        livekit_url: str,
        status_topic: str,
        logger: logging.Logger,
    ) -> None:
        self._livekit_url = livekit_url
        self._status_topic = status_topic
        self._logger = logger
        self._room = None
        self._control_topic = DEFAULT_CONTROL_TOPIC
        self._control_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._video_sources: dict[str, object] = {}
        self._audio_sources: dict[str, object] = {}
        self._outputs: dict[str, OutputSpec] = {}
        self._published_frames = 0
        self._input_files: dict[str, InputFile] = {}
        self._input_file_slots: dict[str, InputFileSlot] = {}
        self._latest_upload_by_slot: dict[InputFileSlot, str] = {}
        self._active_upload_by_slot: dict[InputFileSlot, str] = {}
        self._upload_tasks: set[asyncio.Task[None]] = set()
        self._upload_dir: Path | None = None
        self._status_sender: StatusSender | None = None

    def set_status_sender(self, sender: StatusSender | None) -> None:
        self._status_sender = sender

    async def connect(self, assignment: Assignment, outputs: tuple[OutputSpec, ...]) -> None:
        rtc = _load_livekit_rtc()
        self._control_topic = assignment.control_topic
        self._outputs = {output.name: output for output in outputs}
        self._upload_dir = Path(tempfile.mkdtemp(prefix=f"lucid-upload-{assignment.session_id}-"))
        self._room = rtc.Room()

        @self._room.on("data_received")
        def _on_data(data_packet: object) -> None:  # pragma: no cover - SDK callback
            payload = getattr(data_packet, "data", None)
            topic = getattr(data_packet, "topic", "") or ""
            if payload is None or (topic and topic != self._control_topic):
                return
            try:
                self._control_queue.put_nowait(bytes(payload))
            except asyncio.QueueFull:
                self._logger.warning("dropping control message because queue is full")

        await self._room.connect(self._livekit_url, assignment.worker_access_token)
        register_byte_stream_handler = getattr(self._room, "register_byte_stream_handler", None)
        if register_byte_stream_handler is None:
            raise LiveKitUnavailableError(
                "livekit byte streams require a newer livekit package; install lucid[livekit]"
            )

        def _on_input_file_stream(reader: object, participant_identity: str) -> None:
            _ = participant_identity
            task = asyncio.create_task(
                self._consume_input_file_stream(reader, assignment.session_id),
                name=f"{assignment.session_id}:upload:{getattr(getattr(reader, 'info', None), 'stream_id', 'unknown')}",
            )
            self._upload_tasks.add(task)
            task.add_done_callback(self._upload_tasks.discard)

        register_byte_stream_handler(DEFAULT_INPUT_FILE_TOPIC, _on_input_file_stream)
        for output in outputs:
            if output.kind == "video":
                source = rtc.VideoSource(int(output.config["width"]), int(output.config["height"]))
                track = rtc.LocalVideoTrack.create_video_track(output.name, source)
                options = _build_track_publish_options(rtc)
                if options is None:
                    await self._room.local_participant.publish_track(track)
                else:
                    await self._room.local_participant.publish_track(track, options)
                self._video_sources[output.name] = source
                continue
            if output.kind == "audio":
                source = rtc.AudioSource(
                    int(output.config["sample_rate_hz"]),
                    int(output.config["channels"]),
                )
                track = rtc.LocalAudioTrack.create_audio_track(output.name, source)
                await self._room.local_participant.publish_track(track)
                self._audio_sources[output.name] = source
        self._logger.info("connected to livekit room=%s", assignment.room_name)

    async def disconnect(self) -> None:
        room = self._room
        if room is not None:
            unregister_byte_stream_handler = getattr(room, "unregister_byte_stream_handler", None)
            if unregister_byte_stream_handler is not None:
                try:
                    unregister_byte_stream_handler(DEFAULT_INPUT_FILE_TOPIC)
                except Exception:
                    self._logger.warning("failed to unregister input file stream handler", exc_info=True)
        if self._upload_tasks:
            for task in tuple(self._upload_tasks):
                task.cancel()
            await asyncio.gather(*self._upload_tasks, return_exceptions=True)
            self._upload_tasks.clear()
        if self._room is not None:
            await self._room.disconnect()
        self._room = None
        self._video_sources.clear()
        self._audio_sources.clear()
        self._outputs.clear()
        self._control_queue = asyncio.Queue()
        self._cleanup_input_files()

    async def publish_video(self, output_name: str, frame: np.ndarray) -> None:
        source = self._video_sources.get(output_name)
        if source is None:
            raise LiveKitUnavailableError(f"video output is not initialized: {output_name}")
        rtc = _load_livekit_rtc("livekit package is missing")
        source.capture_frame(
            rtc.VideoFrame(
                width=frame.shape[1],
                height=frame.shape[0],
                type=rtc.VideoBufferType.RGB24,
                data=memoryview(frame).cast("B"),
            )
        )
        self._published_frames += 1
        if self._published_frames == 1:
            self._logger.info(
                "published first livekit frame size=%sx%s mean=%.2f std=%.2f",
                frame.shape[1],
                frame.shape[0],
                float(frame.mean()),
                float(frame.std()),
            )

    async def publish_audio(self, output_name: str, samples: np.ndarray) -> None:
        source = self._audio_sources.get(output_name)
        if source is None:
            raise LiveKitUnavailableError(f"audio output is not initialized: {output_name}")
        spec = self._outputs.get(output_name)
        if spec is None:
            raise LiveKitUnavailableError(f"missing output spec for audio output: {output_name}")
        rtc = _load_livekit_rtc("livekit package is missing")
        pcm = np.ascontiguousarray(samples)
        await _maybe_await(
            source.capture_frame(
                rtc.AudioFrame(
                    data=memoryview(pcm).cast("B"),
                    sample_rate=int(spec.config["sample_rate_hz"]),
                    num_channels=int(spec.config["channels"]),
                    samples_per_channel=int(pcm.shape[0]),
                )
            )
        )

    async def publish_data(
        self,
        output_name: str,
        payload: bytes,
        *,
        reliable: bool = True,
    ) -> None:
        room = self._require_room()
        await room.local_participant.publish_data(
            payload,
            topic=f"wm.output.{output_name}",
            reliable=reliable,
        )

    async def recv_control(self, timeout_s: float) -> bytes | None:
        return await _recv_from_queue(self._control_queue, timeout_s)

    async def send_status(self, payload: bytes) -> None:
        room = self._require_room()
        await room.local_participant.publish_data(
            payload,
            topic=self._status_topic,
            reliable=True,
        )

    def resolve_input_file(self, file_id: str) -> InputFile | None:
        input_file = self._input_files.get(file_id)
        if input_file is None:
            return None
        slot = self._input_file_slots.get(file_id, ("", ""))
        previous_file_id = self._active_upload_by_slot.get(slot)
        if previous_file_id and previous_file_id != file_id:
            self._drop_input_file(previous_file_id)
        self._active_upload_by_slot[slot] = file_id
        self._latest_upload_by_slot[slot] = file_id
        return input_file

    def _require_room(self) -> object:
        if self._room is None:
            raise LiveKitUnavailableError("livekit room is not connected")
        return self._room

    async def _consume_input_file_stream(self, reader: object, session_id: str) -> None:
        info = getattr(reader, "info", None)
        stream_id = str(getattr(info, "stream_id", "") or "").strip()
        if not stream_id:
            self._logger.warning("dropping input file stream without a stream_id")
            return
        attributes = _coerce_stream_attributes(getattr(info, "attributes", None))
        expected_session_id = str(attributes.get("session_id", "") or "").strip()
        if expected_session_id and expected_session_id != session_id:
            self._logger.warning(
                "dropping input file stream with mismatched session_id expected=%s got=%s stream_id=%s",
                session_id,
                expected_session_id,
                stream_id,
            )
            return
        upload_path = self._build_upload_path(
            stream_id=stream_id,
            filename=str(getattr(info, "name", "upload.bin") or "upload.bin"),
        )
        total_size = 0
        digest = hashlib.sha256()
        try:
            with upload_path.open("wb") as handle:
                async for chunk in reader:
                    handle.write(chunk)
                    total_size += len(chunk)
                    digest.update(chunk)
            expected_size = getattr(info, "size", None)
            if expected_size is not None and int(expected_size) != total_size:
                raise ValueError(f"size mismatch for {stream_id}: {total_size} != {expected_size}")
            expected_sha256 = str(attributes.get("sha256", "") or "").strip().lower()
            actual_sha256 = digest.hexdigest()
            if expected_sha256 and expected_sha256 != actual_sha256:
                raise ValueError(f"sha256 mismatch for {stream_id}")
            slot = _coerce_input_file_slot(attributes)
            previous_upload_id = self._latest_upload_by_slot.get(slot)
            if previous_upload_id and previous_upload_id != self._active_upload_by_slot.get(slot):
                self._drop_input_file(previous_upload_id)
            self._input_files[stream_id] = InputFile(
                id=stream_id,
                filename=str(getattr(info, "name", "upload.bin") or "upload.bin"),
                mime_type=str(getattr(info, "mime_type", "application/octet-stream") or "application/octet-stream"),
                size_bytes=total_size,
                sha256=actual_sha256,
                path=upload_path,
            )
            self._input_file_slots[stream_id] = slot
            self._latest_upload_by_slot[slot] = stream_id
            await self._emit_upload_status("upload_ready", {"upload_id": stream_id})
        except asyncio.CancelledError:
            self._delete_path(upload_path)
            raise
        except Exception:
            self._delete_path(upload_path)
            self._logger.warning("failed staging input file stream_id=%s", stream_id, exc_info=True)
            await self._emit_upload_status(
                "upload_error",
                {"upload_id": stream_id, "error_code": "UPLOAD_FAILED"},
            )

    def _build_upload_path(self, *, stream_id: str, filename: str) -> Path:
        base = self._upload_dir
        if base is None:
            raise LiveKitUnavailableError("livekit upload directory is not initialized")
        safe_name = Path(filename).name or "upload.bin"
        return base / f"{stream_id}-{safe_name}"

    async def _emit_upload_status(self, message_type: str, payload: StatusPayload) -> None:
        if self._status_sender is None:
            return
        try:
            await self._status_sender(message_type, payload)
        except Exception:
            self._logger.warning("failed sending upload status message_type=%s", message_type, exc_info=True)

    def _cleanup_input_files(self) -> None:
        for file_id in tuple(self._input_files):
            self._drop_input_file(file_id)
        if self._upload_dir is not None:
            shutil.rmtree(self._upload_dir, ignore_errors=True)
            self._upload_dir = None

    def _drop_input_file(self, file_id: str) -> None:
        input_file = self._input_files.pop(file_id, None)
        slot = self._input_file_slots.pop(file_id, None)
        if slot is not None:
            if self._latest_upload_by_slot.get(slot) == file_id:
                self._latest_upload_by_slot.pop(slot, None)
            if self._active_upload_by_slot.get(slot) == file_id:
                self._active_upload_by_slot.pop(slot, None)
        if input_file is not None:
            self._delete_path(input_file.path)

    @staticmethod
    def _delete_path(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return


@dataclass(slots=True)
class _RunState:
    session_id: str
    runtime_session: RuntimeSession
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    resume_event: asyncio.Event = field(default_factory=asyncio.Event)
    result: SessionResult = field(default_factory=SessionResult)
    phase: str = "ready"

    def stop(self) -> None:
        self.runtime_session.ctx.running = False
        self.runtime_session.ctx.resume()
        self.stop_event.set()


class SessionRunner:
    CONTROL_POLL_SECS = 0.5
    HEARTBEAT_INTERVAL_SECS = 2.0
    METRICS_INTERVAL_SECS = 1.0

    def __init__(
        self,
        host_config: RuntimeConfig,
        session_config: SessionConfig | None,
        logger: logging.Logger,
        *,
        model: ModelTarget,
        model_config: ModelConfigInput = None,
        reporter: SessionLifecycleReporter | None = None,
        livekit_factory: Callable[[], LiveKitTransport] | None = None,
        on_ready: LifecycleHook | None = None,
        on_running: LifecycleHook | None = None,
        runtime: LucidRuntime | None = None,
    ) -> None:
        self._host_config = host_config
        self._session_config = session_config
        self._logger = logger
        self._reporter = reporter
        self._runtime = runtime or LucidRuntime.load_model(
            runtime_config=host_config,
            logger=logger,
            model=model,
            config=model_config,
        )
        self._owns_runtime = runtime is None
        self._livekit_factory = livekit_factory
        self._on_ready = on_ready
        self._on_running = on_running
        self._active_runs: dict[str, _RunState] = {}
        self._loaded = False

    @property
    def manifest(self) -> ManifestDict:
        return self._runtime.manifest()

    @property
    def output_bindings(self) -> list[OutputBinding]:
        return self._runtime.output_bindings()

    async def load(self) -> None:
        if self._loaded:
            return
        start = perf_counter()
        try:
            await self._runtime.load()
        except Exception as exc:
            self._logger.error(
                "session_runner.load failed duration_ms=%.1f model=%s error_type=%s",
                (perf_counter() - start) * 1000.0,
                self._runtime.definition.name,
                exc.__class__.__name__,
            )
            raise
        self._loaded = True
        self._logger.info(
            "session_runner.load complete duration_ms=%.1f model=%s",
            (perf_counter() - start) * 1000.0,
            self._runtime.definition.name,
        )

    async def close(self) -> None:
        if self._reporter is not None:
            await self._reporter.close()
        if self._owns_runtime:
            await self._runtime.unload()

    def stop(self) -> None:
        for state in list(self._active_runs.values()):
            state.stop()

    async def run_session(self, assignment: Assignment) -> SessionResult:
        if not self._loaded:
            await self.load()

        livekit = self._build_livekit_transport()
        output_sink = _OutputSink(self._runtime.outputs, livekit)
        runtime_session = self._runtime.open_session(
            session_id=assignment.session_id,
            room_name=assignment.room_name,
            publish_fn=output_sink.publish,
            metrics_fn=output_sink.snapshot,
            input_file_resolver=livekit.resolve_input_file,
        )
        state = _RunState(session_id=assignment.session_id, runtime_session=runtime_session)
        self._active_runs[assignment.session_id] = state
        status = _StatusChannel(
            livekit=livekit,
            session_id=assignment.session_id,
            logger=self._logger,
        )
        livekit.set_status_sender(status.send)
        model_task = asyncio.create_task(
            self._model_loop(state, assignment.session_id),
            name=f"{assignment.session_id}:model",
        )

        start = perf_counter()
        self._logger.info(
            "starting session session_id=%s room_name=%s model=%s",
            assignment.session_id,
            assignment.room_name,
            self._runtime.definition.name,
        )
        try:
            await livekit.connect(assignment, self._runtime.outputs)
            await status.started(
                self._session_config.worker_id if self._session_config is not None else "lucid-research"
            )
            if self._reporter is not None:
                await self._reporter.ready(assignment.session_id)
            await self._emit_lifecycle_hook(self._on_ready, assignment.session_id)
            await self._control_and_metrics_loop(
                state,
                livekit=livekit,
                status=status,
                output_sink=output_sink,
                model_task=model_task,
            )
        except Exception as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            if isinstance(exc, (ActionDispatchError, LucidError)):
                state.result.error_code = state.result.error_code or "MODEL_RUNTIME_ERROR"
                self._logger.exception("lucid runtime error: %s", exc)
            elif isinstance(exc, RuntimeError) and not self._is_modal_input_cancellation(exc):
                state.result.error_code = state.result.error_code or "LIVEKIT_DISCONNECT"
                self._logger.exception("session failed: %s", exc)
            elif not self._is_modal_input_cancellation(exc):
                state.result.error_code = state.result.error_code or "LIVEKIT_DISCONNECT"
                self._logger.exception("session failed: %s", exc)
        finally:
            state.stop()
            self._active_runs.pop(assignment.session_id, None)
            livekit.set_status_sender(None)
            if not model_task.done():
                model_task.cancel()
            await asyncio.gather(model_task, return_exceptions=True)
            self._consume_task_result(model_task, state.result, cancelled_is_error=False)
            await runtime_session.close()
            await self._emit_terminal_status(status, state.result)
            await livekit.disconnect()
            if self._reporter is not None:
                try:
                    await self._reporter.ended(
                        assignment.session_id,
                        state.result.error_code,
                        _end_reason(state.result),
                    )
                except Exception as exc:
                    self._logger.warning("failed to mark session ended: %s", exc)
            self._logger.info(
                "session_runner.run_session complete duration_ms=%.1f session_id=%s error_code=%s ended_by_control=%s",
                (perf_counter() - start) * 1000.0,
                assignment.session_id,
                state.result.error_code,
                state.result.ended_by_control,
            )
        return state.result

    def _build_livekit_transport(self) -> LiveKitTransport:
        if self._livekit_factory is not None:
            return self._livekit_factory()
        return _RealLiveKitTransport(
            livekit_url=self._host_config.livekit_url,
            status_topic=self._host_config.status_topic,
            logger=self._logger,
        )

    async def _control_and_metrics_loop(
        self,
        state: _RunState,
        *,
        livekit: LiveKitTransport,
        status: _StatusChannel,
        output_sink: _OutputSink,
        model_task: asyncio.Task[None],
    ) -> None:
        next_metrics = time.monotonic() + self.METRICS_INTERVAL_SECS
        next_heartbeat = time.monotonic() + self.HEARTBEAT_INTERVAL_SECS

        while not state.stop_event.is_set():
            if model_task.done():
                self._consume_task_result(model_task, state.result)
                state.stop()
                return

            now = time.monotonic()
            deadlines = [self.CONTROL_POLL_SECS, max(next_metrics - now, 0.0)]
            if self._reporter is not None:
                deadlines.append(max(next_heartbeat - now, 0.0))
            raw = await livekit.recv_control(timeout_s=min(deadlines))
            if raw is not None:
                outcome = await _reduce_control_message(
                    runtime_session=state.runtime_session,
                    logger=self._logger,
                    raw=raw,
                    session_id=state.session_id,
                )
                if outcome.stop_requested:
                    state.result.ended_by_control = True
                    state.stop()
                    continue
                if outcome.pause_requested:
                    await self._pause(state)
                    continue
                if outcome.resume_requested:
                    await self._resume(state)
                    continue
                if outcome.pong_payload is not None:
                    await status.pong(outcome.pong_payload)

            now = time.monotonic()
            if now >= next_metrics:
                await status.frame_metrics(
                    effective_fps=float(output_sink.snapshot()["effective_fps"]),
                    inference_ms_p50=state.runtime_session.ctx.inference_ms_p50(),
                )
                next_metrics = now + self.METRICS_INTERVAL_SECS
            if self._reporter is not None and now >= next_heartbeat:
                try:
                    await self._reporter.heartbeat(state.session_id)
                except Exception as exc:
                    self._logger.warning("failed sending session lifecycle heartbeat: %s", exc)
                next_heartbeat = now + self.HEARTBEAT_INTERVAL_SECS

    async def _model_loop(self, state: _RunState, session_id: str) -> None:
        if not await self._wait_for_initial_resume(state):
            return
        if self._reporter is not None:
            await self._reporter.running(session_id)
        state.phase = "running"
        await self._emit_lifecycle_hook(self._on_running, session_id)
        await state.runtime_session.run()
        state.stop_event.set()

    async def _wait_for_initial_resume(self, state: _RunState) -> bool:
        wait_task = asyncio.create_task(state.resume_event.wait(), name=f"{state.session_id}:resume")
        stop_task = asyncio.create_task(state.stop_event.wait(), name=f"{state.session_id}:resume-stop")
        try:
            done, _ = await asyncio.wait([wait_task, stop_task], return_when=asyncio.FIRST_COMPLETED)
            return stop_task not in done
        finally:
            wait_task.cancel()
            stop_task.cancel()
            await asyncio.gather(wait_task, stop_task, return_exceptions=True)

    async def _pause(self, state: _RunState) -> None:
        if state.phase != "running":
            return
        if not state.runtime_session.ctx.pause():
            return
        state.phase = "paused"
        if self._reporter is not None:
            await self._reporter.paused(state.session_id)

    async def _resume(self, state: _RunState) -> None:
        if state.phase == "ready":
            state.phase = "starting"
            state.resume_event.set()
            return
        if state.phase != "paused":
            return
        if not state.runtime_session.ctx.resume():
            return
        state.phase = "running"
        if self._reporter is not None:
            await self._reporter.running(state.session_id)

    @staticmethod
    async def _emit_lifecycle_hook(hook: LifecycleHook | None, session_id: str) -> None:
        if hook is None:
            return
        result = hook(session_id)
        if inspect.isawaitable(result):
            await result

    async def _emit_terminal_status(
        self,
        status: _StatusChannel,
        result: SessionResult,
    ) -> None:
        if result.error_code:
            await status.error(result.error_code)
        await status.ended(ended_by_control=result.ended_by_control)

    def _consume_task_result(
        self,
        task: asyncio.Task[None],
        result: SessionResult,
        *,
        cancelled_is_error: bool = True,
    ) -> None:
        if task.cancelled():
            if cancelled_is_error:
                result.error_code = result.error_code or "WORKER_TASK_ERROR"
                self._logger.error("session task cancelled unexpectedly task=%s", task.get_name())
            return
        exc = task.exception()
        if exc is None:
            return
        if isinstance(exc, LucidError):
            result.error_code = result.error_code or "MODEL_RUNTIME_ERROR"
        else:
            result.error_code = result.error_code or "WORKER_TASK_ERROR"
        self._logger.exception("session task failed task=%s error=%s", task.get_name(), exc)

    @staticmethod
    def _is_modal_input_cancellation(exc: BaseException) -> bool:
        return (
            exc.__class__.__module__ == "modal.exception"
            and exc.__class__.__name__ == "InputCancellation"
        )


async def _reduce_control_message(
    *,
    runtime_session: RuntimeSession,
    logger: logging.Logger,
    raw: bytes | str,
    session_id: str,
) -> _ControlResult:
    try:
        message = _parse_control_message(raw)
    except _ProtocolError as exc:
        logger.warning("invalid control payload: %s", exc)
        return _ControlResult()

    if message.session_id and message.session_id != session_id:
        return _ControlResult()
    if message.kind == "end":
        return _ControlResult(stop_requested=True)
    if message.kind == "pause":
        return _ControlResult(pause_requested=True)
    if message.kind == "resume":
        return _ControlResult(resume_requested=True)
    if message.kind == "ping":
        return _ControlResult(pong_payload={"client_ts_ms": message.payload.get("client_ts_ms")})
    if message.kind != "action":
        return _ControlResult()

    name = str(message.payload.get("name", "")).strip()
    args = message.payload.get("args", {})
    if not name or not isinstance(args, dict):
        logger.warning(
            "invalid action payload name=%s args_type=%s",
            name,
            type(args),
        )
        return _ControlResult()
    if runtime_session.ctx.is_paused() and not runtime_session.allows_input_while_paused(name):
        logger.debug("dropping paused action name=%s", name)
        return _ControlResult()
    try:
        await runtime_session.dispatch_input(name, args)
    except ActionDispatchError as exc:
        logger.warning("failed dispatching action %s: %s", name, exc)
    return _ControlResult()


def _parse_control_message(raw: bytes | str) -> _ControlMessage:
    try:
        payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _ProtocolError(f"invalid control JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise _ProtocolError("invalid control envelope: expected object")

    kind = str(payload.get("type", "")).strip()
    if kind not in {"action", "end", "ping", "pause", "resume"}:
        raise _ProtocolError(
            "invalid control envelope: type must be one of ['action', 'end', 'ping', 'pause', 'resume']"
        )
    seq = _require_non_negative_int(payload.get("seq"), "seq")
    ts_ms = _require_non_negative_int(payload.get("ts_ms"), "ts_ms")
    session_id = payload.get("session_id")
    if session_id is not None and not isinstance(session_id, str):
        raise _ProtocolError("invalid control envelope: session_id must be a string or null")
    message_payload = payload.get("payload", {})
    if not isinstance(message_payload, dict):
        raise _ProtocolError("invalid control envelope: payload must be an object")
    return _ControlMessage(
        kind=kind,
        seq=seq,
        ts_ms=ts_ms,
        session_id=session_id,
        payload=message_payload,
    )


def _encode_status_message(
    message_type: str,
    *,
    session_id: str | None,
    seq: int,
    payload: StatusPayload,
) -> bytes:
    return json.dumps(
        {
            "type": message_type,
            "seq": seq,
            "ts_ms": int(time.time() * 1000),
            "session_id": session_id,
            "payload": payload,
        },
        separators=(",", ":"),
        sort_keys=False,
    ).encode("utf-8")


def _coerce_stream_attributes(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    attributes: dict[str, str] = {}
    for key, item in value.items():
        if isinstance(key, str) and isinstance(item, str):
            attributes[key] = item
    return attributes


def _coerce_input_file_slot(attributes: Mapping[str, str]) -> InputFileSlot:
    input_name = attributes.get("input_name", "").strip()
    arg_name = attributes.get("arg_name", "").strip()
    return (input_name, arg_name) if input_name and arg_name else ("", "")


def _encode_jwt(payload: ManifestDict, secret: str) -> str:
    def _b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode("utf-8"))
    body = _b64url(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signature = hmac.new(
        secret.encode("utf-8"),
        f"{header}.{body}".encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{header}.{body}.{_b64url(signature)}"


def _require_non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _ProtocolError(
            f"invalid control envelope: {field_name} must be a non-negative integer"
        )
    return value


def _build_track_publish_options(rtc: object) -> object | None:
    try:
        return rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_CAMERA,
            simulcast=True,
        )
    except Exception:
        return None


def _load_livekit_rtc(message: str = "livekit package is missing; install lucid[livekit]") -> object:
    try:
        from livekit import rtc  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise LiveKitUnavailableError(message) from exc
    return rtc


async def _recv_from_queue(queue: asyncio.Queue[bytes], timeout_s: float) -> bytes | None:
    timeout_s = max(timeout_s, 0.0)
    if timeout_s == 0:
        try:
            return queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
    try:
        return await asyncio.wait_for(queue.get(), timeout=timeout_s)
    except TimeoutError:
        return None


async def _maybe_await(value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value


def _end_reason(result: SessionResult) -> str | None:
    if result.error_code:
        return "WORKER_REPORTED_ERROR"
    if result.ended_by_control:
        return "CONTROL_REQUESTED"
    return None
