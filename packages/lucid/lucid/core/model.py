from __future__ import annotations

import asyncio
import json
import logging
import statistics
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, TypeAdapter

if TYPE_CHECKING:
    from .spec import OutputSpec


class LucidError(RuntimeError):
    pass


class OutputValidationError(LucidError):
    pass


class _EmptyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


TConfig = TypeVar("TConfig", bound=BaseModel)
TModel = TypeVar("TModel", bound="LucidModel[Any]")
JsonValue: TypeAlias = dict[str, object] | list[object] | str | int | float | bool | None
ManifestDict: TypeAlias = dict[str, JsonValue]
MetricsSnapshot: TypeAlias = dict[str, float | int]
NormalizedOutput: TypeAlias = np.ndarray | bytes
PublishSample: TypeAlias = BaseModel | JsonValue | bytes | bytearray | memoryview | np.ndarray
PublishFn: TypeAlias = Callable[[str, NormalizedOutput, int | None], Awaitable[None]]
MetricsFn: TypeAlias = Callable[[], MetricsSnapshot]


@dataclass(frozen=True, slots=True)
class LoadContext:
    config: BaseModel
    logger: logging.Logger


class SessionContext:
    def __init__(
        self,
        *,
        session_id: str,
        room_name: str,
        outputs: tuple["OutputSpec", ...],
        publish_fn: PublishFn,
        logger: logging.Logger,
        metrics_fn: MetricsFn | None = None,
    ) -> None:
        self.session_id = session_id
        self.room_name = room_name
        self.logger = logger
        self.running = True
        self._outputs = {output.name: output for output in outputs}
        self._publish_fn = publish_fn
        self._metrics_fn = metrics_fn
        self._inference_ms: deque[float] = deque(maxlen=128)
        self._resume_event = asyncio.Event()
        self._resume_event.set()
        self._paused = False

    def record_inference_ms(self, value: float) -> None:
        self._inference_ms.append(float(value))

    def inference_ms_p50(self) -> float:
        if not self._inference_ms:
            return 0.0
        return float(statistics.median(self._inference_ms))

    def output_metrics(self) -> MetricsSnapshot:
        if self._metrics_fn is None:
            return {"effective_fps": 0.0}
        metrics = dict(self._metrics_fn())
        metrics.setdefault("effective_fps", 0.0)
        return metrics

    async def publish(self, output_name: str, sample: PublishSample, ts_ms: int | None = None) -> None:
        spec = self._outputs.get(output_name)
        if spec is None:
            raise OutputValidationError(f"unknown output: {output_name}")
        await self._publish_fn(output_name, _normalize_output(spec, sample), ts_ms)

    def pause(self) -> bool:
        if self._paused:
            return False
        self._paused = True
        self._resume_event.clear()
        return True

    def resume(self) -> bool:
        if not self._paused:
            return False
        self._paused = False
        self._resume_event.set()
        return True

    def is_paused(self) -> bool:
        return self._paused

    async def wait_if_paused(self) -> None:
        while self.running and self._paused:
            await self._resume_event.wait()


class LucidModel(Generic[TConfig]):
    name = "lucid-model"
    description: str | None = None
    config_cls: type[BaseModel] = _EmptyConfig
    outputs: tuple["OutputSpec", ...] = ()

    def __init__(self, config: TConfig) -> None:
        self.config = config
        self.runtime_config: object | None = None
        self.logger: logging.Logger | None = None

    def bind_runtime(self, runtime_config: object, logger: logging.Logger) -> None:
        self.runtime_config = runtime_config
        self.logger = logger

    async def load(self, ctx: LoadContext) -> None:
        _ = ctx

    async def unload(self) -> None:
        return None

    def create_session(
        self,
        ctx: SessionContext,
    ) -> "LucidSession[LucidModel[TConfig]]":  # pragma: no cover - interface
        raise NotImplementedError


class LucidSession(Generic[TModel]):
    def __init__(self, model: TModel, ctx: SessionContext) -> None:
        self.model = model
        self.ctx = ctx

    async def run(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def close(self) -> None:
        return None


def _normalize_output(spec: "OutputSpec", sample: PublishSample) -> NormalizedOutput:
    if spec.kind == "video":
        if not isinstance(sample, np.ndarray):
            raise OutputValidationError(f"{spec.name} expects numpy.ndarray video frames")
        if sample.dtype != np.uint8:
            raise OutputValidationError(f"{spec.name} expects uint8 frames")
        expected_shape = (spec.config["height"], spec.config["width"], 3)
        if sample.shape != expected_shape:
            raise OutputValidationError(
                f"{spec.name} expects frame shape {expected_shape}, got {sample.shape}"
            )
        if not sample.flags.c_contiguous:
            raise OutputValidationError(f"{spec.name} expects C-contiguous video frames")
        if spec.config.get("pixel_format") != "rgb24":
            raise OutputValidationError("only rgb24 video outputs are supported")
        return sample

    if spec.kind == "audio":
        if not isinstance(sample, np.ndarray):
            raise OutputValidationError(f"{spec.name} expects numpy.ndarray audio samples")
        if sample.dtype not in {np.float32, np.int16}:
            raise OutputValidationError(f"{spec.name} expects float32 or int16 audio samples")
        if sample.ndim not in {1, 2}:
            raise OutputValidationError(f"{spec.name} expects 1D or 2D audio samples")
        channels = spec.config["channels"]
        if sample.ndim == 2 and sample.shape[1] != channels:
            raise OutputValidationError(
                f"{spec.name} expects {channels} audio channels, got {sample.shape[1]}"
            )
        if sample.ndim == 1 and channels != 1:
            raise OutputValidationError(f"{spec.name} expects {channels} channels, got mono audio")
        return np.ascontiguousarray(sample)

    if spec.kind == "json":
        payload = sample.model_dump(mode="json") if isinstance(sample, BaseModel) else TypeAdapter(
            JsonValue
        ).validate_python(sample)
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        if len(encoded) > spec.config["max_bytes"]:
            raise OutputValidationError(
                f"{spec.name} JSON payload exceeds {spec.config['max_bytes']} bytes"
            )
        return encoded

    if spec.kind == "bytes":
        if isinstance(sample, memoryview):
            encoded = sample.tobytes()
        elif isinstance(sample, bytearray):
            encoded = bytes(sample)
        elif isinstance(sample, bytes):
            encoded = sample
        else:
            raise OutputValidationError(f"{spec.name} expects bytes-like payloads")
        if len(encoded) > spec.config["max_bytes"]:
            raise OutputValidationError(
                f"{spec.name} bytes payload exceeds {spec.config['max_bytes']} bytes"
            )
        return encoded

    raise OutputValidationError(f"unsupported output kind: {spec.kind}")
