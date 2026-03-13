from __future__ import annotations

import asyncio
import inspect
import json
import logging
import statistics
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Generic, TypeVar, get_type_hints

import numpy as np
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, create_model

from .discovery import ModelTarget, resolve_model_class
from .publish import OutputSpec


class LucidError(RuntimeError):
    pass


class ManifestGenerationError(LucidError):
    pass


class ActionDispatchError(LucidError):
    pass


class OutputValidationError(LucidError):
    pass


class _ArgsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _EmptyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


TConfig = TypeVar("TConfig", bound=BaseModel)
TModel = TypeVar("TModel", bound="LucidModel[Any]")


class LoadContext:
    def __init__(
        self,
        *,
        config: BaseModel,
        logger: logging.Logger,
    ) -> None:
        self.config = config
        self.logger = logger


@dataclass(frozen=True, slots=True)
class HoldBinding:
    keys: tuple[str, ...] = ()
    mouse_buttons: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.keys and not self.mouse_buttons:
            raise ValueError("hold() requires at least one key or mouse button")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "kind": "hold",
            "keys": list(self.keys),
            "mouse_buttons": list(self.mouse_buttons),
        }


@dataclass(frozen=True, slots=True)
class PressBinding:
    keys: tuple[str, ...] = ()
    mouse_buttons: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.keys and not self.mouse_buttons:
            raise ValueError("press() requires at least one key or mouse button")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "kind": "press",
            "keys": list(self.keys),
            "mouse_buttons": list(self.mouse_buttons),
        }


@dataclass(frozen=True, slots=True)
class AxisBinding:
    positive_keys: tuple[str, ...]
    negative_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.positive_keys:
            raise ValueError("axis() requires at least one positive key")
        if not self.negative_keys:
            raise ValueError("axis() requires at least one negative key")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "kind": "axis",
            "positive_keys": list(self.positive_keys),
            "negative_keys": list(self.negative_keys),
        }


@dataclass(frozen=True, slots=True)
class PointerBinding:
    pointer_lock: bool = True

    def to_manifest(self) -> dict[str, Any]:
        return {
            "kind": "pointer",
            "pointer_lock": self.pointer_lock,
        }


@dataclass(frozen=True, slots=True)
class WheelBinding:
    step: int = 120

    def to_manifest(self) -> dict[str, Any]:
        return {
            "kind": "wheel",
            "step": self.step,
        }


InputBinding = HoldBinding | PressBinding | AxisBinding | PointerBinding | WheelBinding


def hold(
    *,
    keys: Sequence[str] = (),
    mouse_buttons: Sequence[int] = (),
) -> HoldBinding:
    return HoldBinding(tuple(keys), tuple(int(button) for button in mouse_buttons))


def press(
    *,
    keys: Sequence[str] = (),
    mouse_buttons: Sequence[int] = (),
) -> PressBinding:
    return PressBinding(tuple(keys), tuple(int(button) for button in mouse_buttons))


def axis(
    *,
    positive_keys: Sequence[str],
    negative_keys: Sequence[str],
) -> AxisBinding:
    return AxisBinding(tuple(positive_keys), tuple(negative_keys))


def pointer(*, pointer_lock: bool = True) -> PointerBinding:
    return PointerBinding(pointer_lock=pointer_lock)


def wheel(*, step: int = 120) -> WheelBinding:
    return WheelBinding(step=int(step))


@dataclass(frozen=True, slots=True)
class InputMetadata:
    name: str
    description: str | None
    binding: InputBinding | None


@dataclass(frozen=True, slots=True)
class InputDefinition:
    metadata: InputMetadata
    arg_model: type[BaseModel]
    handler_name: str

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def binding(self) -> InputBinding | None:
        return self.metadata.binding

    def to_manifest(self) -> dict[str, Any]:
        payload = {
            "name": self.metadata.name,
            "description": self.metadata.description,
            "args_schema": self.arg_model.model_json_schema(),
        }
        if self.metadata.binding is not None:
            payload["binding"] = self.metadata.binding.to_manifest()
        return payload


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    name: str
    description: str | None
    cls: type["LucidModel[Any]"]
    session_cls: type["LucidSession[Any]"]
    config_cls: type[BaseModel]
    inputs: tuple[InputDefinition, ...]
    outputs: tuple[OutputSpec, ...]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "model": {
                "name": self.name,
                "description": self.description,
            },
            "inputs": [item.to_manifest() for item in self.inputs],
            "outputs": [output.to_manifest() for output in self.outputs],
        }


def input(
    *,
    name: str | None = None,
    description: str | None = None,
    binding: InputBinding | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        resolved_name = (name or fn.__name__).strip()
        if not resolved_name:
            raise ValueError("input name cannot be empty")
        setattr(
            fn,
            "_lucid_input_metadata",
            InputMetadata(
                name=resolved_name,
                description=(description or "").strip() or None,
                binding=binding,
            ),
        )
        return fn

    return decorator


def _collect_outputs(cls: type["LucidModel[Any]"]) -> tuple[OutputSpec, ...]:
    declared = getattr(cls, "outputs", None)
    if declared is not None:
        outputs = tuple(declared)
        if any(not isinstance(output, OutputSpec) for output in outputs):
            raise ManifestGenerationError(f"{cls.__name__}.outputs must contain only OutputSpec values")
        return outputs

    outputs: list[OutputSpec] = []
    for value in cls.__dict__.values():
        if isinstance(value, OutputSpec):
            outputs.append(value)
    return tuple(outputs)


def _resolve_session_cls(cls: type["LucidModel[Any]"]) -> type["LucidSession[Any]"]:
    declared = getattr(cls, "session_cls", None)
    if inspect.isclass(declared) and issubclass(declared, LucidSession):
        return declared

    try:
        hints = get_type_hints(cls.create_session, include_extras=True)
    except Exception as exc:  # pragma: no cover - defensive type-hint resolution
        raise ManifestGenerationError(
            f"failed resolving create_session() return annotation for {cls.__name__}"
        ) from exc

    session_cls = hints.get("return")
    if not inspect.isclass(session_cls) or not issubclass(session_cls, LucidSession):
        raise ManifestGenerationError(
            f"{cls.__name__}.create_session() must return a LucidSession subclass"
        )
    return session_cls


def build_model_definition(
    cls: type["LucidModel[Any]"],
) -> ModelDefinition:
    outputs = tuple(sorted(_collect_outputs(cls), key=lambda output: output.name))
    if not outputs:
        raise ManifestGenerationError(f"lucid model {cls.__name__} must declare at least one output")

    config_cls = getattr(cls, "config_cls", _EmptyConfig)
    if not inspect.isclass(config_cls) or not issubclass(config_cls, BaseModel):
        raise ManifestGenerationError(f"{cls.__name__}.config_cls must be a BaseModel subclass")

    session_cls = _resolve_session_cls(cls)
    inputs = tuple(
        sorted(_collect_input_definitions(session_cls), key=lambda item: item.name)
    )

    name = str(getattr(cls, "name", "") or cls.__name__).strip()
    if not name:
        raise ManifestGenerationError(f"{cls.__name__}.name cannot be empty")

    description = getattr(cls, "description", None)
    if description is not None:
        description = str(description).strip() or None

    return ModelDefinition(
        name=name,
        description=description,
        cls=cls,
        session_cls=session_cls,
        config_cls=config_cls,
        inputs=inputs,
        outputs=outputs,
    )


def _collect_input_definitions(
    session_cls: type["LucidSession[Any]"],
) -> list[InputDefinition]:
    inputs: list[InputDefinition] = []
    for cls in reversed(session_cls.__mro__):
        if cls in {LucidSession, object}:
            continue
        for attr_name, value in cls.__dict__.items():
            metadata = getattr(value, "_lucid_input_metadata", None)
            if metadata is None:
                continue
            inputs.append(_build_input_definition(attr_name, value, metadata))
    return inputs


def _build_input_definition(
    handler_name: str,
    handler: Callable[..., Any],
    metadata: InputMetadata,
) -> InputDefinition:
    signature = inspect.signature(handler)
    type_hints = get_type_hints(handler, include_extras=True)
    fields: dict[str, tuple[Any, Any]] = {}
    param_names: list[str] = []

    for parameter_name, parameter in signature.parameters.items():
        if parameter_name == "self":
            continue
        annotation = type_hints.get(parameter_name, parameter.annotation)
        if annotation is inspect.Signature.empty:
            annotation = Any
        default = parameter.default
        if default is inspect.Signature.empty:
            default = ...
        param_names.append(parameter_name)
        fields[parameter_name] = (annotation, default)

    arg_model = create_model(
        f"{metadata.name.title().replace('.', '').replace('_', '')}InputArgs",
        __base__=_ArgsModel,
        **fields,
    )
    schema = arg_model.model_json_schema()
    _ensure_flat_schema(metadata.name, schema)
    _validate_binding_signature(metadata.name, metadata.binding, param_names, schema)
    return InputDefinition(
        metadata=metadata,
        arg_model=arg_model,
        handler_name=handler_name,
    )


def _ensure_flat_schema(input_name: str, schema: dict[str, Any]) -> None:
    properties = schema.get("properties", {})
    for property_name, property_schema in properties.items():
        property_type = property_schema.get("type")
        items = property_schema.get("items", {})
        if property_type == "object":
            raise ManifestGenerationError(
                f"input {input_name} has unsupported nested object field {property_name}"
            )
        if property_type == "array" and isinstance(items, dict) and items.get("type") == "object":
            raise ManifestGenerationError(
                f"input {input_name} has unsupported nested array field {property_name}"
            )


def _validate_binding_signature(
    input_name: str,
    binding: InputBinding | None,
    param_names: list[str],
    schema: dict[str, Any],
) -> None:
    if binding is None:
        return

    properties = schema.get("properties", {})

    if isinstance(binding, HoldBinding):
        if param_names != ["pressed"] or properties.get("pressed", {}).get("type") != "boolean":
            raise ManifestGenerationError(
                f"input {input_name} bound with hold() must have signature (pressed: bool)"
            )
        return

    if isinstance(binding, PressBinding):
        if param_names:
            raise ManifestGenerationError(
                f"input {input_name} bound with press() must have signature ()"
            )
        return

    if isinstance(binding, AxisBinding):
        if param_names != ["value"] or properties.get("value", {}).get("type") != "number":
            raise ManifestGenerationError(
                f"input {input_name} bound with axis() must have signature (value: float)"
            )
        return

    if isinstance(binding, PointerBinding):
        if (
            param_names != ["dx", "dy"]
            or properties.get("dx", {}).get("type") != "number"
            or properties.get("dy", {}).get("type") != "number"
        ):
            raise ManifestGenerationError(
                f"input {input_name} bound with pointer() must have signature (dx: float, dy: float)"
            )
        return

    if isinstance(binding, WheelBinding):
        if param_names != ["delta"] or properties.get("delta", {}).get("type") != "number":
            raise ManifestGenerationError(
                f"input {input_name} bound with wheel() must have signature (delta: float)"
            )
        return

    raise ManifestGenerationError(f"unsupported binding for input {input_name}: {binding!r}")


class SessionContext:
    def __init__(
        self,
        *,
        session_id: str,
        room_name: str,
        outputs: tuple[OutputSpec, ...],
        publish_fn: Callable[[str, Any, int | None], Awaitable[None]],
        logger: logging.Logger,
        metrics_fn: Callable[[], dict[str, float | int]] | None = None,
    ) -> None:
        self.session_id = session_id
        self.room_name = room_name
        self.logger = logger
        self.running = True
        self._outputs = {output.name: output for output in outputs}
        self._publish_fn = publish_fn
        self._metrics_fn = metrics_fn
        self._inference_ms: deque[float] = deque(maxlen=128)
        self._initial_input_event = asyncio.Event()
        self._pause_cleared_event = asyncio.Event()
        self._pause_cleared_event.set()
        self._paused = False

    def record_inference_ms(self, value: float) -> None:
        self._inference_ms.append(float(value))

    def inference_ms_p50(self) -> float:
        if not self._inference_ms:
            return 0.0
        return float(statistics.median(self._inference_ms))

    def output_metrics(self) -> dict[str, float | int]:
        if self._metrics_fn is None:
            return {
                "effective_fps": 0.0,
                "queue_depth": 0,
                "dropped_frames": 0,
            }
        snapshot = dict(self._metrics_fn())
        snapshot.setdefault("effective_fps", 0.0)
        snapshot.setdefault("queue_depth", 0)
        snapshot.setdefault("dropped_frames", 0)
        return snapshot

    async def publish(self, output_name: str, sample: Any, ts_ms: int | None = None) -> None:
        spec = self._require_output(output_name)
        normalized = _validate_output_sample(spec, sample)
        await self._publish_fn(output_name, normalized, ts_ms)

    def mark_input_received(self) -> None:
        self._initial_input_event.set()

    def pause(self) -> bool:
        if self._paused:
            return False
        self._paused = True
        self._pause_cleared_event.clear()
        return True

    def resume(self) -> bool:
        if not self._paused:
            return False
        self._paused = False
        self._pause_cleared_event.set()
        return True

    def is_paused(self) -> bool:
        return self._paused

    async def wait_if_paused(self) -> None:
        while self.running and self._paused:
            await self._pause_cleared_event.wait()

    async def wait_for_initial_input(self, timeout_s: float) -> bool:
        if self._initial_input_event.is_set():
            return True
        try:
            await asyncio.wait_for(self._initial_input_event.wait(), timeout=timeout_s)
        except TimeoutError:
            return False
        return True

    def _require_output(self, name: str) -> OutputSpec:
        if name not in self._outputs:
            raise OutputValidationError(f"unknown output: {name}")
        return self._outputs[name]


class LucidModel(Generic[TConfig]):
    name = "lucid-model"
    description: str | None = None
    config_cls: type[BaseModel] = _EmptyConfig
    outputs: tuple[OutputSpec, ...] = ()

    def __init__(self, config: TConfig) -> None:
        self.config = config
        self.runtime_config: Any = None
        self.logger: logging.Logger | None = None

    def bind_runtime(self, runtime_config: Any, logger: logging.Logger) -> None:
        self.runtime_config = runtime_config
        self.logger = logger

    async def load(self, ctx: LoadContext) -> None:
        _ = ctx

    async def unload(self) -> None:
        return None

    def create_session(self, ctx: SessionContext) -> "LucidSession[LucidModel[TConfig]]":  # pragma: no cover - interface
        raise NotImplementedError


class LucidSession(Generic[TModel]):
    def __init__(self, model: TModel, ctx: SessionContext) -> None:
        self.model = model
        self.ctx = ctx

    async def run(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def close(self) -> None:
        return None


def _validate_output_sample(spec: OutputSpec, sample: Any) -> Any:
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
        if spec.config.get("pixel_format") != "rgb24":
            raise OutputValidationError("only rgb24 video outputs are supported")
        return np.ascontiguousarray(sample)

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
        if isinstance(sample, BaseModel):
            payload = sample.model_dump(mode="json")
        else:
            payload = TypeAdapter(
                dict[str, Any] | list[Any] | str | int | float | bool | None
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


class LucidRuntime:
    def __init__(
        self,
        *,
        definition: ModelDefinition,
        model: LucidModel[Any],
        logger: logging.Logger,
        model_target: ModelTarget,
    ) -> None:
        self.definition = definition
        self.model = model
        self.logger = logger
        self.model_target = model_target
        self._inputs = {item.name: item for item in definition.inputs}
        self._outputs = tuple(definition.outputs)
        self._sessions: dict[str, LucidSession[Any]] = {}
        self._loaded = False
        self._unloaded = False

    @classmethod
    def load_model(
        cls,
        *,
        runtime_config: Any,
        logger: logging.Logger,
        model: ModelTarget,
        config: BaseModel | dict[str, Any] | None = None,
    ) -> "LucidRuntime":
        model_cls = resolve_model_class(model)
        definition = build_model_definition(model_cls)
        model_config = _coerce_model_config(definition.config_cls, config)
        instance = definition.cls(model_config)
        instance.bind_runtime(runtime_config, logger)
        return cls(
            definition=definition,
            model=instance,
            logger=logger,
            model_target=model,
        )

    async def load(self) -> None:
        if self._loaded:
            return
        start = perf_counter()
        load_ctx = LoadContext(
            config=self.model.config,
            logger=self.logger,
        )
        try:
            result = self.model.load(load_ctx)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            self.logger.error(
                "lucid.runtime.load failed duration_ms=%.1f model=%s error_type=%s",
                (perf_counter() - start) * 1000.0,
                self.definition.name,
                exc.__class__.__name__,
            )
            raise
        self._loaded = True
        self.logger.info(
            "lucid.runtime.load complete duration_ms=%.1f model=%s",
            (perf_counter() - start) * 1000.0,
            self.definition.name,
        )

    async def unload(self) -> None:
        if not self._loaded or self._unloaded:
            return
        result = self.model.unload()
        if inspect.isawaitable(result):
            await result
        self._unloaded = True

    def manifest(self) -> dict[str, Any]:
        return self.definition.to_manifest()

    @property
    def outputs(self) -> tuple[OutputSpec, ...]:
        return self._outputs

    def output_bindings(self) -> list[dict[str, Any]]:
        bindings: list[dict[str, Any]] = []
        for output in self.outputs:
            if output.kind in {"video", "audio"}:
                bindings.append(
                    {
                        "name": output.name,
                        "kind": output.kind,
                        "track_name": output.name,
                    }
                )
            else:
                bindings.append(
                    {
                        "name": output.name,
                        "kind": output.kind,
                        "topic": f"wm.output.{output.name}",
                    }
                )
        return bindings

    def create_session_context(
        self,
        *,
        session_id: str,
        room_name: str,
        publish_fn: Callable[[str, Any, int | None], Awaitable[None]],
        metrics_fn: Callable[[], dict[str, float | int]] | None = None,
    ) -> SessionContext:
        ctx = SessionContext(
            session_id=session_id,
            room_name=room_name,
            outputs=self.outputs,
            publish_fn=publish_fn,
            metrics_fn=metrics_fn,
            logger=self.logger,
        )
        session = self.model.create_session(ctx)
        if inspect.isawaitable(session):
            raise LucidError("create_session() must be synchronous")
        if not isinstance(session, self.definition.session_cls):
            raise LucidError(
                f"create_session() returned {session.__class__.__name__}, expected {self.definition.session_cls.__name__}"
            )
        self._sessions[session_id] = session
        return ctx

    async def run_session(self, ctx: SessionContext) -> None:
        session = self._require_session(ctx.session_id)
        result = session.run()
        if inspect.isawaitable(result):
            await result

    async def close_session(self, ctx: SessionContext) -> None:
        session = self._sessions.pop(ctx.session_id, None)
        if session is None:
            return
        result = session.close()
        if inspect.isawaitable(result):
            await result

    async def dispatch_input(self, ctx: SessionContext, name: str, args: dict[str, Any]) -> None:
        if name not in self._inputs:
            raise ActionDispatchError(f"unknown input: {name}")
        definition = self._inputs[name]
        try:
            validated = definition.arg_model.model_validate(args)
        except ValidationError as exc:
            raise ActionDispatchError(f"invalid input args for {name}: {exc}") from exc
        session = self._require_session(ctx.session_id)
        handler = getattr(session, definition.handler_name)
        result = handler(**validated.model_dump())
        if inspect.isawaitable(result):
            await result
        ctx.mark_input_received()

    async def dispatch_action(self, ctx: SessionContext, name: str, args: dict[str, Any]) -> None:
        await self.dispatch_input(ctx, name, args)

    def allows_input_while_paused(self, name: str) -> bool:
        definition = self._inputs.get(name)
        if definition is None:
            return False
        binding = definition.binding
        if binding is None:
            return name == "set_prompt"
        return isinstance(binding, HoldBinding | AxisBinding)

    def _require_session(self, session_id: str) -> LucidSession[Any]:
        session = self._sessions.get(session_id)
        if session is None:
            raise LucidError(f"unknown lucid session: {session_id}")
        return session


def _coerce_model_config(
    config_cls: type[BaseModel],
    config: BaseModel | dict[str, Any] | None,
) -> BaseModel:
    if config is None:
        return config_cls()
    if isinstance(config, config_cls):
        return config
    if isinstance(config, BaseModel):
        raw: dict[str, Any] = config.model_dump(mode="python")
    elif isinstance(config, dict):
        raw = config
    else:
        raise LucidError(
            f"lucid model config must be a BaseModel or dict, got {type(config).__name__}"
        )
    try:
        return config_cls.model_validate(raw)
    except ValidationError as exc:
        raise LucidError(f"invalid lucid model config: {exc}") from exc
