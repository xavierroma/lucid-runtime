from __future__ import annotations

import asyncio
import inspect
import json
import logging
import statistics
from collections.abc import Awaitable, Callable
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_type_hints

import numpy as np
from pydantic import BaseModel, Field, TypeAdapter, create_model
import yaml

from .publish import OutputSpec


class LucidError(RuntimeError):
    pass


class ManifestGenerationError(LucidError):
    pass


class ActionDispatchError(LucidError):
    pass


class OutputValidationError(LucidError):
    pass


@dataclass(frozen=True, slots=True)
class ActionMetadata:
    name: str
    description: str
    mode: str


@dataclass(frozen=True, slots=True)
class ActionDefinition:
    metadata: ActionMetadata
    arg_model: type[BaseModel]
    handler_name: str
    accepts_ctx: bool

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def mode(self) -> str:
        return self.metadata.mode

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.metadata.name,
            "description": self.metadata.description,
            "mode": self.metadata.mode,
            "args_schema": self.arg_model.model_json_schema(),
        }


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    name: str
    description: str | None
    config_path: Path
    cls: type["LucidModel"]
    actions: tuple[ActionDefinition, ...]
    outputs: tuple[OutputSpec, ...]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "model": {
                "name": self.name,
                "description": self.description,
            },
            "actions": [
                action.to_manifest() for action in self.actions
            ]
            + [runtime_action for runtime_action in runtime_actions_manifest()],
            "outputs": [output.to_manifest() for output in self.outputs],
        }


class Registry:
    def __init__(self) -> None:
        self._models: dict[str, ModelDefinition] = {}

    def register(self, definition: ModelDefinition) -> None:
        self._models[definition.name] = definition

    def get(self, name: str) -> ModelDefinition:
        if name not in self._models:
            raise LucidError(f"unknown lucid model: {name}")
        return self._models[name]

    def all(self) -> list[ModelDefinition]:
        return list(self._models.values())


registry = Registry()


def action(*, name: str, description: str, mode: str = "state") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    if mode not in {"state", "command"}:
        raise ValueError(f"unsupported action mode: {mode}")

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        setattr(
            fn,
            "_lucid_action_metadata",
            ActionMetadata(name=name, description=description, mode=mode),
        )
        return fn

    return decorator


def model(*, name: str, config: str, description: str | None = None) -> Callable[[type["LucidModel"]], type["LucidModel"]]:
    def decorator(cls: type["LucidModel"]) -> type["LucidModel"]:
        definition = build_model_definition(
            cls=cls,
            name=name,
            description=description,
            config_path=_resolve_config_path(cls, config),
        )
        setattr(cls, "_lucid_definition", definition)
        registry.register(definition)
        return cls

    return decorator


def _resolve_config_path(cls: type["LucidModel"], config: str) -> Path:
    module = inspect.getmodule(cls)
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise ManifestGenerationError(f"cannot resolve config path for {cls.__name__}")
    return Path(module_file).resolve().parent / config


def build_model_definition(
    *,
    cls: type["LucidModel"],
    name: str,
    description: str | None,
    config_path: Path,
) -> ModelDefinition:
    actions: list[ActionDefinition] = []
    outputs: list[OutputSpec] = []

    for attr_name, value in cls.__dict__.items():
        if isinstance(value, OutputSpec):
            outputs.append(value)
            continue
        metadata = getattr(value, "_lucid_action_metadata", None)
        if metadata is None:
            continue
        actions.append(_build_action_definition(attr_name, value, metadata))

    if not outputs:
        raise ManifestGenerationError(f"lucid model {name} must declare at least one output")

    return ModelDefinition(
        name=name,
        description=description,
        config_path=config_path,
        cls=cls,
        actions=tuple(sorted(actions, key=lambda item: item.name)),
        outputs=tuple(sorted(outputs, key=lambda item: item.name)),
    )


def _build_action_definition(
    handler_name: str,
    handler: Callable[..., Any],
    metadata: ActionMetadata,
) -> ActionDefinition:
    signature = inspect.signature(handler)
    type_hints = get_type_hints(handler, include_extras=True)
    fields: dict[str, tuple[Any, Any]] = {}
    accepts_ctx = False

    for parameter_name, parameter in signature.parameters.items():
        if parameter_name == "self":
            continue
        if not accepts_ctx and parameter_name == "ctx":
            accepts_ctx = True
            continue
        annotation = type_hints.get(parameter_name, parameter.annotation)
        if annotation is inspect.Signature.empty:
            annotation = Any
        default = parameter.default
        if default is inspect.Signature.empty:
            default = ...
        fields[parameter_name] = (annotation, default)

    arg_model = create_model(
        f"{metadata.name.title().replace('.', '').replace('_', '')}Args",
        __base__=BaseModel,
        **fields,
    )
    _ensure_flat_schema(metadata.name, arg_model.model_json_schema())
    return ActionDefinition(
        metadata=metadata,
        arg_model=arg_model,
        handler_name=handler_name,
        accepts_ctx=accepts_ctx,
    )


def _ensure_flat_schema(action_name: str, schema: dict[str, Any]) -> None:
    properties = schema.get("properties", {})
    for property_name, property_schema in properties.items():
        property_type = property_schema.get("type")
        items = property_schema.get("items", {})
        if property_type == "object":
            raise ManifestGenerationError(
                f"action {action_name} has unsupported nested object field {property_name}"
            )
        if property_type == "array" and isinstance(items, dict) and items.get("type") == "object":
            raise ManifestGenerationError(
                f"action {action_name} has unsupported nested array field {property_name}"
            )


class LucidModel:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.runtime_config: Any = None
        self.logger: logging.Logger | None = None

    def bind_runtime(self, runtime_config: Any, logger: logging.Logger) -> None:
        self.runtime_config = runtime_config
        self.logger = logger

    async def load(self) -> None:
        return None

    async def start_session(self, ctx: "SessionContext") -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def end_session(self, ctx: "SessionContext") -> None:
        _ = ctx

    def resolve_outputs(self, outputs: tuple[OutputSpec, ...]) -> tuple[OutputSpec, ...]:
        return outputs


class VideoModel(LucidModel):
    pass


def runtime_actions_manifest() -> list[dict[str, Any]]:
    return [
        {
            "name": "lucid.runtime.pause",
            "description": "Pause model stepping for the current session.",
            "mode": "command",
            "args_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "lucid.runtime.resume",
            "description": "Resume model stepping for the current session.",
            "mode": "command",
            "args_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "lucid.runtime.set_output_enabled",
            "description": "Enable or disable a named output for the current session.",
            "mode": "state",
            "args_schema": {
                "type": "object",
                "properties": {
                    "output": {"type": "string"},
                    "enabled": {"type": "boolean"},
                },
                "required": ["output", "enabled"],
                "additionalProperties": False,
            },
        },
        {
            "name": "lucid.runtime.set_output_rate",
            "description": "Throttle a named output to a maximum publish rate in Hertz.",
            "mode": "state",
            "args_schema": {
                "type": "object",
                "properties": {
                    "output": {"type": "string"},
                    "max_rate_hz": {"type": ["number", "null"], "minimum": 0},
                },
                "required": ["output", "max_rate_hz"],
                "additionalProperties": False,
            },
        },
    ]


class _StateView:
    def __init__(self) -> None:
        self._values: dict[str, BaseModel] = {}

    def set(self, name: str, value: BaseModel) -> None:
        self._values[name] = value

    def clear(self, name: str) -> None:
        self._values.pop(name, None)

    def get(self, name: str) -> BaseModel | None:
        return self._values.get(name)

    def reset(self) -> None:
        self._values.clear()

    def __getattr__(self, item: str) -> BaseModel:
        if item not in self._values:
            raise AttributeError(item)
        return self._values[item]


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
        self.paused = False
        self.state = _StateView()
        self._outputs = {output.name: output for output in outputs}
        self._publish_fn = publish_fn
        self._metrics_fn = metrics_fn
        self._output_enabled = {output.name: True for output in outputs}
        self._output_rate_hz = {output.name: None for output in outputs}
        self._last_publish_s = {output.name: 0.0 for output in outputs}
        self._inference_ms: deque[float] = deque(maxlen=128)

    def clear_state(self, action_name: str) -> None:
        self.state.clear(action_name)

    def reset(self) -> None:
        self.state.reset()
        self.paused = False
        self._output_enabled = {name: True for name in self._outputs}
        self._output_rate_hz = {name: None for name in self._outputs}

    def set_output_enabled(self, name: str, enabled: bool) -> None:
        self._require_output(name)
        self._output_enabled[name] = enabled

    def set_output_rate(self, name: str, max_rate_hz: float | None) -> None:
        spec = self._require_output(name)
        if spec.kind == "audio":
            raise ActionDispatchError("lucid.runtime.set_output_rate does not support audio outputs")
        self._output_rate_hz[name] = None if max_rate_hz is None else float(max_rate_hz)

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
        if not self._output_enabled[output_name]:
            return
        if self._should_throttle(output_name):
            return
        normalized = _validate_output_sample(spec, sample)
        await self._publish_fn(output_name, normalized, ts_ms)

    def _require_output(self, name: str) -> OutputSpec:
        if name not in self._outputs:
            raise OutputValidationError(f"unknown output: {name}")
        return self._outputs[name]

    def _should_throttle(self, name: str) -> bool:
        max_rate_hz = self._output_rate_hz[name]
        if not max_rate_hz:
            return False
        now_s = asyncio.get_running_loop().time()
        elapsed = now_s - self._last_publish_s[name]
        if elapsed < 1.0 / max(max_rate_hz, 1e-6):
            return True
        self._last_publish_s[name] = now_s
        return False


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
            payload = TypeAdapter(dict[str, Any] | list[Any] | str | int | float | bool | None).validate_python(sample)
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
    def __init__(self, definition: ModelDefinition, model: LucidModel, logger: logging.Logger) -> None:
        self.definition = definition
        self.model = model
        self.logger = logger
        self._actions = {action.name: action for action in definition.actions}
        self._outputs = model.resolve_outputs(definition.outputs)

    @classmethod
    def load_selected(
        cls,
        *,
        runtime_config: Any,
        logger: logging.Logger,
        model_name: str | None = None,
    ) -> "LucidRuntime":
        definitions = registry.all()
        if not definitions:
            try:
                from .discovery import ensure_model_module_loaded

                ensure_model_module_loaded()
            except Exception as exc:
                raise LucidError("no lucid models are registered") from exc
            definitions = registry.all()
        if not definitions:
            raise LucidError("no lucid models are registered")
        if model_name:
            definition = registry.get(model_name)
        elif len(definitions) == 1:
            definition = definitions[0]
        else:
            raise LucidError("multiple lucid models registered; set WM_MODEL_NAME")
        config = _load_model_config(definition.config_path)
        model = definition.cls(config)
        model.bind_runtime(runtime_config, logger)
        return cls(definition=definition, model=model, logger=logger)

    async def load(self) -> None:
        await self.model.load()

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
        return SessionContext(
            session_id=session_id,
            room_name=room_name,
            outputs=self.outputs,
            publish_fn=publish_fn,
            metrics_fn=metrics_fn,
            logger=self.logger,
        )

    async def dispatch_action(self, ctx: SessionContext, name: str, args: dict[str, Any]) -> None:
        if name.startswith("lucid.runtime."):
            await self._dispatch_runtime_action(ctx, name, args)
            return
        if name not in self._actions:
            raise ActionDispatchError(f"unknown action: {name}")
        definition = self._actions[name]
        validated = definition.arg_model.model_validate(args)
        if definition.mode == "state":
            ctx.state.set(name, validated)
        handler = getattr(self.model, definition.handler_name)
        kwargs = validated.model_dump()
        result = handler(ctx, **kwargs) if definition.accepts_ctx else handler(**kwargs)
        if inspect.isawaitable(result):
            await result

    async def _dispatch_runtime_action(
        self,
        ctx: SessionContext,
        name: str,
        args: dict[str, Any],
    ) -> None:
        if name == "lucid.runtime.pause":
            ctx.paused = True
            return
        if name == "lucid.runtime.resume":
            ctx.paused = False
            return
        if name == "lucid.runtime.set_output_enabled":
            payload = _RuntimeOutputEnabled.model_validate(args)
            ctx.set_output_enabled(payload.output, payload.enabled)
            return
        if name == "lucid.runtime.set_output_rate":
            payload = _RuntimeOutputRate.model_validate(args)
            ctx.set_output_rate(payload.output, payload.max_rate_hz)
            return
        raise ActionDispatchError(f"unknown runtime action: {name}")


class _RuntimeOutputEnabled(BaseModel):
    output: str = Field(..., min_length=1)
    enabled: bool


class _RuntimeOutputRate(BaseModel):
    output: str = Field(..., min_length=1)
    max_rate_hz: float | None = Field(default=None, ge=0)


def _load_model_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise LucidError(f"lucid model config does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise LucidError(f"lucid model config must be an object: {path}")
    return raw
