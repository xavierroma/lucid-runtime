from __future__ import annotations

import importlib
import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal, TypeAlias, TypedDict, get_type_hints

from pydantic import BaseModel, ConfigDict, create_model

from .model import JsonValue, LucidError, LucidModel, LucidSession, ManifestDict


ModelTarget: TypeAlias = str | type[Any]

_loaded_modules: dict[str, ModuleType] = {}
_loaded_model_classes: dict[str | type[Any], type[LucidModel[Any]]] = {}


class ManifestGenerationError(LucidError):
    pass


class _ArgsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MediaOutputBinding(TypedDict):
    name: str
    kind: Literal["video", "audio"]
    track_name: str


class DataOutputBinding(TypedDict):
    name: str
    kind: Literal["json", "bytes"]
    topic: str


OutputBinding: TypeAlias = MediaOutputBinding | DataOutputBinding
InputHandler: TypeAlias = Callable[..., Awaitable[None] | None]


@dataclass(frozen=True, slots=True)
class OutputSpec:
    name: str
    kind: str
    config: ManifestDict

    def to_manifest(self) -> ManifestDict:
        return {
            "name": self.name,
            "kind": self.kind,
            **self.config,
        }


class PublishNamespace:
    @staticmethod
    def video(
        *,
        name: str,
        width: int,
        height: int,
        fps: int,
        pixel_format: str = "rgb24",
    ) -> OutputSpec:
        return OutputSpec(
            name=name,
            kind="video",
            config={
                "width": int(width),
                "height": int(height),
                "fps": int(fps),
                "pixel_format": pixel_format,
            },
        )

    @staticmethod
    def audio(
        *,
        name: str,
        sample_rate_hz: int,
        channels: int,
        sample_format: str = "float32",
    ) -> OutputSpec:
        return OutputSpec(
            name=name,
            kind="audio",
            config={
                "sample_rate_hz": int(sample_rate_hz),
                "channels": int(channels),
                "sample_format": sample_format,
            },
        )

    @staticmethod
    def json(
        *,
        name: str,
        schema: ManifestDict | None = None,
        max_bytes: int = 16 * 1024,
    ) -> OutputSpec:
        return OutputSpec(
            name=name,
            kind="json",
            config={
                "schema": schema or {"type": "object"},
                "max_bytes": int(max_bytes),
            },
        )

    @staticmethod
    def bytes(
        *,
        name: str,
        content_type: str = "application/octet-stream",
        max_bytes: int = 16 * 1024,
    ) -> OutputSpec:
        return OutputSpec(
            name=name,
            kind="bytes",
            config={
                "content_type": content_type,
                "max_bytes": int(max_bytes),
            },
        )


publish = PublishNamespace()


@dataclass(frozen=True, slots=True)
class HoldBinding:
    keys: tuple[str, ...] = ()
    mouse_buttons: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.keys and not self.mouse_buttons:
            raise ValueError("hold() requires at least one key or mouse button")

    def to_manifest(self) -> ManifestDict:
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

    def to_manifest(self) -> ManifestDict:
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

    def to_manifest(self) -> ManifestDict:
        return {
            "kind": "axis",
            "positive_keys": list(self.positive_keys),
            "negative_keys": list(self.negative_keys),
        }


@dataclass(frozen=True, slots=True)
class PointerBinding:
    pointer_lock: bool = True

    def to_manifest(self) -> ManifestDict:
        return {
            "kind": "pointer",
            "pointer_lock": self.pointer_lock,
        }


@dataclass(frozen=True, slots=True)
class WheelBinding:
    step: int = 120

    def to_manifest(self) -> ManifestDict:
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
    paused: bool | None


@dataclass(frozen=True, slots=True)
class InputDefinition:
    metadata: InputMetadata
    arg_model: type[BaseModel]
    handler_name: str
    paused: bool

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def binding(self) -> InputBinding | None:
        return self.metadata.binding

    def to_manifest(self) -> ManifestDict:
        payload = {
            "name": self.name,
            "description": self.metadata.description,
            "args_schema": self.arg_model.model_json_schema(),
        }
        if self.binding is not None:
            payload["binding"] = self.binding.to_manifest()
        return payload


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    name: str
    description: str | None
    cls: type[LucidModel[Any]]
    session_cls: type[LucidSession[Any]]
    config_cls: type[BaseModel]
    inputs: tuple[InputDefinition, ...]
    outputs: tuple[OutputSpec, ...]

    def to_manifest(self) -> ManifestDict:
        return {
            "model": {
                "name": self.name,
                "description": self.description,
            },
            "inputs": [item.to_manifest() for item in self.inputs],
            "outputs": [output.to_manifest() for output in self.outputs],
        }

    def output_bindings(self) -> list[OutputBinding]:
        bindings: list[OutputBinding] = []
        for output in self.outputs:
            if output.kind in {"video", "audio"}:
                bindings.append(
                    {
                        "name": output.name,
                        "kind": output.kind,
                        "track_name": output.name,
                    }
                )
                continue
            bindings.append(
                {
                    "name": output.name,
                    "kind": output.kind,
                    "topic": f"wm.output.{output.name}",
                }
            )
        return bindings


def input(
    *,
    name: str | None = None,
    description: str | None = None,
    binding: InputBinding | None = None,
    paused: bool | None = None,
) -> Callable[[InputHandler], InputHandler]:
    def decorator(fn: InputHandler) -> InputHandler:
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
                paused=paused,
            ),
        )
        return fn

    return decorator


def load_model_module(model: ModelTarget) -> ModuleType:
    if inspect.isclass(model):
        module_name = model.__module__
    elif isinstance(model, str):
        module_name, _ = _split_model_spec(model)
    else:
        raise RuntimeError(f"invalid lucid model target: {model!r}")

    loaded = _loaded_modules.get(module_name)
    if loaded is not None:
        return loaded

    module = importlib.import_module(module_name)
    _loaded_modules[module_name] = module
    return module


def resolve_model_class(model: ModelTarget) -> type[LucidModel[Any]]:
    loaded = _loaded_model_classes.get(model)
    if loaded is not None:
        return loaded

    if inspect.isclass(model):
        candidate = model
    elif isinstance(model, str):
        module_name, class_name = _split_model_spec(model)
        candidate = getattr(load_model_module(model), class_name, None)
        if candidate is None:
            raise RuntimeError(f"lucid model class not found: {module_name}:{class_name}")
    else:
        raise RuntimeError(f"invalid lucid model target: {model!r}")

    if not inspect.isclass(candidate) or not issubclass(candidate, LucidModel) or candidate is LucidModel:
        raise RuntimeError(f"lucid model target must be a LucidModel subclass: {model!r}")

    _loaded_model_classes[model] = candidate
    return candidate


def resolve_model_definition(model: ModelTarget) -> ModelDefinition:
    return build_model_definition(resolve_model_class(model))


def build_model_definition(model_cls: type[LucidModel[Any]]) -> ModelDefinition:
    cached = getattr(model_cls, "__lucid_definition__", None)
    if isinstance(cached, ModelDefinition):
        return cached

    outputs = tuple(getattr(model_cls, "outputs", ()))
    if not outputs:
        raise ManifestGenerationError(
            f"lucid model {model_cls.__name__} must declare explicit outputs"
        )
    if any(not isinstance(output, OutputSpec) for output in outputs):
        raise ManifestGenerationError(
            f"{model_cls.__name__}.outputs must contain only OutputSpec values"
        )

    config_cls = getattr(model_cls, "config_cls", LucidModel.config_cls)
    if not inspect.isclass(config_cls) or not issubclass(config_cls, BaseModel):
        raise ManifestGenerationError(f"{model_cls.__name__}.config_cls must be a BaseModel subclass")

    session_cls = getattr(model_cls, "session_cls", None)
    if (
        not inspect.isclass(session_cls)
        or not issubclass(session_cls, LucidSession)
        or session_cls is LucidSession
    ):
        raise ManifestGenerationError(
            f"{model_cls.__name__} must declare an explicit session_cls"
        )

    definition = ModelDefinition(
        name=_resolve_name(model_cls),
        description=_resolve_description(model_cls),
        cls=model_cls,
        session_cls=session_cls,
        config_cls=config_cls,
        inputs=tuple(sorted(_collect_inputs(session_cls), key=lambda item: item.name)),
        outputs=tuple(sorted(outputs, key=lambda output: output.name)),
    )
    setattr(model_cls, "__lucid_definition__", definition)
    return definition


def manifest(model_cls: type[LucidModel[Any]]) -> ManifestDict:
    return build_model_definition(model_cls).to_manifest()


def _split_model_spec(model_spec: str) -> tuple[str, str]:
    module_name, _, class_name = model_spec.partition(":")
    resolved_module_name = module_name.strip()
    resolved_class_name = class_name.strip()
    if not resolved_module_name or not resolved_class_name:
        raise RuntimeError(
            f"invalid lucid model spec {model_spec!r}; expected 'pkg.module:ClassName'"
        )
    return resolved_module_name, resolved_class_name


def _resolve_name(model_cls: type[LucidModel[Any]]) -> str:
    name = str(getattr(model_cls, "name", "") or model_cls.__name__).strip()
    if not name:
        raise ManifestGenerationError(f"{model_cls.__name__}.name cannot be empty")
    return name


def _resolve_description(model_cls: type[LucidModel[Any]]) -> str | None:
    description = getattr(model_cls, "description", None)
    if description is None:
        return None
    return str(description).strip() or None


def _collect_inputs(session_cls: type[LucidSession[Any]]) -> list[InputDefinition]:
    inputs: list[InputDefinition] = []
    for cls in reversed(session_cls.__mro__):
        if cls in {LucidSession, object}:
            continue
        for handler_name, handler in cls.__dict__.items():
            metadata = getattr(handler, "_lucid_input_metadata", None)
            if metadata is None:
                continue
            inputs.append(_build_input_definition(handler_name, handler, metadata))
    return inputs


def _build_input_definition(
    handler_name: str,
    handler: InputHandler,
    metadata: InputMetadata,
) -> InputDefinition:
    signature = inspect.signature(handler)
    type_hints = get_type_hints(handler, include_extras=True)
    fields: dict[str, tuple[object, object]] = {}
    param_names: list[str] = []

    for parameter_name, parameter in signature.parameters.items():
        if parameter_name == "self":
            continue
        annotation = type_hints.get(parameter_name, parameter.annotation)
        if annotation is inspect.Signature.empty:
            raise ManifestGenerationError(
                f"input {metadata.name} parameter {parameter_name} must have a type annotation"
            )
        fields[parameter_name] = (
            annotation,
            ... if parameter.default is inspect.Signature.empty else parameter.default,
        )
        param_names.append(parameter_name)

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
        paused=_resolve_paused(metadata),
    )


def _resolve_paused(metadata: InputMetadata) -> bool:
    if metadata.paused is not None:
        return metadata.paused
    return isinstance(metadata.binding, (HoldBinding, AxisBinding)) or metadata.name == "set_prompt"


def _ensure_flat_schema(input_name: str, schema: ManifestDict) -> None:
    for property_name, property_schema in schema.get("properties", {}).items():
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
    schema: ManifestDict,
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
