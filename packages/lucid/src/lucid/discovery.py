from __future__ import annotations

import importlib
import inspect
from types import ModuleType
from typing import Any, TypeAlias


ModelTarget: TypeAlias = str | type[Any]

_loaded_modules: dict[str, ModuleType] = {}
_loaded_model_classes: dict[str | type[Any], type[Any]] = {}


def _split_model_spec(model_spec: str) -> tuple[str, str]:
    module_name, _, class_name = model_spec.partition(":")
    resolved_module_name = module_name.strip()
    resolved_class_name = class_name.strip()
    if not resolved_module_name or not resolved_class_name:
        raise RuntimeError(
            f"invalid lucid model spec {model_spec!r}; expected 'pkg.module:ClassName'"
        )
    return resolved_module_name, resolved_class_name


def load_model_module(model: ModelTarget) -> ModuleType:
    if inspect.isclass(model):
        module_name = model.__module__
    elif isinstance(model, str):
        module_name, _class_name = _split_model_spec(model)
    else:
        raise RuntimeError(f"invalid lucid model target: {model!r}")

    loaded = _loaded_modules.get(module_name)
    if loaded is not None:
        return loaded

    module = importlib.import_module(module_name)
    _loaded_modules[module_name] = module
    return module


def resolve_model_class(model: ModelTarget) -> type[Any]:
    loaded = _loaded_model_classes.get(model)
    if loaded is not None:
        return loaded

    if inspect.isclass(model):
        candidate = model
    elif isinstance(model, str):
        module_name, class_name = _split_model_spec(model)
        module = load_model_module(model)
        candidate = getattr(module, class_name, None)
        if candidate is None:
            raise RuntimeError(f"lucid model class not found: {module_name}:{class_name}")
    else:
        raise RuntimeError(f"invalid lucid model target: {model!r}")

    from .runtime import LucidModel

    if not inspect.isclass(candidate) or not issubclass(candidate, LucidModel) or candidate is LucidModel:
        raise RuntimeError(f"lucid model target must be a LucidModel subclass: {model!r}")

    _loaded_model_classes[model] = candidate
    return candidate
