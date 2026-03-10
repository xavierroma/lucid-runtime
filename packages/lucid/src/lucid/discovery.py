from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Any


_loaded_modules: dict[str, ModuleType] = {}


def configured_model_module() -> str:
    configured = os.getenv("WM_MODEL_MODULE", "").strip()
    if not configured:
        raise RuntimeError("WM_MODEL_MODULE is required")
    return configured


def load_model_module(module_name: str | None = None) -> ModuleType:
    resolved_name = module_name or configured_model_module()
    loaded = _loaded_modules.get(resolved_name)
    if loaded is not None:
        return loaded
    module = importlib.import_module(resolved_name)
    _loaded_modules[resolved_name] = module
    return module


def ensure_model_module_loaded(module_name: str | None = None) -> str:
    return load_model_module(module_name).__name__


def configured_model_packages(module_name: str | None = None) -> tuple[str, ...]:
    resolved_name = module_name or configured_model_module()
    package_name = resolved_name.split(".", 1)[0].strip()
    return (package_name,) if package_name else ()


def build_model_runtime_config(host_config: Any, module_name: str | None = None) -> Any:
    module = load_model_module(module_name)
    builder = getattr(module, "build_runtime_config", None)
    if builder is None:
        return host_config
    return builder(host_config)
