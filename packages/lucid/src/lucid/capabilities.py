from __future__ import annotations

import os
from typing import Any

from .discovery import ensure_model_module_loaded
from .runtime import registry


DEFAULT_CONTROL_TOPIC = "wm.control"
DEFAULT_STATUS_TOPIC = "wm.status"


def selected_model_name() -> str:
    ensure_model_module_loaded()
    configured = os.getenv("WM_MODEL_NAME", "").strip()
    if configured:
        return configured
    models = registry.all()
    if len(models) != 1:
        raise RuntimeError("multiple lucid models registered; set WM_MODEL_NAME")
    return models[0].name


def manifest(model_name: str | None = None) -> dict[str, Any]:
    ensure_model_module_loaded()
    definition = registry.get(model_name or selected_model_name())
    return definition.to_manifest()


def output_bindings(model_name: str | None = None) -> list[dict[str, Any]]:
    ensure_model_module_loaded()
    definition = registry.get(model_name or selected_model_name())
    bindings: list[dict[str, Any]] = []
    for output in definition.outputs:
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


def capabilities(
    *,
    control_topic: str = DEFAULT_CONTROL_TOPIC,
    status_topic: str = DEFAULT_STATUS_TOPIC,
    model_name: str | None = None,
) -> dict[str, Any]:
    return {
        "control_topic": control_topic,
        "status_topic": status_topic,
        "manifest": manifest(model_name),
        "output_bindings": output_bindings(model_name),
    }
