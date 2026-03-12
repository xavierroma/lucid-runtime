from __future__ import annotations

from typing import Any

from .discovery import ModelTarget, resolve_model_class
from .runtime import build_model_definition


DEFAULT_CONTROL_TOPIC = "wm.control"
DEFAULT_STATUS_TOPIC = "wm.status"


def selected_model_name(model: ModelTarget) -> str:
    definition = build_model_definition(resolve_model_class(model))
    return definition.name


def manifest(model: ModelTarget) -> dict[str, Any]:
    definition = build_model_definition(resolve_model_class(model))
    return definition.to_manifest()


def output_bindings(model: ModelTarget) -> list[dict[str, Any]]:
    definition = build_model_definition(resolve_model_class(model))
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
    model: ModelTarget,
) -> dict[str, Any]:
    return {
        "control_topic": control_topic,
        "status_topic": status_topic,
        "manifest": manifest(model),
        "output_bindings": output_bindings(model),
    }
