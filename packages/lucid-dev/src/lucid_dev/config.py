from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError
import yaml

from lucid import RuntimeConfig


class ConfigError(ValueError):
    pass


def _get_required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(f"{name} is required")
    return value


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got: {raw}") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be > 0, got: {value}")
    return value


def load_runtime_config_from_env() -> RuntimeConfig:
    return RuntimeConfig(
        livekit_url=_get_required("LIVEKIT_URL"),
        frame_width=_get_int("WM_FRAME_WIDTH", 1280),
        frame_height=_get_int("WM_FRAME_HEIGHT", 720),
        target_fps=_get_int("WM_TARGET_FPS", 16),
        status_topic=os.getenv("WM_STATUS_TOPIC", "wm.status"),
        max_queue_frames=_get_int("WM_MAX_QUEUE_FRAMES", 32),
        livekit_mode=os.getenv("WM_LIVEKIT_MODE", "fake").strip().lower(),
    )


def load_livekit_api_credentials_from_env() -> tuple[str, str]:
    return (_get_required("LIVEKIT_API_KEY"), _get_required("LIVEKIT_API_SECRET"))


def load_model_config_from_path(
    config_cls: type[BaseModel],
    path: str | Path | None,
) -> BaseModel:
    if path is None:
        return config_cls()
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise ConfigError(f"lucid model config does not exist: {resolved}")
    raw_text = resolved.read_text(encoding="utf-8")
    if resolved.suffix.lower() == ".json":
        raw = json.loads(raw_text) if raw_text.strip() else {}
    else:
        raw = yaml.safe_load(raw_text) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"lucid model config must be an object: {resolved}")
    try:
        return config_cls.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid lucid model config {resolved}: {exc}") from exc


def load_model_config_from_values(
    config_cls: type[BaseModel],
    raw: BaseModel | dict[str, Any] | None,
) -> BaseModel:
    if raw is None:
        return config_cls()
    if isinstance(raw, config_cls):
        return raw
    if isinstance(raw, BaseModel):
        data = raw.model_dump(mode="python")
    elif isinstance(raw, dict):
        data = raw
    else:
        raise ConfigError(f"lucid model config must be a BaseModel or dict, got {type(raw).__name__}")
    try:
        return config_cls.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid lucid model config: {exc}") from exc
