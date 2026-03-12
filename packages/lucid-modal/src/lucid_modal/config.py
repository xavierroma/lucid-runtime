from __future__ import annotations

import os

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
