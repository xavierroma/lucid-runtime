from __future__ import annotations

import os

from ..livekit.config import RuntimeConfig


class ConfigError(ValueError):
    pass


def _get_required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(f"{name} is required")
    return value


def load_runtime_config_from_env() -> RuntimeConfig:
    return RuntimeConfig(
        livekit_url=_get_required("LIVEKIT_URL"),
        status_topic=os.getenv("WM_STATUS_TOPIC", "wm.status"),
    )
