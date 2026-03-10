from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from lucid.config import ConfigError, RuntimeConfig as HostRuntimeConfig


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


@dataclass(frozen=True, slots=True)
class YumeRuntimeConfig:
    livekit_url: str
    frame_width: int
    frame_height: int
    target_fps: int
    status_topic: str
    max_queue_frames: int
    livekit_mode: str
    wm_engine: str
    yume_model_dir: Path
    yume_chunk_frames: int
    yume_base_prompt: str


def build_runtime_config(host_config: HostRuntimeConfig) -> YumeRuntimeConfig:
    advertised_fps = min(
        host_config.target_fps,
        _get_int("YUME_ADVERTISED_FPS", 2),
    )
    return YumeRuntimeConfig(
        livekit_url=host_config.livekit_url,
        frame_width=host_config.frame_width,
        frame_height=host_config.frame_height,
        target_fps=advertised_fps,
        status_topic=host_config.status_topic,
        max_queue_frames=host_config.max_queue_frames,
        livekit_mode=host_config.livekit_mode,
        wm_engine=os.getenv("WM_ENGINE", "fake").strip().lower(),
        yume_model_dir=Path(os.getenv("YUME_MODEL_DIR", "/models/Yume-5B-720P")),
        yume_chunk_frames=_get_int("YUME_CHUNK_FRAMES", 8),
        yume_base_prompt=os.getenv(
            "YUME_BASE_PROMPT",
            "POV of a character walking in a minecraft scene",
        ).strip(),
    )
