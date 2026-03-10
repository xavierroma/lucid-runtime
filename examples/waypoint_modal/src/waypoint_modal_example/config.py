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


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean, got: {raw}")


def _frame_defaults(model_source: str) -> tuple[int, int]:
    normalized = model_source.strip().lower()
    if "1.5" in normalized:
        return 1024, 512
    return 640, 360


@dataclass(frozen=True, slots=True)
class WaypointRuntimeConfig:
    livekit_url: str
    frame_width: int
    frame_height: int
    target_fps: int
    status_topic: str
    max_queue_frames: int
    livekit_mode: str
    wm_engine: str
    waypoint_model_source: str
    waypoint_ae_source: str
    waypoint_prompt_encoder_source: str
    waypoint_default_prompt: str
    waypoint_seed_image: Path | None
    waypoint_warmup_on_load: bool


def build_runtime_config(host_config: HostRuntimeConfig) -> WaypointRuntimeConfig:
    model_source = os.getenv("WAYPOINT_MODEL_SOURCE", "/models/Waypoint-1.1-Small").strip()
    default_width, default_height = _frame_defaults(model_source)
    seed_image_raw = os.getenv("WAYPOINT_SEED_IMAGE", "").strip()
    return WaypointRuntimeConfig(
        livekit_url=host_config.livekit_url,
        frame_width=_get_int("WM_FRAME_WIDTH", default_width),
        frame_height=_get_int("WM_FRAME_HEIGHT", default_height),
        target_fps=_get_int("WM_TARGET_FPS", 20),
        status_topic=host_config.status_topic,
        max_queue_frames=host_config.max_queue_frames,
        livekit_mode=host_config.livekit_mode,
        wm_engine=os.getenv("WM_ENGINE", "waypoint").strip().lower(),
        waypoint_model_source=model_source,
        waypoint_ae_source=os.getenv(
            "WAYPOINT_AE_SOURCE",
            "/models/owl_vae_f16_c16_distill_v0_nogan",
        ).strip(),
        waypoint_prompt_encoder_source=os.getenv(
            "WAYPOINT_PROMPT_ENCODER_SOURCE",
            "/models/google-umt5-xl",
        ).strip(),
        waypoint_default_prompt=os.getenv(
            "WAYPOINT_DEFAULT_PROMPT",
            "An explorable world with coherent geometry, stable lighting, and smooth forward motion.",
        ).strip(),
        waypoint_seed_image=Path(seed_image_raw) if seed_image_raw else None,
        waypoint_warmup_on_load=_get_bool("WAYPOINT_WARMUP_ON_LOAD", False),
    )
