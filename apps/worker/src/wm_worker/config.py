from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    pass


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
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    worker_id: str
    coordinator_base_url: str
    worker_internal_token: str
    livekit_url: str
    yume_model_dir: Path
    yume_prompt_refiner_dir: Path | None
    hf_home: Path | None
    frame_width: int
    frame_height: int
    target_fps: int
    heartbeat_interval_ms: int
    assignment_poll_ms: int
    control_topic: str
    status_topic: str
    yume_chunk_frames: int
    yume_max_queue_frames: int
    worker_health_port: int
    yume_enable_prompt_refiner: bool
    yume_base_prompt: str
    wm_engine: str
    livekit_mode: str

    @classmethod
    def from_env(cls, worker_id_override: str | None = None) -> WorkerConfig:
        worker_id = worker_id_override or os.getenv("WORKER_ID", "wm-worker-1")
        coordinator_base_url = os.getenv("COORDINATOR_BASE_URL", "").strip().rstrip("/")
        worker_internal_token = os.getenv("WORKER_INTERNAL_TOKEN", "").strip()
        livekit_url = os.getenv("LIVEKIT_URL", "").strip()
        if not coordinator_base_url:
            raise ConfigError("COORDINATOR_BASE_URL is required")
        if not worker_internal_token:
            raise ConfigError("WORKER_INTERNAL_TOKEN is required")
        if not livekit_url:
            raise ConfigError("LIVEKIT_URL is required")

        yume_model_dir = Path(os.getenv("YUME_MODEL_DIR", "/models/Yume-5B-720P"))
        yume_prompt_refiner_raw = os.getenv("YUME_PROMPT_REFINER_DIR")
        yume_prompt_refiner_dir = (
            Path(yume_prompt_refiner_raw) if yume_prompt_refiner_raw else None
        )
        hf_home_raw = os.getenv("HF_HOME")
        hf_home = Path(hf_home_raw) if hf_home_raw else None

        return cls(
            worker_id=worker_id,
            coordinator_base_url=coordinator_base_url,
            worker_internal_token=worker_internal_token,
            livekit_url=livekit_url,
            yume_model_dir=yume_model_dir,
            yume_prompt_refiner_dir=yume_prompt_refiner_dir,
            hf_home=hf_home,
            frame_width=_get_int("WM_FRAME_WIDTH", 1280),
            frame_height=_get_int("WM_FRAME_HEIGHT", 720),
            target_fps=_get_int("WM_TARGET_FPS", 16),
            heartbeat_interval_ms=_get_int("WM_HEARTBEAT_INTERVAL_MS", 2000),
            assignment_poll_ms=_get_int("WM_ASSIGNMENT_POLL_MS", 1000),
            control_topic=os.getenv("WM_CONTROL_TOPIC", "wm.control.v1"),
            status_topic=os.getenv("WM_STATUS_TOPIC", "wm.status.v1"),
            yume_chunk_frames=_get_int("YUME_CHUNK_FRAMES", 8),
            yume_max_queue_frames=_get_int("YUME_MAX_QUEUE_FRAMES", 32),
            worker_health_port=_get_int("WORKER_HEALTH_PORT", 8090),
            yume_enable_prompt_refiner=_get_bool("YUME_ENABLE_PROMPT_REFINER", False),
            yume_base_prompt=os.getenv(
                "YUME_BASE_PROMPT", "An explorable realistic world"
            ).strip(),
            wm_engine=os.getenv("WM_ENGINE", "fake").strip().lower(),
            livekit_mode=os.getenv("WM_LIVEKIT_MODE", "fake").strip().lower(),
        )
