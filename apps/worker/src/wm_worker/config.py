from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    livekit_url: str
    yume_model_dir: Path
    frame_width: int
    frame_height: int
    target_fps: int
    status_topic: str
    yume_chunk_frames: int
    yume_max_queue_frames: int
    yume_base_prompt: str
    wm_engine: str
    livekit_mode: str

    @classmethod
    def from_env(cls) -> RuntimeConfig:
        yume_model_dir = Path(os.getenv("YUME_MODEL_DIR", "/models/Yume-5B-720P"))

        return cls(
            livekit_url=_get_required("LIVEKIT_URL"),
            yume_model_dir=yume_model_dir,
            frame_width=_get_int("WM_FRAME_WIDTH", 1280),
            frame_height=_get_int("WM_FRAME_HEIGHT", 720),
            target_fps=_get_int("WM_TARGET_FPS", 16),
            status_topic=os.getenv("WM_STATUS_TOPIC", "wm.status.v1"),
            yume_chunk_frames=_get_int("YUME_CHUNK_FRAMES", 8),
            yume_max_queue_frames=_get_int("YUME_MAX_QUEUE_FRAMES", 32),
            yume_base_prompt=os.getenv(
                "YUME_BASE_PROMPT",
                (
                    "POV of a character walking in a minecraft scene"
                ),
            ).strip(),
            wm_engine=os.getenv("WM_ENGINE", "fake").strip().lower(),
            livekit_mode=os.getenv("WM_LIVEKIT_MODE", "fake").strip().lower(),
        )


@dataclass(frozen=True, slots=True)
class SessionConfig:
    worker_id: str
    coordinator_base_url: str
    worker_internal_token: str

    @classmethod
    def from_env(cls, worker_id_override: str | None = None) -> SessionConfig:
        return cls.from_values(
            worker_id=worker_id_override or os.getenv("WORKER_ID", "wm-worker-1"),
            coordinator_base_url=os.getenv("COORDINATOR_BASE_URL", ""),
            worker_internal_token=os.getenv("WORKER_INTERNAL_TOKEN", ""),
        )

    @classmethod
    def from_values(
        cls,
        *,
        worker_id: str,
        coordinator_base_url: str,
        worker_internal_token: str,
    ) -> SessionConfig:
        normalized_worker_id = worker_id.strip() or "wm-worker-1"
        normalized_base_url = coordinator_base_url.strip().rstrip("/")
        normalized_token = worker_internal_token.strip()
        if not normalized_base_url:
            raise ConfigError("COORDINATOR_BASE_URL is required")
        if not normalized_token:
            raise ConfigError("WORKER_INTERNAL_TOKEN is required")
        return cls(
            worker_id=normalized_worker_id,
            coordinator_base_url=normalized_base_url,
            worker_internal_token=normalized_token,
        )
