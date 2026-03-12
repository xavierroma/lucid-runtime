from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    livekit_url: str
    frame_width: int = 1280
    frame_height: int = 720
    target_fps: int = 16
    status_topic: str = "wm.status"
    max_queue_frames: int = 32
    livekit_mode: str = "fake"


@dataclass(frozen=True, slots=True)
class SessionConfig:
    worker_id: str = "wm-worker-1"
    coordinator_base_url: str = ""
    worker_internal_token: str = ""
