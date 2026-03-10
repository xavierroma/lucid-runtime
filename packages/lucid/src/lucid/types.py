from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Assignment:
    session_id: str
    room_name: str
    worker_access_token: str
    control_topic: str


@dataclass(frozen=True, slots=True)
class OutputBinding:
    name: str
    kind: str
    track_name: str | None = None
    topic: str | None = None
    sample_rate_hz: int | None = None
    channels: int | None = None
    sample_format: str | None = None


@dataclass(slots=True)
class FrameMetrics:
    effective_fps: float
    queue_depth: int
    inference_ms_p50: float
    publish_dropped_frames: int


@dataclass(slots=True)
class SessionResult:
    error_code: str | None = None
    ended_by_control: bool = False
