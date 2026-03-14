from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    livekit_url: str
    status_topic: str = "wm.status"


@dataclass(frozen=True, slots=True)
class SessionConfig:
    worker_id: str = "wm-worker-1"


@dataclass(frozen=True, slots=True)
class Assignment:
    session_id: str
    room_name: str
    worker_access_token: str
    control_topic: str


@dataclass(slots=True)
class SessionResult:
    error_code: str | None = None
    ended_by_control: bool = False
