from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class WorkerLifecycleState(str, Enum):
    BOOTING = "BOOTING"
    IDLE = "IDLE"
    BUSY = "BUSY"
    STOPPING = "STOPPING"


class ControlMessageType(str, Enum):
    ACTION = "action"
    SET_PROMPT = "set_prompt"
    END = "end"
    PING = "ping"


class StatusMessageType(str, Enum):
    STARTED = "started"
    BUSY = "busy"
    FRAME_METRICS = "frame_metrics"
    ERROR = "error"
    ENDED = "ended"
    PONG = "pong"


@dataclass(slots=True)
class Assignment:
    session_id: str
    room_name: str
    worker_access_token: str
    video_track_name: str
    control_topic: str


@dataclass(slots=True)
class ActionPayload:
    keys: list[str] = field(default_factory=list)
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    actual_distance: float = 1.0
    angular_change_rate: float = 1.0
    view_rotation_speed: float = 1.0


@dataclass(slots=True)
class ControlEnvelope:
    v: str
    type: ControlMessageType
    seq: int
    ts_ms: int
    session_id: str | None
    payload: dict[str, Any]


@dataclass(slots=True)
class ActionSnapshot:
    prompt: str
    action: ActionPayload
    last_seq: int
    updated_at_ms: int


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
