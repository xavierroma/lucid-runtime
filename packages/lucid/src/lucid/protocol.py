from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class ControlMessageType(str, Enum):
    ACTION = "action"
    END = "end"
    PING = "ping"
    PAUSE = "pause"
    RESUME = "resume"


class StatusMessageType(str, Enum):
    STARTED = "started"
    FRAME_METRICS = "frame_metrics"
    ERROR = "error"
    ENDED = "ended"
    PONG = "pong"


@dataclass(slots=True)
class ControlEnvelope:
    type: ControlMessageType
    seq: int
    ts_ms: int
    session_id: str | None
    payload: dict[str, Any]


class _ControlEnvelopeModel(BaseModel):
    type: ControlMessageType
    seq: int = Field(ge=0)
    ts_ms: int = Field(ge=0)
    session_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class ProtocolError(ValueError):
    pass


def parse_control_message(raw: bytes | str) -> ControlEnvelope:
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid control JSON: {exc}") from exc
    try:
        envelope = _ControlEnvelopeModel.model_validate(parsed)
    except ValidationError as exc:
        raise ProtocolError(f"invalid control envelope: {exc}") from exc
    return ControlEnvelope(
        type=envelope.type,
        seq=envelope.seq,
        ts_ms=envelope.ts_ms,
        session_id=envelope.session_id,
        payload=envelope.payload,
    )


def encode_status_message(
    msg_type: StatusMessageType,
    *,
    session_id: str | None,
    seq: int,
    payload: dict[str, Any],
) -> bytes:
    envelope = {
        "type": msg_type.value,
        "seq": seq,
        "ts_ms": int(time.time() * 1000),
        "session_id": session_id,
        "payload": payload,
    }
    return json.dumps(envelope, separators=(",", ":"), sort_keys=False).encode("utf-8")
