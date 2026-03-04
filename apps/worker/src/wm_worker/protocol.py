from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from wm_worker.models import ControlEnvelope, ControlMessageType, StatusMessageType


class _ControlEnvelopeModel(BaseModel):
    v: str = Field(default="v1")
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
        v=envelope.v,
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
        "v": "v1",
        "type": msg_type.value,
        "seq": seq,
        "ts_ms": int(time.time() * 1000),
        "session_id": session_id,
        "payload": payload,
    }
    return json.dumps(envelope, separators=(",", ":"), sort_keys=False).encode("utf-8")
