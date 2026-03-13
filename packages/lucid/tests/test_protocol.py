from __future__ import annotations

import pytest

from lucid.protocol import (
    ControlMessageType,
    ProtocolError,
    StatusMessageType,
    encode_status_message,
    parse_control_message,
)


def test_parse_action_control_message() -> None:
    raw = (
        b'{"type":"action","seq":2,"ts_ms":10,'
        b'"session_id":"s1","payload":{"name":"set_prompt","args":{"prompt":"foggy valley"}}}'
    )
    envelope = parse_control_message(raw)
    assert envelope.type == ControlMessageType.ACTION
    assert envelope.seq == 2
    assert envelope.payload["name"] == "set_prompt"


def test_parse_resume_control_message() -> None:
    raw = b'{"type":"resume","seq":3,"ts_ms":20,"session_id":"s1","payload":{}}'
    envelope = parse_control_message(raw)
    assert envelope.type == ControlMessageType.RESUME
    assert envelope.seq == 3
    assert envelope.session_id == "s1"


def test_parse_pause_control_message() -> None:
    raw = b'{"type":"pause","seq":4,"ts_ms":30,"session_id":"s1","payload":{}}'
    envelope = parse_control_message(raw)
    assert envelope.type == ControlMessageType.PAUSE
    assert envelope.seq == 4
    assert envelope.session_id == "s1"


def test_parse_invalid_control_message() -> None:
    with pytest.raises(ProtocolError):
        parse_control_message(b"not-json")


def test_encode_status_message() -> None:
    encoded = encode_status_message(
        StatusMessageType.STARTED,
        session_id="abc",
        seq=3,
        payload={"worker_id": "w1"},
    )
    assert b'"type":"started"' in encoded
    assert b'"seq":3' in encoded
