from __future__ import annotations

import pytest

from lucid.livekit.runner import _ProtocolError, _encode_status_message, _parse_control_message


def test_parse_action_control_message() -> None:
    raw = (
        b'{"type":"action","seq":2,"ts_ms":10,'
        b'"session_id":"s1","payload":{"name":"set_prompt","args":{"prompt":"foggy valley"}}}'
    )
    message = _parse_control_message(raw)
    assert message.kind == "action"
    assert message.seq == 2
    assert message.payload["name"] == "set_prompt"


def test_parse_resume_control_message() -> None:
    raw = b'{"type":"resume","seq":3,"ts_ms":20,"session_id":"s1","payload":{}}'
    message = _parse_control_message(raw)
    assert message.kind == "resume"
    assert message.seq == 3
    assert message.session_id == "s1"


def test_parse_pause_control_message() -> None:
    raw = b'{"type":"pause","seq":4,"ts_ms":30,"session_id":"s1","payload":{}}'
    message = _parse_control_message(raw)
    assert message.kind == "pause"
    assert message.seq == 4
    assert message.session_id == "s1"


def test_parse_invalid_control_message() -> None:
    with pytest.raises(_ProtocolError):
        _parse_control_message(b"not-json")


def test_encode_status_message() -> None:
    encoded = _encode_status_message(
        "started",
        session_id="abc",
        seq=3,
        payload={"worker_id": "w1"},
    )
    assert b'"type":"started"' in encoded
    assert b'"seq":3' in encoded
