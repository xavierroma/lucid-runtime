from __future__ import annotations

import pytest

from wm_worker.models import ControlMessageType, StatusMessageType
from wm_worker.protocol import ProtocolError, encode_status_message, parse_control_message


def test_parse_action_control_message() -> None:
    raw = (
        b'{"v":"v1","type":"action","seq":2,"ts_ms":10,'
        b'"session_id":"s1","payload":{"keys":["W"],"mouse_dx":1}}'
    )
    envelope = parse_control_message(raw)
    assert envelope.type == ControlMessageType.ACTION
    assert envelope.seq == 2
    assert envelope.payload["keys"] == ["W"]


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
