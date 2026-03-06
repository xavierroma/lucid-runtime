from __future__ import annotations

import logging

import pytest

from wm_worker.session_control import SessionControlReducer


@pytest.mark.asyncio
async def test_session_control_returns_pong_payload() -> None:
    control = SessionControlReducer("base prompt", logging.getLogger("tests.session_control"))

    outcome = await control.reduce(
        b'{"v":"v1","type":"ping","seq":1,"ts_ms":10,"session_id":"s1","payload":{"client_ts_ms":42}}',
        session_id="s1",
    )

    assert outcome.stop_requested is False
    assert outcome.pong_payload == {"client_ts_ms": 42}


@pytest.mark.asyncio
async def test_session_control_ignores_other_session_messages() -> None:
    control = SessionControlReducer("base prompt", logging.getLogger("tests.session_control"))

    outcome = await control.reduce(
        b'{"v":"v1","type":"set_prompt","seq":1,"ts_ms":10,"session_id":"other","payload":{"prompt":"new prompt"}}',
        session_id="s1",
    )
    snapshot = await control.snapshot()

    assert outcome.stop_requested is False
    assert outcome.pong_payload is None
    assert snapshot.prompt == "base prompt"
