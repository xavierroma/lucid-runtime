from __future__ import annotations

import pytest

from wm_worker.action_buffer import ActionBuffer
from wm_worker.models import ControlEnvelope, ControlMessageType


@pytest.mark.asyncio
async def test_action_buffer_latest_wins() -> None:
    buffer = ActionBuffer("base prompt")

    newer = ControlEnvelope(
        v="v1",
        type=ControlMessageType.ACTION,
        seq=10,
        ts_ms=10,
        session_id=None,
        payload={"keys": ["W"], "mouse_dx": 1},
    )
    older = ControlEnvelope(
        v="v1",
        type=ControlMessageType.ACTION,
        seq=9,
        ts_ms=9,
        session_id=None,
        payload={"keys": ["S"], "mouse_dx": 99},
    )

    await buffer.apply_control(newer)
    await buffer.apply_control(older)
    snapshot = await buffer.snapshot()
    assert snapshot.last_seq == 10
    assert snapshot.action.keys == ["W"]
    assert snapshot.action.mouse_dx == 1.0


@pytest.mark.asyncio
async def test_action_buffer_prompt_update() -> None:
    buffer = ActionBuffer("base prompt")
    message = ControlEnvelope(
        v="v1",
        type=ControlMessageType.SET_PROMPT,
        seq=1,
        ts_ms=1,
        session_id=None,
        payload={"prompt": "new prompt"},
    )
    await buffer.apply_control(message)
    snapshot = await buffer.snapshot()
    assert snapshot.prompt == "new prompt"
