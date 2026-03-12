from __future__ import annotations

import logging
from typing import cast

import pytest

from lucid import LucidRuntime
from lucid.config import RuntimeConfig
from lucid.livekit import SessionControlReducer
from yume_modal_example.model import YumeLucidModel


def _runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        livekit_url="wss://example.livekit.invalid",
        frame_width=64,
        frame_height=64,
        livekit_mode="fake",
    )


async def _build_control() -> tuple[SessionControlReducer, LucidRuntime]:
    runtime = LucidRuntime.load_model(
        runtime_config=_runtime_config(),
        logger=logging.getLogger("tests.session_control"),
        model=YumeLucidModel,
    )
    session_ctx = runtime.create_session_context(
        session_id="s1",
        room_name="wm-s1",
        publish_fn=_noop_publish,
    )
    return (
        SessionControlReducer(runtime, session_ctx, logging.getLogger("tests.session_control")),
        runtime,
    )


async def _noop_publish(_name: str, _payload, _ts_ms: int | None) -> None:
    return None


@pytest.mark.asyncio
async def test_session_control_returns_pong_payload() -> None:
    control, _runtime = await _build_control()

    outcome = await control.reduce(
        b'{"type":"ping","seq":1,"ts_ms":10,"session_id":"s1","payload":{"client_ts_ms":42}}',
        session_id="s1",
    )

    assert outcome.stop_requested is False
    assert outcome.pong_payload == {"client_ts_ms": 42}


@pytest.mark.asyncio
async def test_session_control_ignores_other_session_messages() -> None:
    control, runtime = await _build_control()
    session_ctx = runtime.create_session_context(
        session_id="s1",
        room_name="wm-s1",
        publish_fn=_noop_publish,
    )
    control = SessionControlReducer(runtime, session_ctx, logging.getLogger("tests.session_control"))

    outcome = await control.reduce(
        b'{"type":"action","seq":1,"ts_ms":10,"session_id":"other","payload":{"name":"set_prompt","args":{"prompt":"new prompt"}}}',
        session_id="s1",
    )

    assert outcome.stop_requested is False
    assert outcome.pong_payload is None
    assert (
        cast(object, runtime._sessions["s1"]).prompt
        == "POV of a character walking in a minecraft scene"
    )


@pytest.mark.asyncio
async def test_session_control_dispatches_input() -> None:
    runtime = LucidRuntime.load_model(
        runtime_config=_runtime_config(),
        logger=logging.getLogger("tests.session_control"),
        model="yume_modal_example.model:YumeLucidModel",
    )
    session_ctx = runtime.create_session_context(
        session_id="s1",
        room_name="wm-s1",
        publish_fn=_noop_publish,
    )
    control = SessionControlReducer(runtime, session_ctx, logging.getLogger("tests.session_control"))

    await control.reduce(
        b'{"type":"action","seq":1,"ts_ms":10,"session_id":"s1","payload":{"name":"set_prompt","args":{"prompt":"new prompt"}}}',
        session_id="s1",
    )

    assert cast(object, runtime._sessions["s1"]).prompt == "new prompt"


@pytest.mark.asyncio
async def test_session_control_ignores_invalid_action_args() -> None:
    runtime = LucidRuntime.load_model(
        runtime_config=_runtime_config(),
        logger=logging.getLogger("tests.session_control"),
        model=YumeLucidModel,
    )
    session_ctx = runtime.create_session_context(
        session_id="s1",
        room_name="wm-s1",
        publish_fn=_noop_publish,
    )
    control = SessionControlReducer(runtime, session_ctx, logging.getLogger("tests.session_control"))

    outcome = await control.reduce(
        b'{"type":"action","seq":1,"ts_ms":10,"session_id":"s1","payload":{"name":"set_prompt","args":{}}}',
        session_id="s1",
    )

    assert outcome.stop_requested is False
    assert outcome.pong_payload is None
    assert (
        cast(object, runtime._sessions["s1"]).prompt
        == "POV of a character walking in a minecraft scene"
    )
