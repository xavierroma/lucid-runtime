from __future__ import annotations

import logging

import pytest

from lucid import LucidRuntime

from lucid.config import RuntimeConfig
from lucid.livekit import SessionControlReducer


@pytest.fixture
def worker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.invalid")
    monkeypatch.setenv("WM_ENGINE", "fake")
    monkeypatch.setenv("WM_LIVEKIT_MODE", "fake")
    monkeypatch.setenv("YUME_MODEL_DIR", "/tmp/yume-model")
    monkeypatch.setenv("YUME_CHUNK_FRAMES", "1")
    monkeypatch.setenv("WM_FRAME_WIDTH", "64")
    monkeypatch.setenv("WM_FRAME_HEIGHT", "64")


async def _build_control() -> tuple[SessionControlReducer, LucidRuntime]:
    runtime = LucidRuntime.load_selected(
        runtime_config=RuntimeConfig.from_env(),
        logger=logging.getLogger("tests.session_control"),
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
async def test_session_control_returns_pong_payload(worker_env: None) -> None:
    control, _runtime = await _build_control()

    outcome = await control.reduce(
        b'{"type":"ping","seq":1,"ts_ms":10,"session_id":"s1","payload":{"client_ts_ms":42}}',
        session_id="s1",
    )

    assert outcome.stop_requested is False
    assert outcome.pong_payload == {"client_ts_ms": 42}


@pytest.mark.asyncio
async def test_session_control_ignores_other_session_messages(worker_env: None) -> None:
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
    assert session_ctx.state.get("set_prompt") is None


@pytest.mark.asyncio
async def test_session_control_dispatches_state_action(worker_env: None) -> None:
    runtime = LucidRuntime.load_selected(
        runtime_config=RuntimeConfig.from_env(),
        logger=logging.getLogger("tests.session_control"),
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

    assert session_ctx.state.set_prompt.prompt == "new prompt"


@pytest.mark.asyncio
async def test_session_control_ignores_invalid_action_args(worker_env: None) -> None:
    runtime = LucidRuntime.load_selected(
        runtime_config=RuntimeConfig.from_env(),
        logger=logging.getLogger("tests.session_control"),
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
    assert session_ctx.state.get("set_prompt") is None
