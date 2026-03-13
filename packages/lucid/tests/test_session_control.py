from __future__ import annotations

import logging
from typing import cast

import pytest

from lucid import (
    LucidModel,
    LucidRuntime,
    LucidSession,
    axis,
    hold,
    input,
    pointer,
    press,
    publish,
    wheel,
)
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


class PausePolicySession(LucidSession["PausePolicyModel"]):
    def __init__(self, model: "PausePolicyModel", ctx) -> None:
        super().__init__(model, ctx)
        self.prompt = "initial prompt"
        self.hold_pressed = False
        self.axis_value = 0.0
        self.press_count = 0
        self.pointer_events: list[tuple[float, float]] = []
        self.wheel_events: list[float] = []

    @input(description="Update prompt.")
    def set_prompt(self, prompt: str) -> None:
        self.prompt = prompt

    @input(binding=hold(keys=("KeyH",)))
    def hold_action(self, pressed: bool) -> None:
        self.hold_pressed = pressed

    @input(binding=axis(positive_keys=("KeyL",), negative_keys=("KeyJ",)))
    def axis_action(self, value: float) -> None:
        self.axis_value = float(value)

    @input(binding=press(keys=("Space",)))
    def press_action(self) -> None:
        self.press_count += 1

    @input(binding=pointer())
    def look(self, dx: float, dy: float) -> None:
        self.pointer_events.append((float(dx), float(dy)))

    @input(binding=wheel())
    def scroll(self, delta: float) -> None:
        self.wheel_events.append(float(delta))

    async def run(self) -> None:
        self.ctx.running = False


class PausePolicyModel(LucidModel):
    name = "pause-policy"
    outputs = (publish.bytes(name="state"),)
    session_cls = PausePolicySession

    def create_session(self, ctx) -> PausePolicySession:
        return PausePolicySession(self, ctx)


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
async def test_session_control_returns_pause_outcome() -> None:
    control, _runtime = await _build_control()

    outcome = await control.reduce(
        b'{"type":"pause","seq":1,"ts_ms":10,"session_id":"s1","payload":{}}',
        session_id="s1",
    )

    assert outcome.pause_requested is True
    assert outcome.resume_requested is False
    assert outcome.stop_requested is False


@pytest.mark.asyncio
async def test_session_control_returns_resume_outcome() -> None:
    control, _runtime = await _build_control()

    outcome = await control.reduce(
        b'{"type":"resume","seq":1,"ts_ms":10,"session_id":"s1","payload":{}}',
        session_id="s1",
    )

    assert outcome.pause_requested is False
    assert outcome.resume_requested is True
    assert outcome.stop_requested is False


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


@pytest.mark.asyncio
async def test_session_control_allows_only_persistent_inputs_while_paused() -> None:
    runtime = LucidRuntime.load_model(
        runtime_config=_runtime_config(),
        logger=logging.getLogger("tests.session_control"),
        model=PausePolicyModel,
    )
    session_ctx = runtime.create_session_context(
        session_id="s1",
        room_name="wm-s1",
        publish_fn=_noop_publish,
    )
    session_ctx.pause()
    control = SessionControlReducer(runtime, session_ctx, logging.getLogger("tests.session_control"))

    await control.reduce(
        b'{"type":"action","seq":1,"ts_ms":10,"session_id":"s1","payload":{"name":"set_prompt","args":{"prompt":"paused prompt"}}}',
        session_id="s1",
    )
    await control.reduce(
        b'{"type":"action","seq":2,"ts_ms":10,"session_id":"s1","payload":{"name":"hold_action","args":{"pressed":true}}}',
        session_id="s1",
    )
    await control.reduce(
        b'{"type":"action","seq":3,"ts_ms":10,"session_id":"s1","payload":{"name":"axis_action","args":{"value":1}}}',
        session_id="s1",
    )
    await control.reduce(
        b'{"type":"action","seq":4,"ts_ms":10,"session_id":"s1","payload":{"name":"press_action","args":{}}}',
        session_id="s1",
    )
    await control.reduce(
        b'{"type":"action","seq":5,"ts_ms":10,"session_id":"s1","payload":{"name":"look","args":{"dx":4,"dy":-2}}}',
        session_id="s1",
    )
    await control.reduce(
        b'{"type":"action","seq":6,"ts_ms":10,"session_id":"s1","payload":{"name":"scroll","args":{"delta":120}}}',
        session_id="s1",
    )

    session = cast(PausePolicySession, runtime._sessions["s1"])
    assert session.prompt == "paused prompt"
    assert session.hold_pressed is True
    assert session.axis_value == 1.0
    assert session.press_count == 0
    assert session.pointer_events == []
    assert session.wheel_events == []
