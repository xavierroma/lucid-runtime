from __future__ import annotations

import asyncio
import logging
import sys
import types

import numpy as np
from pydantic import BaseModel
import pytest

from lucid import SessionContext

from waypoint_modal_example.config import WaypointRuntimeConfig
from waypoint_modal_example.engine import WaypointControlState, WaypointEngine
from waypoint_modal_example.model import WaypointLucidModel


class _PromptState(BaseModel):
    prompt: str


class _ControlsState(BaseModel):
    forward: bool = False
    mouse_x: float = 0.0
    mouse_y: float = 0.0
    scroll_wheel: int = 0


class _StubEngine:
    def __init__(self) -> None:
        self.prompt = ""
        self.start_prompts: list[str] = []
        self.updated_prompts: list[str] = []
        self.controls: list[WaypointControlState] = []
        self.end_calls = 0

    async def load(self) -> None:
        return None

    async def start_session(self, prompt: str) -> None:
        self.prompt = prompt
        self.start_prompts.append(prompt)

    async def update_prompt(self, prompt: str) -> None:
        self.prompt = prompt
        self.updated_prompts.append(prompt)

    async def generate_frame(self, controls: WaypointControlState) -> tuple[np.ndarray, float]:
        self.controls.append(controls)
        fill = 200 if self.prompt == "new prompt" else 10
        frame = np.full((4, 4, 3), fill, dtype=np.uint8)
        return frame, 5.0

    async def end_session(self) -> None:
        self.end_calls += 1


def _runtime_config(*, warmup_on_load: bool = False) -> WaypointRuntimeConfig:
    return WaypointRuntimeConfig(
        livekit_url="wss://example.livekit.invalid",
        frame_width=4,
        frame_height=4,
        target_fps=100,
        status_topic="wm.status",
        max_queue_frames=4,
        livekit_mode="fake",
        wm_engine="waypoint",
        waypoint_model_source="/models/Waypoint-1.1-Small",
        waypoint_ae_source="/models/owl_vae_f16_c16_distill_v0_nogan",
        waypoint_prompt_encoder_source="/models/google-umt5-xl",
        waypoint_default_prompt="old prompt",
        waypoint_seed_image=None,
        waypoint_warmup_on_load=warmup_on_load,
    )


def _build_model(engine: _StubEngine) -> WaypointLucidModel:
    model = WaypointLucidModel({})
    model.bind_runtime(_runtime_config(), logging.getLogger("tests.waypoint_model"))
    model._engine = engine
    return model


@pytest.mark.asyncio
async def test_start_session_updates_prompt_and_controls() -> None:
    published: list[np.ndarray] = []
    ctx: SessionContext | None = None
    engine = _StubEngine()
    model = _build_model(engine)

    async def publish_fn(_name: str, payload: object, _ts_ms: int | None) -> None:
        assert ctx is not None
        published.append(np.array(payload, copy=True))
        if len(published) == 1:
            ctx.state.set("set_prompt", _PromptState(prompt="new prompt"))
            ctx.state.set(
                "set_controls",
                _ControlsState(forward=True, mouse_x=0.25, mouse_y=-0.5, scroll_wheel=1),
            )
            return
        ctx.running = False

    ctx = SessionContext(
        session_id="s1",
        room_name="wm-s1",
        outputs=model.resolve_outputs((WaypointLucidModel.main_video,)),
        publish_fn=publish_fn,
        logger=logging.getLogger("tests.waypoint_model"),
    )

    await asyncio.wait_for(model.start_session(ctx), timeout=1.0)

    assert [int(frame[0, 0, 0]) for frame in published] == [10, 200]
    assert engine.start_prompts == ["old prompt"]
    assert engine.updated_prompts == ["new prompt"]
    assert engine.controls[0] == WaypointControlState()
    assert engine.controls[1] == WaypointControlState(
        forward=True,
        mouse_x=0.25,
        mouse_y=-0.5,
        scroll_wheel=1,
    )


@pytest.mark.asyncio
async def test_start_session_commits_compiler_cache_after_first_frame(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    engine = _StubEngine()
    model = _build_model(engine)
    commit_calls: list[str] = []

    class _FakeVolume:
        def commit(self) -> None:
            commit_calls.append("commit")

    fake_modal = types.SimpleNamespace(
        Volume=types.SimpleNamespace(
            from_name=lambda name: commit_calls.append(name) or _FakeVolume()
        )
    )

    monkeypatch.setenv("MODAL_HF_CACHE_VOLUME", "lucid-hf-cache")
    monkeypatch.setenv("MODAL_COMPILER_CACHE_ROOT", str(tmp_path))
    monkeypatch.setitem(sys.modules, "modal", fake_modal)

    ctx: SessionContext | None = None

    async def publish_fn(_name: str, _payload: object, _ts_ms: int | None) -> None:
        assert ctx is not None
        if len(commit_calls) >= 2:
            ctx.running = False

    ctx = SessionContext(
        session_id="s1",
        room_name="wm-s1",
        outputs=model.resolve_outputs((WaypointLucidModel.main_video,)),
        publish_fn=publish_fn,
        logger=logging.getLogger("tests.waypoint_model"),
    )

    await asyncio.wait_for(model.start_session(ctx), timeout=1.0)

    assert commit_calls == ["lucid-hf-cache", "commit"]


@pytest.mark.asyncio
async def test_end_session_delegates_to_engine() -> None:
    engine = _StubEngine()
    model = _build_model(engine)

    async def publish_fn(_name: str, _payload: object, _ts_ms: int | None) -> None:
        return None

    ctx = SessionContext(
        session_id="s1",
        room_name="wm-s1",
        outputs=model.resolve_outputs((WaypointLucidModel.main_video,)),
        publish_fn=publish_fn,
        logger=logging.getLogger("tests.waypoint_model"),
    )

    await model.end_session(ctx)

    assert engine.end_calls == 1


@pytest.mark.asyncio
async def test_end_session_commits_compiler_cache_volume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    engine = _StubEngine()
    model = _build_model(engine)
    commit_calls: list[str] = []

    class _FakeVolume:
        def commit(self) -> None:
            commit_calls.append("commit")

    fake_modal = types.SimpleNamespace(
        Volume=types.SimpleNamespace(
            from_name=lambda name: commit_calls.append(name) or _FakeVolume()
        )
    )

    monkeypatch.setenv("MODAL_HF_CACHE_VOLUME", "lucid-hf-cache")
    monkeypatch.setenv("MODAL_COMPILER_CACHE_ROOT", str(tmp_path))
    monkeypatch.setitem(sys.modules, "modal", fake_modal)

    async def publish_fn(_name: str, _payload: object, _ts_ms: int | None) -> None:
        return None

    ctx = SessionContext(
        session_id="s1",
        room_name="wm-s1",
        outputs=model.resolve_outputs((WaypointLucidModel.main_video,)),
        publish_fn=publish_fn,
        logger=logging.getLogger("tests.waypoint_model"),
    )

    await model.end_session(ctx)

    assert engine.end_calls == 1
    assert commit_calls == ["lucid-hf-cache", "commit"]


@pytest.mark.asyncio
async def test_model_load_commits_compiler_cache_after_warmup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    commit_calls: list[str] = []

    class _FakeVolume:
        def commit(self) -> None:
            commit_calls.append("commit")

    class _StubLoadEngine:
        def __init__(self, runtime_config, logger) -> None:
            self.runtime_config = runtime_config
            self.logger = logger

        async def load(self) -> None:
            return None

    fake_modal = types.SimpleNamespace(
        Volume=types.SimpleNamespace(
            from_name=lambda name: commit_calls.append(name) or _FakeVolume()
        )
    )

    monkeypatch.setenv("MODAL_HF_CACHE_VOLUME", "lucid-hf-cache")
    monkeypatch.setenv("MODAL_COMPILER_CACHE_ROOT", str(tmp_path))
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setattr("waypoint_modal_example.model.WaypointEngine", _StubLoadEngine)

    model = WaypointLucidModel({})
    model.bind_runtime(_runtime_config(warmup_on_load=True), logging.getLogger("tests.waypoint_model"))

    await model.load()

    assert commit_calls == ["lucid-hf-cache", "commit"]


@pytest.mark.asyncio
async def test_engine_load_skips_warmup_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = WaypointEngine(_runtime_config(warmup_on_load=False), logging.getLogger("tests.waypoint"))

    async def immediate(fn):
        return fn()

    monkeypatch.setattr(engine, "_run_on_cuda_thread", immediate)
    monkeypatch.setattr(engine, "_load_engine_sync", lambda: setattr(engine, "_engine", object()))
    monkeypatch.setattr(
        engine,
        "_load_seed_frame",
        lambda: np.zeros((4, 4, 3), dtype=np.uint8),
    )

    warmup_calls: list[str] = []

    async def warmup() -> None:
        warmup_calls.append("warmup")

    monkeypatch.setattr(engine, "_warmup", warmup)

    await engine.load()

    assert warmup_calls == []


@pytest.mark.asyncio
async def test_engine_load_runs_warmup_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = WaypointEngine(_runtime_config(warmup_on_load=True), logging.getLogger("tests.waypoint"))

    async def immediate(fn):
        return fn()

    monkeypatch.setattr(engine, "_run_on_cuda_thread", immediate)
    monkeypatch.setattr(engine, "_load_engine_sync", lambda: setattr(engine, "_engine", object()))
    monkeypatch.setattr(
        engine,
        "_load_seed_frame",
        lambda: np.zeros((4, 4, 3), dtype=np.uint8),
    )

    warmup_calls: list[str] = []

    async def warmup() -> None:
        warmup_calls.append("warmup")

    monkeypatch.setattr(engine, "_warmup", warmup)

    await engine.load()

    assert warmup_calls == ["warmup"]


def test_engine_rolls_session_before_exceeding_frame_history(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = WaypointEngine(_runtime_config(), logging.getLogger("tests.waypoint"))
    engine._engine = types.SimpleNamespace(
        model_cfg=types.SimpleNamespace(n_frames=3),
        frame_ts=np.array([[3]], dtype=np.int64),
    )
    engine._seed_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    engine._last_frame = np.full((4, 4, 3), 9, dtype=np.uint8)
    engine._current_prompt = "roll prompt"

    rollover_calls: list[tuple[str, int]] = []

    def fake_reset(prompt: str, seed_frame: np.ndarray | None = None) -> None:
        assert seed_frame is not None
        rollover_calls.append((prompt, int(seed_frame[0, 0, 0])))

    monkeypatch.setattr(engine, "_reset_session_sync", fake_reset)

    engine._roll_session_if_needed_sync()

    assert rollover_calls == [("roll prompt", 9)]
