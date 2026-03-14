from __future__ import annotations

import asyncio
import logging
import sys
import types

import numpy as np
import pytest

from lucid import LoadContext, SessionContext, build_model_definition

from waypoint_modal_example.config import WaypointRuntimeConfig
from waypoint_modal_example.engine import WaypointControlState, WaypointEngine
from waypoint_modal_example.model import WaypointLucidModel


class _StubEngine:
    def __init__(self) -> None:
        self.prompt = ""
        self.start_prompts: list[str] = []
        self.updated_prompts: list[str] = []
        self.controls: list[WaypointControlState] = []
        self.generate_calls = 0
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
        self.generate_calls += 1
        self.controls.append(controls)
        fill = 200 if self.prompt == "new prompt" else 10
        frame = np.full((360, 640, 3), fill, dtype=np.uint8)
        return frame, 5.0

    async def end_session(self) -> None:
        self.end_calls += 1


def _runtime_config() -> WaypointRuntimeConfig:
    return WaypointRuntimeConfig(
        waypoint_model_source="/models/Waypoint-1.1-Small",
        waypoint_ae_source="/models/owl_vae_f16_c16_distill_v0_nogan",
        waypoint_prompt_encoder_source="/models/google-umt5-xl",
        waypoint_default_prompt="old prompt",
        waypoint_seed_image=None,
    )


def _build_model(engine: _StubEngine) -> WaypointLucidModel:
    model = WaypointLucidModel(_runtime_config())
    model.bind_runtime(None, logging.getLogger("tests.waypoint_model"))
    model._engine = engine
    return model


def test_waypoint_manifest_exposes_inputs() -> None:
    manifest = build_model_definition(WaypointLucidModel).to_manifest()
    input_names = {item["name"] for item in manifest["inputs"]}

    assert {
        "set_prompt",
        "forward",
        "backward",
        "left",
        "right",
        "jump",
        "sprint",
        "crouch",
        "primary_fire",
        "secondary_fire",
        "look",
        "scroll",
    }.issubset(input_names)


@pytest.mark.asyncio
async def test_session_persists_buttons_and_drains_transient_inputs() -> None:
    published: list[np.ndarray] = []
    ctx: SessionContext | None = None
    session = None
    engine = _StubEngine()
    model = _build_model(engine)

    async def publish_fn(_name: str, payload: object, _ts_ms: int | None) -> None:
        assert ctx is not None
        assert session is not None
        published.append(np.array(payload, copy=True))
        if len(published) == 1:
            session.set_prompt("new prompt")
            session.forward(True)
            session.look(10, -4)
            session.look(5, 1)
            session.scroll(120)
            session.scroll(120)
            return
        if len(published) == 3:
            ctx.running = False

    ctx = SessionContext(
        session_id="s1",
        room_name="wm-s1",
        outputs=model.outputs,
        publish_fn=publish_fn,
        logger=logging.getLogger("tests.waypoint_model"),
    )
    session = model.create_session(ctx)

    await asyncio.wait_for(session.run(), timeout=1.0)

    assert [int(frame[0, 0, 0]) for frame in published] == [10, 200, 200]
    assert engine.start_prompts == ["old prompt"]
    assert engine.updated_prompts == ["new prompt"]
    assert engine.controls[0] == WaypointControlState()
    assert engine.controls[1] == WaypointControlState(
        buttons=frozenset({0x57}),
        mouse_dx=15,
        mouse_dy=-3,
        scroll_amount=240,
    )
    assert engine.controls[2] == WaypointControlState(
        buttons=frozenset({0x57}),
        mouse_dx=0.0,
        mouse_dy=0.0,
        scroll_amount=0,
    )


@pytest.mark.asyncio
async def test_session_close_delegates_to_engine() -> None:
    engine = _StubEngine()
    model = _build_model(engine)
    ctx = SessionContext(
        session_id="s1",
        room_name="wm-s1",
        outputs=model.outputs,
        publish_fn=_noop_publish,
        logger=logging.getLogger("tests.waypoint_model"),
    )
    session = model.create_session(ctx)

    await session.close()

    assert engine.end_calls == 1


@pytest.mark.asyncio
async def test_session_pause_stops_new_generation_until_resume() -> None:
    published: list[np.ndarray] = []
    first_frame_paused = asyncio.Event()
    ctx: SessionContext | None = None
    engine = _StubEngine()
    model = _build_model(engine)

    async def publish_fn(_name: str, payload: object, _ts_ms: int | None) -> None:
        assert ctx is not None
        published.append(np.array(payload, copy=True))
        if len(published) == 1:
            ctx.pause()
            first_frame_paused.set()
            return
        if len(published) == 2:
            ctx.running = False

    ctx = SessionContext(
        session_id="s1",
        room_name="wm-s1",
        outputs=model.outputs,
        publish_fn=publish_fn,
        logger=logging.getLogger("tests.waypoint_model"),
    )
    session = model.create_session(ctx)

    task = asyncio.create_task(session.run())
    await asyncio.wait_for(first_frame_paused.wait(), timeout=1.0)

    await asyncio.sleep(0.05)
    assert engine.generate_calls == 1

    ctx.resume()
    await asyncio.wait_for(task, timeout=1.0)

    assert [int(frame[0, 0, 0]) for frame in published] == [10, 10]
    assert engine.start_prompts == ["old prompt"]
    assert engine.generate_calls == 2


@pytest.mark.asyncio
async def test_session_commits_compiler_cache_after_first_frame(
) -> None:
    engine = _StubEngine()
    model = _build_model(engine)
    commit_calls: list[str] = []
    model.compiler_cache_commit_hook = lambda _logger, reason: commit_calls.append(reason) or True

    ctx: SessionContext | None = None
    session = None

    async def publish_fn(_name: str, _payload: object, _ts_ms: int | None) -> None:
        assert ctx is not None
        if len(commit_calls) >= 1:
            ctx.running = False

    ctx = SessionContext(
        session_id="s1",
        room_name="wm-s1",
        outputs=model.outputs,
        publish_fn=publish_fn,
        logger=logging.getLogger("tests.waypoint_model"),
    )
    session = model.create_session(ctx)

    await asyncio.wait_for(session.run(), timeout=1.0)

    assert commit_calls == ["first_frame"]


@pytest.mark.asyncio
async def test_model_load_commits_compiler_cache_after_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit_calls: list[str] = []
    warmup_flags: list[bool] = []

    class _StubLoadEngine:
        def __init__(self, runtime_config, logger) -> None:
            self.runtime_config = runtime_config
            self.logger = logger

        async def load(self, *, warmup: bool = True) -> None:
            warmup_flags.append(warmup)
            return None
    monkeypatch.setattr("waypoint_modal_example.model.WaypointEngine", _StubLoadEngine)

    model = WaypointLucidModel(_runtime_config())
    model.bind_runtime(None, logging.getLogger("tests.waypoint_model"))
    model.compiler_cache_commit_hook = lambda _logger, reason: commit_calls.append(reason) or True

    await model.load(
        LoadContext(
            config=model.config,
            logger=logging.getLogger("tests.waypoint_model"),
        )
    )

    assert warmup_flags == [True]
    assert commit_calls == ["post_warmup"]


@pytest.mark.asyncio
async def test_engine_load_runs_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = WaypointEngine(_runtime_config(), logging.getLogger("tests.waypoint"))

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


@pytest.mark.asyncio
async def test_model_load_skips_warmup_when_compiler_cache_marker_matches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    commit_calls: list[str] = []
    warmup_flags: list[bool] = []

    class _StubLoadEngine:
        def __init__(self, runtime_config, logger) -> None:
            self.runtime_config = runtime_config
            self.logger = logger

        async def load(self, *, warmup: bool = True) -> None:
            warmup_flags.append(warmup)

    monkeypatch.setattr("waypoint_modal_example.model.WaypointEngine", _StubLoadEngine)
    monkeypatch.setenv("MODAL_COMPILER_CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("MODAL_GPU", "RTX-PRO-6000")
    monkeypatch.setenv("WORLD_ENGINE_COMMIT", "a30a00c302380c0f657347e8456bb6837ff37c22")

    model = WaypointLucidModel(_runtime_config())
    model.bind_runtime(None, logging.getLogger("tests.waypoint_model"))
    model.compiler_cache_commit_hook = lambda _logger, reason: commit_calls.append(reason) or True
    model._write_compiled_cache_marker()

    await model.load(
        LoadContext(
            config=model.config,
            logger=logging.getLogger("tests.waypoint_model"),
        )
    )

    assert warmup_flags == [False]
    assert commit_calls == []


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


def test_generate_frame_sync_reuses_single_cpu_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeDevice:
        type = "cpu"

    class _FakeTensor:
        def __init__(self, array: np.ndarray, *, contiguous: bool = True) -> None:
            self._array = array
            self.dtype = "uint8"
            self.device = _FakeDevice()
            self._contiguous = contiguous

        @property
        def shape(self) -> tuple[int, ...]:
            return self._array.shape

        def is_contiguous(self) -> bool:
            return self._contiguous

        def contiguous(self) -> "_FakeTensor":
            return _FakeTensor(np.ascontiguousarray(self._array), contiguous=True)

        def clamp(self, _min: int, _max: int) -> "_FakeTensor":
            return self

        def to(self, *, dtype=None, device=None) -> "_FakeTensor":
            _ = device
            if dtype is not None and dtype != "uint8":
                raise AssertionError(f"unexpected dtype conversion: {dtype}")
            return self

        def numpy(self) -> np.ndarray:
            return self._array

    fake_torch = types.SimpleNamespace(uint8="uint8")
    engine = WaypointEngine(_runtime_config(), logging.getLogger("tests.waypoint"))
    frame = _FakeTensor(np.zeros((360, 640, 3), dtype=np.uint8), contiguous=False)
    engine._engine = types.SimpleNamespace(gen_frame=lambda ctrl: frame)
    engine._ctrl_cls = lambda **kwargs: kwargs

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(engine, "_roll_session_if_needed_sync", lambda: None)

    generated = engine._generate_frame_sync(WaypointControlState())

    assert generated.flags.c_contiguous
    assert engine._last_frame is generated


async def _noop_publish(_name: str, _payload: object, _ts_ms: int | None) -> None:
    return None
