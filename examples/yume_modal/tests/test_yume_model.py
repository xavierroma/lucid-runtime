from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import numpy as np
import pytest

from lucid import SessionContext, build_model_definition

from yume_modal_example.config import YumeRuntimeConfig
from yume_modal_example.engine import ChunkResult
from yume_modal_example.model import YumeLucidModel


class _StubEngine:
    def __init__(self, generate_chunk_fn) -> None:
        self._generate_chunk_fn = generate_chunk_fn
        self.prompt = ""
        self.start_prompts: list[str] = []
        self.updated_prompts: list[str] = []
        self.generate_prompts: list[str] = []
        self.end_calls = 0

    async def start_session(self, prompt: str) -> None:
        self.prompt = prompt
        self.start_prompts.append(prompt)

    async def update_prompt(self, prompt: str) -> None:
        self.prompt = prompt
        self.updated_prompts.append(prompt)

    async def generate_chunk(self) -> ChunkResult:
        self.generate_prompts.append(self.prompt)
        return await self._generate_chunk_fn(self.prompt, len(self.generate_prompts))

    async def end_session(self) -> None:
        self.end_calls += 1


def _runtime_config() -> YumeRuntimeConfig:
    return YumeRuntimeConfig(
        frame_width=1280,
        frame_height=720,
        target_fps=100,
        wm_engine="fake",
        yume_model_dir=Path("/tmp/yume"),
        yume_chunk_frames=2,
        yume_base_prompt="old prompt",
    )


def _build_model(engine: _StubEngine) -> YumeLucidModel:
    model = YumeLucidModel(_runtime_config())
    model.bind_runtime(None, logging.getLogger("tests.yume_model"))
    model._engine = engine
    return model


def _frame(fill: int) -> np.ndarray:
    return np.full((720, 1280, 3), fill, dtype=np.uint8)


def test_yume_manifest_exposes_inputs() -> None:
    manifest = build_model_definition(YumeLucidModel).to_manifest()
    input_names = {item["name"] for item in manifest["inputs"]}

    assert {"set_prompt"}.issubset(input_names)


@pytest.mark.asyncio
async def test_session_drops_stale_chunk_when_prompt_changes_during_generation() -> None:
    published: list[np.ndarray] = []
    ctx: SessionContext | None = None
    session = None

    async def generate_chunk(prompt: str, call_count: int) -> ChunkResult:
        assert session is not None
        if call_count == 1:
            session.set_prompt("new prompt")
            return ChunkResult(
                frames=[_frame(10)],
                chunk_ms=1.0,
                inference_ms=1.0,
            )
        fill = 200 if prompt == "new prompt" else 10
        return ChunkResult(
            frames=[_frame(fill)],
            chunk_ms=1.0,
            inference_ms=1.0,
        )

    engine = _StubEngine(generate_chunk)
    model = _build_model(engine)

    async def publish_fn(_name: str, payload: object, _ts_ms: int | None) -> None:
        assert ctx is not None
        published.append(np.array(payload, copy=True))
        ctx.running = False

    ctx = SessionContext(
        session_id="s1",
        room_name="wm-s1",
        outputs=model.outputs,
        publish_fn=publish_fn,
        logger=logging.getLogger("tests.yume_model"),
    )
    session = model.create_session(ctx)

    await asyncio.wait_for(session.run(), timeout=1.0)

    assert [int(frame[0, 0, 0]) for frame in published] == [200]
    assert engine.start_prompts == ["old prompt"]
    assert engine.updated_prompts == ["new prompt"]
    assert engine.generate_prompts == ["old prompt", "new prompt"]


@pytest.mark.asyncio
async def test_session_stops_publishing_old_chunk_after_prompt_update() -> None:
    published: list[np.ndarray] = []
    ctx: SessionContext | None = None
    session = None

    async def generate_chunk(prompt: str, _call_count: int) -> ChunkResult:
        if prompt == "new prompt":
            frames = [_frame(200)]
        else:
            frames = [_frame(10), _frame(10)]
        return ChunkResult(
            frames=frames,
            chunk_ms=1.0,
            inference_ms=1.0,
        )

    engine = _StubEngine(generate_chunk)
    model = _build_model(engine)

    async def publish_fn(_name: str, payload: object, _ts_ms: int | None) -> None:
        assert ctx is not None
        assert session is not None
        frame = np.array(payload, copy=True)
        published.append(frame)
        if len(published) == 1:
            session.set_prompt("new prompt")
            return
        ctx.running = False

    ctx = SessionContext(
        session_id="s1",
        room_name="wm-s1",
        outputs=model.outputs,
        publish_fn=publish_fn,
        logger=logging.getLogger("tests.yume_model"),
    )
    session = model.create_session(ctx)

    await asyncio.wait_for(session.run(), timeout=1.0)

    assert [int(frame[0, 0, 0]) for frame in published] == [10, 200]
    assert engine.updated_prompts == ["new prompt"]
    assert engine.generate_prompts == ["old prompt", "new prompt"]


@pytest.mark.asyncio
async def test_session_close_delegates_to_engine() -> None:
    engine = _StubEngine(lambda _prompt, _count: asyncio.sleep(0))
    model = _build_model(engine)
    ctx = SessionContext(
        session_id="s1",
        room_name="wm-s1",
        outputs=model.outputs,
        publish_fn=_noop_publish,
        logger=logging.getLogger("tests.yume_model"),
    )
    session = model.create_session(ctx)

    await session.close()

    assert engine.end_calls == 1


@pytest.mark.asyncio
async def test_session_resume_continues_paused_chunk_before_applying_new_prompt() -> None:
    published: list[np.ndarray] = []
    first_frame_paused = asyncio.Event()
    ctx: SessionContext | None = None
    session = None

    async def generate_chunk(prompt: str, _call_count: int) -> ChunkResult:
        frames = [_frame(200)] if prompt == "new prompt" else [_frame(10), _frame(11)]
        return ChunkResult(
            frames=frames,
            chunk_ms=1.0,
            inference_ms=1.0,
        )

    engine = _StubEngine(generate_chunk)
    model = _build_model(engine)

    async def publish_fn(_name: str, payload: object, _ts_ms: int | None) -> None:
        assert ctx is not None
        assert session is not None
        published.append(np.array(payload, copy=True))
        if len(published) == 1:
            ctx.pause()
            session.set_prompt("new prompt")
            first_frame_paused.set()
            return
        if len(published) == 3:
            ctx.running = False

    ctx = SessionContext(
        session_id="s1",
        room_name="wm-s1",
        outputs=model.outputs,
        publish_fn=publish_fn,
        logger=logging.getLogger("tests.yume_model"),
    )
    session = model.create_session(ctx)

    task = asyncio.create_task(session.run())
    await asyncio.wait_for(first_frame_paused.wait(), timeout=1.0)

    await asyncio.sleep(0.05)
    assert engine.generate_prompts == ["old prompt"]

    ctx.resume()
    await asyncio.wait_for(task, timeout=1.0)

    assert [int(frame[0, 0, 0]) for frame in published] == [10, 11, 200]
    assert engine.start_prompts == ["old prompt"]
    assert engine.updated_prompts == ["new prompt"]
    assert engine.generate_prompts == ["old prompt", "new prompt"]


async def _noop_publish(_name: str, _payload: object, _ts_ms: int | None) -> None:
    return None
