from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import numpy as np
from pydantic import BaseModel
import pytest

from lucid import SessionContext

from yume_modal_example.config import YumeRuntimeConfig
from yume_modal_example.engine import ChunkResult
from yume_modal_example.model import YumeLucidModel


class _PromptState(BaseModel):
    prompt: str


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
        livekit_url="wss://example.livekit.invalid",
        frame_width=4,
        frame_height=4,
        target_fps=100,
        status_topic="wm.status",
        max_queue_frames=4,
        livekit_mode="fake",
        wm_engine="fake",
        yume_model_dir=Path("/tmp/yume"),
        yume_chunk_frames=2,
        yume_base_prompt="old prompt",
    )


def _build_model(engine: _StubEngine) -> YumeLucidModel:
    model = YumeLucidModel({})
    model.bind_runtime(_runtime_config(), logging.getLogger("tests.yume_model"))
    model._engine = engine
    return model


def _frame(fill: int) -> np.ndarray:
    return np.full((4, 4, 3), fill, dtype=np.uint8)


@pytest.mark.asyncio
async def test_start_session_drops_stale_chunk_when_prompt_changes_during_generation() -> None:
    published: list[np.ndarray] = []
    ctx: SessionContext | None = None

    async def generate_chunk(prompt: str, call_count: int) -> ChunkResult:
        assert ctx is not None
        if call_count == 1:
            ctx.state.set("set_prompt", _PromptState(prompt="new prompt"))
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
        outputs=model.resolve_outputs((YumeLucidModel.main_video,)),
        publish_fn=publish_fn,
        logger=logging.getLogger("tests.yume_model"),
    )

    await asyncio.wait_for(model.start_session(ctx), timeout=1.0)

    assert [int(frame[0, 0, 0]) for frame in published] == [200]
    assert engine.start_prompts == ["old prompt"]
    assert engine.updated_prompts == ["new prompt"]
    assert engine.generate_prompts == ["old prompt", "new prompt"]


@pytest.mark.asyncio
async def test_start_session_stops_publishing_old_chunk_after_prompt_update() -> None:
    published: list[np.ndarray] = []
    ctx: SessionContext | None = None

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
        frame = np.array(payload, copy=True)
        published.append(frame)
        if len(published) == 1:
            ctx.state.set("set_prompt", _PromptState(prompt="new prompt"))
            return
        ctx.running = False

    ctx = SessionContext(
        session_id="s1",
        room_name="wm-s1",
        outputs=model.resolve_outputs((YumeLucidModel.main_video,)),
        publish_fn=publish_fn,
        logger=logging.getLogger("tests.yume_model"),
    )

    await asyncio.wait_for(model.start_session(ctx), timeout=1.0)

    assert [int(frame[0, 0, 0]) for frame in published] == [10, 200]
    assert engine.updated_prompts == ["new prompt"]
    assert engine.generate_prompts == ["old prompt", "new prompt"]
