from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import numpy as np
import pytest

from lucid.config import RuntimeConfig
from yume_modal_example.config import YumeRuntimeConfig, build_runtime_config
from yume_modal_example.engine import YumeEngine


@pytest.fixture
def fake_engine_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.invalid")
    monkeypatch.setenv("WM_ENGINE", "fake")
    monkeypatch.setenv("WM_LIVEKIT_MODE", "fake")
    monkeypatch.setenv("WM_FRAME_WIDTH", "64")
    monkeypatch.setenv("WM_FRAME_HEIGHT", "64")
    monkeypatch.setenv("YUME_CHUNK_FRAMES", "2")
    monkeypatch.setenv(
        "YUME_BASE_PROMPT",
        "POV of a character walking in a minecraft scene",
    )


def _build_engine() -> tuple[YumeEngine, RuntimeConfig]:
    host_config = RuntimeConfig.from_env()
    engine = YumeEngine(
        build_runtime_config(host_config),
        logging.getLogger("tests.yume_engine"),
    )
    return engine, host_config


@pytest.mark.asyncio
async def test_fake_engine_generates_rgb_frames_without_torch(fake_engine_env: None) -> None:
    engine, host_config = _build_engine()

    await engine.load()
    await engine.start_session("A snowy pine forest at dawn")
    chunk = await engine.generate_chunk()

    assert len(chunk.frames) == 2
    assert chunk.chunk_ms >= chunk.inference_ms
    assert chunk.inference_ms >= 0
    assert chunk.frames[0].shape == (
        host_config.frame_height,
        host_config.frame_width,
        3,
    )
    assert chunk.frames[0].dtype == np.uint8


@pytest.mark.asyncio
async def test_fake_engine_replaces_prompt_on_update(fake_engine_env: None) -> None:
    engine, _host_config = _build_engine()

    await engine.load()
    await engine.start_session("A bright desert canyon")
    first_chunk = await engine.generate_chunk()
    await engine.update_prompt("A rainy brutalist plaza at night")
    second_chunk = await engine.generate_chunk()

    assert engine._prompt == "A rainy brutalist plaza at night"
    assert not np.array_equal(first_chunk.frames[0], second_chunk.frames[0])


class _BlockingRuntime:
    def __init__(self) -> None:
        self.cached_prompts: list[str] = []
        self.reset_calls = 0

    def cache_prompt(self, prompt: str) -> None:
        time.sleep(0.1)
        self.cached_prompts.append(prompt)

    def reset_session_state(self) -> None:
        self.reset_calls += 1

    def generate_chunk(self, prompt: str, chunk_frames: int) -> list[np.ndarray]:
        time.sleep(0.1)
        _ = prompt
        return [
            np.zeros((64, 64, 3), dtype=np.uint8)
            for _ in range(chunk_frames)
        ]


@pytest.mark.asyncio
async def test_real_mode_generation_does_not_block_event_loop() -> None:
    engine = YumeEngine(
        YumeRuntimeConfig(
            livekit_url="wss://example.livekit.invalid",
            frame_width=64,
            frame_height=64,
            target_fps=8,
            status_topic="wm.status",
            max_queue_frames=4,
            livekit_mode="fake",
            wm_engine="yume",
            yume_model_dir=Path("/tmp/yume"),
            yume_chunk_frames=2,
            yume_base_prompt="A forest trail at dawn",
        ),
        logging.getLogger("tests.yume_engine"),
    )
    engine._loaded = True
    engine._runtime = _BlockingRuntime()

    ticks = 0

    async def _ticker() -> None:
        nonlocal ticks
        end = asyncio.get_running_loop().time() + 0.06
        while asyncio.get_running_loop().time() < end:
            ticks += 1
            await asyncio.sleep(0.01)

    await asyncio.gather(engine.start_session("A city street"), _ticker())
    assert ticks >= 3
    assert engine._runtime.cached_prompts == ["A city street"]
    assert engine._runtime.reset_calls == 1

    ticks = 0
    await asyncio.gather(engine.update_prompt("A misty harbor at sunrise"), _ticker())
    assert ticks >= 3
    assert engine._runtime.cached_prompts == ["A city street", "A misty harbor at sunrise"]
    assert engine._runtime.reset_calls == 1

    ticks = 0
    chunk, _ = await asyncio.gather(engine.generate_chunk(), _ticker())
    assert len(chunk.frames) == 2
    assert ticks >= 3
    await engine.end_session()
    assert engine._runtime.reset_calls == 2


def test_build_runtime_config_clamps_advertised_fps(fake_engine_env: None) -> None:
    host_config = RuntimeConfig.from_env()
    runtime_config = build_runtime_config(host_config)

    assert host_config.target_fps == 16
    assert runtime_config.target_fps == 2
