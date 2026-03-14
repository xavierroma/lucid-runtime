from __future__ import annotations

import asyncio
import logging

import numpy as np
import pytest

from lucid import SessionContext, build_model_definition

from helios_modal_example.config import HeliosRuntimeConfig
from helios_modal_example.engine import ChunkResult
from helios_modal_example.model import HeliosLucidModel


class _StubEngine:
    def __init__(self) -> None:
        self.prompt = ""
        self.start_prompts: list[str] = []
        self.updated_prompts: list[str] = []
        self.generate_prompts: list[str] = []
        self.end_calls = 0

    async def load(self) -> None:
        return None

    async def start_session(self, prompt: str) -> None:
        self.prompt = prompt
        self.start_prompts.append(prompt)

    async def update_prompt(self, prompt: str) -> None:
        self.prompt = prompt
        self.updated_prompts.append(prompt)

    async def generate_chunk(self) -> ChunkResult:
        self.generate_prompts.append(self.prompt)
        base = 200 if self.prompt == "new prompt" else 10
        frames = [
            np.full((384, 640, 3), base + offset, dtype=np.uint8)
            for offset in range(2)
        ]
        return ChunkResult(frames=frames, chunk_ms=20.0, inference_ms=10.0)

    async def end_session(self) -> None:
        self.end_calls += 1


def _runtime_config() -> HeliosRuntimeConfig:
    return HeliosRuntimeConfig(
        backend="fake",
        helios_model_source="/models/Helios-Distilled",
        helios_default_prompt="old prompt",
        helios_negative_prompt="avoid blur",
        helios_chunk_frames=2,
        helios_guidance_scale=1.0,
        helios_pyramid_steps=(2, 2, 2),
        helios_amplify_first_chunk=True,
        helios_enable_group_offloading=False,
        helios_group_offloading_type="leaf_level",
        helios_max_sequence_length=512,
    )


def _build_model(engine: _StubEngine) -> HeliosLucidModel:
    model = HeliosLucidModel(_runtime_config())
    model.bind_runtime(None, logging.getLogger("tests.helios_model"))
    model._engine = engine
    return model


def test_helios_manifest_exposes_prompt_input_and_video_output() -> None:
    manifest = build_model_definition(HeliosLucidModel).to_manifest()

    assert manifest["model"]["name"] == "helios"
    assert [item["name"] for item in manifest["inputs"]] == ["set_prompt"]
    assert manifest["outputs"] == [
        {
            "fps": 24,
            "height": 384,
            "kind": "video",
            "name": "main_video",
            "pixel_format": "rgb24",
            "width": 640,
        }
    ]


@pytest.mark.asyncio
async def test_session_applies_prompt_changes_on_next_chunk_boundary() -> None:
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
            return
        if len(published) == 4:
            ctx.running = False

    ctx = SessionContext(
        session_id="s1",
        room_name="wm-s1",
        outputs=model.outputs,
        publish_fn=publish_fn,
        logger=logging.getLogger("tests.helios_model"),
    )
    session = model.create_session(ctx)

    await asyncio.wait_for(session.run(), timeout=1.0)

    assert [int(frame[0, 0, 0]) for frame in published] == [10, 11, 200, 201]
    assert engine.start_prompts == ["old prompt"]
    assert engine.updated_prompts == ["new prompt"]
    assert engine.generate_prompts == ["old prompt", "new prompt"]
    assert ctx.inference_ms_p50() == 10.0


@pytest.mark.asyncio
async def test_session_close_delegates_to_engine() -> None:
    engine = _StubEngine()
    model = _build_model(engine)
    ctx = SessionContext(
        session_id="s1",
        room_name="wm-s1",
        outputs=model.outputs,
        publish_fn=_noop_publish,
        logger=logging.getLogger("tests.helios_model"),
    )
    session = model.create_session(ctx)

    await session.close()

    assert engine.end_calls == 1


async def _noop_publish(_name: str, _payload: object, _ts_ms: int | None) -> None:
    return None
