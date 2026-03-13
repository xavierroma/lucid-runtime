from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from helios_modal_example.config import HeliosRuntimeConfig
from helios_modal_example.engine import HeliosEngine


class _StubPipeline:
    def __init__(self, height: int, width: int, frames_per_chunk: int) -> None:
        self._height = height
        self._width = width
        self._frames_per_chunk = frames_per_chunk
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        fill = 20 if kwargs.get("video") is None else 40
        frames = np.full(
            (1, self._frames_per_chunk, self._height, self._width, 3),
            fill + len(self.calls) - 1,
            dtype=np.uint8,
        )
        return SimpleNamespace(frames=frames)


def _config() -> HeliosRuntimeConfig:
    return HeliosRuntimeConfig(
        frame_width=8,
        frame_height=6,
        output_fps=24,
        wm_engine="helios",
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


@pytest.mark.asyncio
async def test_engine_uses_t2v_then_v2v_continuation() -> None:
    config = _config()
    engine = HeliosEngine(config, logging.getLogger("tests.helios_engine"))
    pipeline = _StubPipeline(config.frame_height, config.frame_width, config.helios_chunk_frames)
    engine._loaded = True
    engine._pipeline = pipeline
    engine._prompt = "old prompt"
    engine._generator = object()

    first = await engine.generate_chunk()
    second = await engine.generate_chunk()

    assert len(first.frames) == 2
    assert len(second.frames) == 2

    first_call = pipeline.calls[0]
    assert first_call["prompt"] == "old prompt"
    assert first_call["negative_prompt"] == "avoid blur"
    assert first_call["num_frames"] == 2
    assert first_call["pyramid_num_inference_steps_list"] == [2, 2, 2]
    assert first_call["is_amplify_first_chunk"] is True
    assert "video" not in first_call

    second_call = pipeline.calls[1]
    assert second_call["is_amplify_first_chunk"] is False
    assert isinstance(second_call["video"], np.ndarray)
    assert np.array_equal(second_call["video"], np.stack(first.frames, axis=0))
