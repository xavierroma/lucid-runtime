from __future__ import annotations

import numpy as np
import pytest

from wm_worker.frame_pipeline import FramePipeline


@pytest.mark.asyncio
async def test_frame_pipeline_drops_oldest_when_full() -> None:
    pipeline = FramePipeline(max_frames=1)
    frame_a = np.zeros((2, 2, 3), dtype=np.uint8)
    frame_b = np.ones((2, 2, 3), dtype=np.uint8)

    await pipeline.push(frame_a, inference_ms=5)
    await pipeline.push(frame_b, inference_ms=6)

    popped = await pipeline.pop(timeout_s=0.01)
    assert popped is not None
    assert int(popped[0, 0, 0]) == 1
    assert pipeline.dropped_frames == 1


def test_frame_pipeline_metrics_defaults() -> None:
    pipeline = FramePipeline(max_frames=4)
    metrics = pipeline.metrics()
    assert metrics.queue_depth == 0
    assert metrics.effective_fps == 0.0
