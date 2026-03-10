from __future__ import annotations

import asyncio

import numpy as np
import pytest

from lucid.livekit import FramePipeline


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


@pytest.mark.asyncio
async def test_frame_pipeline_waits_for_first_real_frame() -> None:
    pipeline = FramePipeline(max_frames=2)
    published: list[int] = []
    stop_event = asyncio.Event()

    async def _publish(frame: np.ndarray) -> None:
        published.append(int(frame[0, 0, 0]))
        stop_event.set()

    task = asyncio.create_task(pipeline.publish_loop(_publish, stop_event, target_fps=8))
    await asyncio.sleep(0.05)
    assert published == []

    frame = np.full((2, 2, 3), 7, dtype=np.uint8)
    await pipeline.push(frame, inference_ms=5)
    await asyncio.wait_for(task, timeout=0.5)

    assert published == [7]


@pytest.mark.asyncio
async def test_frame_pipeline_does_not_replay_stale_frames() -> None:
    pipeline = FramePipeline(max_frames=2)
    published: list[int] = []
    stop_event = asyncio.Event()

    async def _publish(frame: np.ndarray) -> None:
        published.append(int(frame[0, 0, 0]))

    task = asyncio.create_task(pipeline.publish_loop(_publish, stop_event, target_fps=30))
    await pipeline.push(np.full((2, 2, 3), 3, dtype=np.uint8), inference_ms=5)
    await asyncio.sleep(0.2)
    stop_event.set()
    await asyncio.wait_for(task, timeout=0.5)

    assert published == [3]
