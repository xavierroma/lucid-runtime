from __future__ import annotations

import asyncio
import statistics
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable

import numpy as np

from wm_worker.models import FrameMetrics


@dataclass(slots=True)
class _FrameItem:
    frame: np.ndarray
    enqueued_at: float


class FramePipeline:
    def __init__(self, max_frames: int) -> None:
        self._queue: asyncio.Queue[_FrameItem] = asyncio.Queue(maxsize=max_frames)
        self._dropped_frames = 0
        self._inference_ms: deque[float] = deque(maxlen=128)
        self._published_frames = 0
        self._first_publish_ts: float | None = None

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    async def push(self, frame: np.ndarray, *, inference_ms: float) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._dropped_frames += 1
            except asyncio.QueueEmpty:
                pass
        await self._queue.put(_FrameItem(frame=frame, enqueued_at=time.monotonic()))
        self._inference_ms.append(inference_ms)

    async def pop(self, timeout_s: float) -> np.ndarray | None:
        try:
            item = await asyncio.wait_for(self._queue.get(), timeout=timeout_s)
            return item.frame
        except TimeoutError:
            return None

    async def publish_loop(
        self,
        publish_fn: Callable[[np.ndarray], Awaitable[None]],
        stop_event: asyncio.Event,
        target_fps: int,
        *,
        fallback_frame: np.ndarray,
    ) -> None:
        frame_period_s = 1.0 / max(target_fps, 1)
        last_frame = fallback_frame
        while not stop_event.is_set():
            loop_start = time.monotonic()
            next_frame = await self.pop(timeout_s=frame_period_s * 0.7)
            if next_frame is not None:
                last_frame = next_frame
            await publish_fn(last_frame)
            if self._first_publish_ts is None:
                self._first_publish_ts = time.monotonic()
            self._published_frames += 1
            elapsed = time.monotonic() - loop_start
            sleep_for = max(0.0, frame_period_s - elapsed)
            await asyncio.sleep(sleep_for)

    def metrics(self) -> FrameMetrics:
        if self._first_publish_ts is None:
            effective_fps = 0.0
        else:
            elapsed = max(time.monotonic() - self._first_publish_ts, 1e-6)
            effective_fps = self._published_frames / elapsed
        inference_ms_p50 = (
            statistics.median(self._inference_ms) if self._inference_ms else 0.0
        )
        return FrameMetrics(
            effective_fps=effective_fps,
            queue_depth=self._queue.qsize(),
            inference_ms_p50=inference_ms_p50,
            publish_dropped_frames=self._dropped_frames,
        )
