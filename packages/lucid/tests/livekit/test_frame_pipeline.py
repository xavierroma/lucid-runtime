from __future__ import annotations

import logging
import sys
import types
from math import isclose

import numpy as np
import pytest

from lucid import publish
import lucid.livekit.runner as livekit_module
from lucid.livekit.runner import _OutputSink, _RealLiveKitTransport


class _StubLiveKitAdapter:
    def __init__(self) -> None:
        self.published_video: list[tuple[str, np.ndarray, float]] = []

    async def publish_video(self, output_name: str, frame: np.ndarray) -> None:
        self.published_video.append((output_name, frame, livekit_module.time.monotonic()))

    async def publish_audio(self, output_name: str, samples: np.ndarray) -> None:
        _ = output_name
        _ = samples

    async def publish_data(
        self,
        output_name: str,
        payload: bytes,
        *,
        reliable: bool = True,
    ) -> None:
        _ = output_name
        _ = payload
        _ = reliable


@pytest.mark.asyncio
async def test_output_router_passes_video_frames_by_reference() -> None:
    adapter = _StubLiveKitAdapter()
    router = _OutputSink(outputs=(publish.video(name="video", width=2, height=2, fps=30),), livekit=adapter)
    frame = np.full((2, 2, 3), 7, dtype=np.uint8)

    await router.publish("video", frame)

    assert len(adapter.published_video) == 1
    assert adapter.published_video[0][0] == "video"
    assert adapter.published_video[0][1] is frame
    snapshot = router.snapshot()
    assert "effective_fps" in snapshot


@pytest.mark.asyncio
async def test_output_router_paces_publish_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": 100.0}

    async def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    monkeypatch.setattr(livekit_module.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(livekit_module.asyncio, "sleep", fake_sleep)

    adapter = _StubLiveKitAdapter()
    router = _OutputSink(outputs=(publish.video(name="video", width=2, height=2, fps=60),), livekit=adapter)

    await router.publish("video", np.zeros((2, 2, 3), dtype=np.uint8))
    await router.publish("video", np.ones((2, 2, 3), dtype=np.uint8))

    publish_times = [entry[2] for entry in adapter.published_video]
    assert isclose(publish_times[0], 100.0)
    assert isclose(publish_times[1], 100.0 + (1.0 / 60.0))


class _CaptureSource:
    def __init__(self) -> None:
        self.frames: list[object] = []

    def capture_frame(self, frame: object) -> None:
        self.frames.append(frame)


@pytest.mark.asyncio
async def test_real_livekit_publish_video_uses_original_frame_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_livekit = types.ModuleType("livekit")

    class _VideoBufferType:
        RGB24 = object()

    class _VideoFrame:
        def __init__(self, *, width: int, height: int, type: object, data: memoryview) -> None:
            self.width = width
            self.height = height
            self.type = type
            self.data = data

    fake_livekit.rtc = types.SimpleNamespace(
        VideoBufferType=_VideoBufferType,
        VideoFrame=_VideoFrame,
    )
    monkeypatch.setitem(sys.modules, "livekit", fake_livekit)

    adapter = _RealLiveKitTransport(
        livekit_url="wss://example.livekit.invalid",
        status_topic="wm.status",
        logger=logging.getLogger("tests.frame_pipeline"),
    )
    source = _CaptureSource()
    adapter._video_sources["video"] = source
    frame = np.arange(12, dtype=np.uint8).reshape((2, 2, 3))

    await adapter.publish_video("video", frame)

    captured = source.frames[0]
    assert isinstance(captured, _VideoFrame)
    assert captured.data.obj is frame
