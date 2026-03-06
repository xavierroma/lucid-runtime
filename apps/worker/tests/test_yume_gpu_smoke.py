from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pytest

from wm_worker.config import RuntimeConfig
from wm_worker.yume_engine import YumeEngine


@pytest.mark.asyncio
async def test_yume_gpu_smoke_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.getenv("RUN_YUME_GPU_TESTS") != "1":
        pytest.skip("set RUN_YUME_GPU_TESTS=1 to run GPU smoke tests")

    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        pytest.skip(f"torch not available: {exc}")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    model_dir = os.getenv("YUME_GPU_TEST_MODEL_DIR") or os.getenv("YUME_MODEL_DIR")
    if not model_dir:
        pytest.skip("set YUME_GPU_TEST_MODEL_DIR (or YUME_MODEL_DIR) for GPU smoke test")
    model_path = Path(model_dir)
    if not model_path.exists():
        pytest.skip(f"model directory does not exist: {model_path}")

    monkeypatch.setenv("COORDINATOR_BASE_URL", "http://localhost:8080")
    monkeypatch.setenv("WORKER_INTERNAL_TOKEN", "test-token")
    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.invalid")
    monkeypatch.setenv("WM_ENGINE", "yume")
    monkeypatch.setenv("WM_LIVEKIT_MODE", "fake")
    monkeypatch.setenv("YUME_MODEL_DIR", str(model_path))
    monkeypatch.setenv("YUME_CHUNK_FRAMES", "2")
    monkeypatch.setenv("WM_FRAME_WIDTH", "1280")
    monkeypatch.setenv("WM_FRAME_HEIGHT", "704")

    config = RuntimeConfig.from_env()
    engine = YumeEngine(config, logging.getLogger("tests.yume_gpu_smoke"))

    await engine.load()
    await engine.start_session("A first-person walk through a city street")
    chunk = await engine.generate_chunk()

    assert len(chunk.frames) == config.yume_chunk_frames
    assert chunk.inference_ms > 0

    first = chunk.frames[0]
    assert isinstance(first, np.ndarray)
    assert first.dtype == np.uint8
    assert first.ndim == 3
    assert first.shape[2] == 3
    assert first.size > 0

    await engine.end_session()
