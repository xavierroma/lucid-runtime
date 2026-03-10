from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass

import numpy as np

from .config import YumeRuntimeConfig
from .single_gpu_runtime import YumeSingleGpuRuntime, YumeSingleGpuRuntimeError


class YumeEngineError(RuntimeError):
    pass


@dataclass(slots=True)
class ChunkResult:
    frames: list[np.ndarray]
    chunk_ms: float
    inference_ms: float


class YumeEngine:
    def __init__(self, config: YumeRuntimeConfig, logger: logging.Logger) -> None:
        self._config = config
        self._logger = logger
        self._frame_idx = 0
        self._prompt = _normalize_prompt(config.yume_base_prompt, config.yume_base_prompt)
        self._device = "cpu"
        self._loaded = False
        self._runtime: YumeSingleGpuRuntime | None = None

    async def load(self) -> None:
        if self._loaded:
            self._logger.info("reusing preloaded yume engine on %s", self._device)
            return

        if self._config.wm_engine not in {"fake", "yume"}:
            raise YumeEngineError(
                f"unsupported WM_ENGINE={self._config.wm_engine}; expected fake or yume"
            )

        if self._config.wm_engine == "fake":
            self._logger.info("starting in fake engine mode")
            self._loaded = True
            return

        load_started = time.perf_counter()
        try:
            import torch
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            raise YumeEngineError("failed importing torch for WM_ENGINE=yume") from exc

        if not torch.cuda.is_available():
            raise YumeEngineError("CUDA is required for WM_ENGINE=yume")

        if not self._config.yume_model_dir.exists():
            raise YumeEngineError(
                f"YUME_MODEL_DIR does not exist: {self._config.yume_model_dir}"
            )

        required_files = [
            "diffusion_pytorch_model.safetensors",
            "config.json",
            "config.yaml",
            "Wan2.2_VAE.pth",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "google/umt5-xxl/tokenizer.json",
            "google/umt5-xxl/spiece.model",
        ]
        missing = [
            file_name
            for file_name in required_files
            if not (self._config.yume_model_dir / file_name).exists()
        ]
        if missing:
            raise YumeEngineError(
                f"YUME model directory is missing required files: {', '.join(missing)}"
            )

        self._device = "cuda"
        torch.backends.cuda.matmul.allow_tf32 = True
        runtime: YumeSingleGpuRuntime = YumeSingleGpuRuntime(
            model_dir=self._config.yume_model_dir,
            frame_width=self._config.frame_width,
            frame_height=self._config.frame_height,
            device=self._device,
            logger=self._logger,
        )
        try:
            runtime.load()
            runtime.cache_prompt(self._config.yume_base_prompt)
        except YumeSingleGpuRuntimeError as exc:
            raise YumeEngineError(str(exc)) from exc

        self._runtime = runtime
        self._loaded = True
        self._logger.info(
            "yume engine initialized in single-GPU mode on %s load_ms=%.2f cache_stats=%s",
            self._device,
            (time.perf_counter() - load_started) * 1000,
            runtime.prompt_cache_stats,
        )

    async def start_session(self, prompt: str | None) -> None:
        if not self._loaded:
            raise YumeEngineError("engine must be loaded before starting session")
        self._frame_idx = 0
        self._prompt = _normalize_prompt(prompt, self._config.yume_base_prompt)
        if self._runtime is not None:
            self._runtime.reset_session_state()
            await asyncio.to_thread(self._runtime.cache_prompt, self._prompt)

    async def update_prompt(self, prompt: str) -> None:
        next_prompt = _normalize_prompt(prompt, self._config.yume_base_prompt)
        if next_prompt == self._prompt:
            return
        self._prompt = next_prompt
        if self._runtime is not None:
            await asyncio.to_thread(self._runtime.cache_prompt, self._prompt)
        self._logger.info(
            "updated yume prompt prompt_len=%s",
            len(self._prompt),
        )

    async def generate_chunk(self) -> ChunkResult:
        if not self._loaded:
            raise YumeEngineError("engine must be loaded before generating frames")
        started = time.perf_counter()
        if self._config.wm_engine == "fake":
            frames = self._generate_fake_chunk()
        else:
            if self._runtime is None:
                raise YumeEngineError("runtime is not initialized")
            frames = await asyncio.to_thread(
                self._runtime.generate_chunk,
                self._prompt,
                self._config.yume_chunk_frames,
            )
        chunk_ms = (time.perf_counter() - started) * 1000
        return ChunkResult(
            frames=frames,
            chunk_ms=chunk_ms,
            inference_ms=chunk_ms / max(len(frames), 1),
        )

    async def end_session(self) -> None:
        self._frame_idx = 0
        if self._runtime is not None:
            self._runtime.reset_session_state()

    def _generate_fake_chunk(self) -> list[np.ndarray]:
        height = int(self._config.frame_height)
        width = int(self._config.frame_width)
        chunk_frames = max(1, int(self._config.yume_chunk_frames))
        digest = hashlib.sha256(self._prompt.encode("utf-8")).digest()
        base = np.frombuffer(digest[:3], dtype=np.uint8).astype(np.uint16)
        accent = np.frombuffer(digest[3:6], dtype=np.uint8).astype(np.uint16)
        x = np.arange(width, dtype=np.uint16)[None, :]
        y = np.arange(height, dtype=np.uint16)[:, None]
        frames: list[np.ndarray] = []

        for offset in range(chunk_frames):
            phase = np.uint16((self._frame_idx + offset) % 256)
            red = (x + base[0] + phase) % 256
            green = (y + base[1] + phase * 3) % 256
            blue = ((x // 2) + (y // 2) + base[2] + phase * 5) % 256
            frame = np.empty((height, width, 3), dtype=np.uint8)
            frame[..., 0] = red.astype(np.uint8)
            frame[..., 1] = green.astype(np.uint8)
            frame[..., 2] = blue.astype(np.uint8)

            stripe_height = max(8, height // 10)
            frame[:stripe_height, :, 0] = np.uint8((accent[0] + phase * 7) % 256)
            frame[:stripe_height, :, 1] = np.uint8((accent[1] + len(self._prompt)) % 256)
            frame[:stripe_height, :, 2] = np.uint8((accent[2] + phase * 11) % 256)
            frames.append(frame)

        self._frame_idx += chunk_frames
        return frames


def _normalize_prompt(prompt: str | None, default_prompt: str) -> str:
    normalized = (prompt or "").strip()
    if normalized:
        return normalized
    return default_prompt.strip() or default_prompt
