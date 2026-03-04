from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from wm_worker.config import WorkerConfig
from wm_worker.models import ActionPayload, ActionSnapshot


class YumeEngineError(RuntimeError):
    pass


@dataclass(slots=True)
class ChunkResult:
    frames: list[np.ndarray]
    inference_ms: float


class YumeEngine:
    def __init__(self, config: WorkerConfig, logger: logging.Logger) -> None:
        self._config = config
        self._logger = logger
        self._frame_idx = 0
        self._prompt = config.yume_base_prompt
        self._latest_snapshot = ActionSnapshot(
            prompt=config.yume_base_prompt,
            action=ActionPayload(),
            last_seq=0,
            updated_at_ms=0,
        )
        self._device = "cpu"
        self._loaded = False
        self._torch = None
        self._gpu_generator = None

    async def load(self) -> None:
        if self._config.wm_engine not in {"fake", "yume"}:
            raise YumeEngineError(
                f"unsupported WM_ENGINE={self._config.wm_engine}; expected fake or yume"
            )

        if self._config.wm_engine == "fake":
            self._logger.info("starting in fake engine mode")
            self._loaded = True
            return

        try:
            import torch  # type: ignore
        except Exception as exc:
            raise YumeEngineError(
                "WM_ENGINE=yume requires torch; install wm-worker[yume]"
            ) from exc
        self._torch = torch

        if not self._config.yume_model_dir.exists():
            raise YumeEngineError(
                f"YUME_MODEL_DIR does not exist: {self._config.yume_model_dir}"
            )

        # "yume" mode keeps the shape of a GPU runtime and enforces model files.
        required_files = [
            "diffusion_pytorch_model.safetensors",
            "config.yaml",
            "Wan2.2_VAE.pth",
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

        if not torch.cuda.is_available():
            raise YumeEngineError("CUDA is required for WM_ENGINE=yume")
        self._device = "cuda"
        torch.backends.cuda.matmul.allow_tf32 = True
        self._gpu_generator = torch.Generator(device=self._device)
        self._loaded = True
        self._logger.info("yume engine initialized in single-GPU mode on %s", self._device)

    async def start_session(self, prompt: str | None) -> None:
        if not self._loaded:
            raise YumeEngineError("engine must be loaded before starting session")
        self._frame_idx = 0
        self._prompt = prompt or self._config.yume_base_prompt

    async def update_snapshot(self, snapshot: ActionSnapshot) -> None:
        self._latest_snapshot = snapshot
        if snapshot.prompt:
            self._prompt = snapshot.prompt

    async def generate_chunk(self) -> ChunkResult:
        if not self._loaded:
            raise YumeEngineError("engine must be loaded before generating frames")
        started = time.perf_counter()
        if self._config.wm_engine == "fake":
            frames = self._generate_fake_chunk()
        else:
            frames = await self._generate_gpu_chunk()
        inference_ms = (time.perf_counter() - started) * 1000
        return ChunkResult(frames=frames, inference_ms=inference_ms / max(len(frames), 1))

    async def end_session(self) -> None:
        self._frame_idx = 0

    def _generate_fake_chunk(self) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        for _ in range(self._config.yume_chunk_frames):
            frames.append(self._make_frame(seed=f"{self._prompt}:{self._frame_idx}"))
            self._frame_idx += 1
        return frames

    async def _generate_gpu_chunk(self) -> list[np.ndarray]:
        # Lightweight GPU-backed placeholder path for Yume integration scaffolding.
        # It keeps single-GPU BF16/TF32 runtime behavior and produces streamable frames.
        if self._torch is None:
            raise YumeEngineError("torch is not initialized")
        torch = self._torch

        frames: list[np.ndarray] = []
        width = self._config.frame_width
        height = self._config.frame_height
        for _ in range(self._config.yume_chunk_frames):
            rand = torch.randint(
                low=0,
                high=256,
                size=(height, width, 3),
                dtype=torch.uint8,
                device=self._device,
                generator=self._gpu_generator,
            )
            frame = rand.cpu().numpy()
            frame = self._apply_action_overlay(frame)
            frames.append(frame)
            self._frame_idx += 1
            await asyncio.sleep(0)
        return frames

    def _make_frame(self, seed: str) -> np.ndarray:
        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        base_r = digest[0]
        base_g = digest[1]
        base_b = digest[2]
        width = self._config.frame_width
        height = self._config.frame_height
        x = np.linspace(0, 1, width, dtype=np.float32)
        y = np.linspace(0, 1, height, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., 0] = np.clip(base_r * xx, 0, 255).astype(np.uint8)
        frame[..., 1] = np.clip(base_g * yy, 0, 255).astype(np.uint8)
        frame[..., 2] = np.clip(base_b * (1 - xx), 0, 255).astype(np.uint8)
        return self._apply_action_overlay(frame)

    def _apply_action_overlay(self, frame: np.ndarray) -> np.ndarray:
        snapshot = self._latest_snapshot
        action = snapshot.action
        shift_x = int(action.mouse_dx) % frame.shape[1]
        shift_y = int(action.mouse_dy) % frame.shape[0]
        shifted = np.roll(frame, shift=(shift_y, shift_x), axis=(0, 1))
        if action.keys:
            marker_size = max(8, min(frame.shape[0], frame.shape[1]) // 20)
            shifted[:marker_size, : marker_size * len(action.keys), 0] = 255
        return shifted
