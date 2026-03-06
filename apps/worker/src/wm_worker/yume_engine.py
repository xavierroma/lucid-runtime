from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass

import numpy as np

from wm_worker.config import RuntimeConfig
from wm_worker.models import ActionPayload, ActionSnapshot
from wm_worker.yume_prompt import compose_yume_prompt


class YumeEngineError(RuntimeError):
    pass


@dataclass(slots=True)
class ChunkResult:
    frames: list[np.ndarray]
    inference_ms: float


class YumeEngine:
    def __init__(self, config: RuntimeConfig, logger: logging.Logger) -> None:
        self._config = config
        self._logger = logger
        self._frame_idx = 0
        self._prompt = config.yume_base_prompt
        self._effective_prompt: str | None = None
        self._latest_snapshot = ActionSnapshot(
            prompt=config.yume_base_prompt,
            action=ActionPayload(),
            last_seq=0,
            updated_at_ms=0,
        )
        self._device = "cpu"
        self._loaded = False
        self._runtime = None

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
            import torch  # type: ignore
        except Exception as exc:
            raise YumeEngineError(
                "WM_ENGINE=yume requires torch; install wm-worker[yume]"
            ) from exc
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

        try:
            from wm_worker.yume_single_gpu_runtime import (
                YumeSingleGpuRuntime,
                YumeSingleGpuRuntimeError,
            )
        except Exception as exc:  # pragma: no cover - dependency/runtime boundary
            raise YumeEngineError("failed importing Yume runtime adapter") from exc

        self._device = "cuda"
        torch.backends.cuda.matmul.allow_tf32 = True
        runtime = YumeSingleGpuRuntime(
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
        self._prompt = prompt or self._config.yume_base_prompt
        self._effective_prompt = None

    async def update_snapshot(self, snapshot: ActionSnapshot) -> None:
        self._latest_snapshot = snapshot
        if snapshot.prompt:
            self._prompt = snapshot.prompt

    async def generate_chunk(self) -> ChunkResult:
        if not self._loaded:
            raise YumeEngineError("engine must be loaded before generating frames")
        started = time.perf_counter()
        effective_prompt = self._build_effective_prompt()
        if self._config.wm_engine == "fake":
            frames = self._generate_fake_chunk(effective_prompt)
        else:
            if self._runtime is None:
                raise YumeEngineError("runtime is not initialized")
            frames = self._runtime.generate_chunk(
                effective_prompt,
                self._config.yume_chunk_frames,
            )
        inference_ms = (time.perf_counter() - started) * 1000
        return ChunkResult(frames=frames, inference_ms=inference_ms / max(len(frames), 1))

    async def end_session(self) -> None:
        self._frame_idx = 0

    def _generate_fake_chunk(self, prompt: str) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        for _ in range(self._config.yume_chunk_frames):
            frames.append(self._make_frame(seed=f"{prompt}:{self._frame_idx}"))
            self._frame_idx += 1
        return frames

    def _build_effective_prompt(self) -> str:
        effective_prompt = compose_yume_prompt(
            self._prompt,
            self._latest_snapshot.action,
            default_scene_prompt=self._config.yume_base_prompt,
        )
        if effective_prompt != self._effective_prompt:
            self._effective_prompt = effective_prompt
            self._logger.info(
                "effective yume prompt updated prompt=%s",
                self._truncate_prompt(effective_prompt),
            )
        return effective_prompt

    @staticmethod
    def _truncate_prompt(prompt: str, limit: int = 240) -> str:
        if len(prompt) <= limit:
            return prompt
        return prompt[: limit - 3] + "..."

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
