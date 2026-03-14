from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import (
    HELIOS_VIDEO_HEIGHT,
    HELIOS_VIDEO_WIDTH,
    HeliosRuntimeConfig,
)


class HeliosEngineError(RuntimeError):
    pass


@dataclass(slots=True)
class ChunkResult:
    frames: list[np.ndarray]
    chunk_ms: float
    inference_ms: float


class HeliosEngine:
    def __init__(self, config: HeliosRuntimeConfig, logger: logging.Logger) -> None:
        self._config = config
        self._logger = logger
        self._loaded = False
        self._prompt = _normalize_prompt(
            config.helios_default_prompt,
            config.helios_default_prompt,
        )
        self._pipeline: Any | None = None
        self._generator: Any | None = None
        self._last_chunk: np.ndarray | None = None
        self._chunk_index = 0

    async def load(self) -> None:
        if self._loaded:
            self._logger.info("reusing preloaded helios engine")
            return

        if self._config.backend == "fake":
            self._loaded = True
            self._logger.info("starting helios in fake engine mode")
            return

        started = time.perf_counter()
        try:
            await asyncio.to_thread(self._load_pipeline_sync)
        except Exception as exc:
            raise HeliosEngineError(str(exc)) from exc
        self._loaded = True
        self._logger.info(
            "helios engine loaded duration_ms=%.1f model_source=%s group_offload=%s offload_type=%s",
            (time.perf_counter() - started) * 1000.0,
            self._config.helios_model_source,
            self._config.helios_enable_group_offloading,
            self._config.helios_group_offloading_type,
        )

    async def start_session(self, prompt: str | None) -> None:
        if not self._loaded:
            raise HeliosEngineError("engine must be loaded before starting session")
        self._prompt = _normalize_prompt(prompt, self._config.helios_default_prompt)
        self._last_chunk = None
        self._chunk_index = 0
        self._generator = None
        if self._config.backend == "real":
            import torch

            self._generator = torch.Generator(device="cuda").manual_seed(secrets.randbits(63))

    async def update_prompt(self, prompt: str) -> None:
        next_prompt = _normalize_prompt(prompt, self._config.helios_default_prompt)
        if next_prompt == self._prompt:
            return
        self._prompt = next_prompt
        self._logger.info("updated helios prompt prompt_len=%s", len(self._prompt))

    async def generate_chunk(self) -> ChunkResult:
        if not self._loaded:
            raise HeliosEngineError("engine must be loaded before generating chunks")

        started = time.perf_counter()
        if self._config.backend == "fake":
            frames = self._generate_fake_chunk()
        else:
            frames = await asyncio.to_thread(self._generate_real_chunk_sync)

        chunk_ms = (time.perf_counter() - started) * 1000.0
        return ChunkResult(
            frames=frames,
            chunk_ms=chunk_ms,
            inference_ms=chunk_ms / max(len(frames), 1),
        )

    async def end_session(self) -> None:
        self._last_chunk = None
        self._chunk_index = 0
        self._generator = None

    def _load_pipeline_sync(self) -> None:
        import torch
        from diffusers import AutoencoderKLWan, HeliosPyramidPipeline

        if not torch.cuda.is_available():
            raise HeliosEngineError("CUDA is required for Helios backend=real")

        torch.backends.cuda.matmul.allow_tf32 = True

        model_source = self._config.helios_model_source
        vae = AutoencoderKLWan.from_pretrained(
            model_source,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        pipeline = HeliosPyramidPipeline.from_pretrained(
            model_source,
            vae=vae,
            torch_dtype=torch.bfloat16,
        )
        if hasattr(pipeline, "set_progress_bar_config"):
            pipeline.set_progress_bar_config(disable=True)

        if self._config.helios_enable_group_offloading:
            pipeline.enable_group_offload(
                onload_device=torch.device("cuda"),
                offload_device=torch.device("cpu"),
                offload_type=self._config.helios_group_offloading_type,
                num_blocks_per_group=(
                    1 if self._config.helios_group_offloading_type == "block_level" else None
                ),
            )
        else:
            pipeline.to("cuda")

        self._pipeline = pipeline

    def _generate_real_chunk_sync(self) -> list[np.ndarray]:
        pipeline = self._require_pipeline()
        call_kwargs: dict[str, Any] = {
            "prompt": self._prompt,
            "negative_prompt": self._config.helios_negative_prompt,
            "height": HELIOS_VIDEO_HEIGHT,
            "width": HELIOS_VIDEO_WIDTH,
            "num_frames": int(self._config.helios_chunk_frames),
            "guidance_scale": float(self._config.helios_guidance_scale),
            "pyramid_num_inference_steps_list": list(self._config.helios_pyramid_steps),
            "is_amplify_first_chunk": (
                self._config.helios_amplify_first_chunk and self._chunk_index == 0
            ),
            "max_sequence_length": int(self._config.helios_max_sequence_length),
            "output_type": "np",
            "generator": self._generator,
        }
        if self._last_chunk is not None:
            call_kwargs["video"] = np.array(self._last_chunk, copy=True)

        output = pipeline(**call_kwargs)
        frames = _normalize_pipeline_frames(
            getattr(output, "frames", output),
            expected_height=HELIOS_VIDEO_HEIGHT,
            expected_width=HELIOS_VIDEO_WIDTH,
        )
        self._last_chunk = np.stack(frames, axis=0)
        self._chunk_index += 1
        return frames

    def _generate_fake_chunk(self) -> list[np.ndarray]:
        height = HELIOS_VIDEO_HEIGHT
        width = HELIOS_VIDEO_WIDTH
        chunk_frames = max(1, int(self._config.helios_chunk_frames))
        digest = hashlib.sha256(self._prompt.encode("utf-8")).digest()
        base = np.frombuffer(digest[:3], dtype=np.uint8).astype(np.uint16)
        accent = np.frombuffer(digest[3:6], dtype=np.uint8).astype(np.uint16)
        x = np.arange(width, dtype=np.uint16)[None, :]
        y = np.arange(height, dtype=np.uint16)[:, None]
        frames: list[np.ndarray] = []

        for offset in range(chunk_frames):
            phase = np.uint16((self._chunk_index + offset) % 256)
            red = (x + base[0] + phase * 3) % 256
            green = (y + base[1] + phase * 5) % 256
            blue = ((x // 2) + (y // 2) + base[2] + phase * 7) % 256
            frame = np.empty((height, width, 3), dtype=np.uint8)
            frame[..., 0] = red.astype(np.uint8)
            frame[..., 1] = green.astype(np.uint8)
            frame[..., 2] = blue.astype(np.uint8)
            stripe_height = max(8, height // 12)
            frame[:stripe_height, :, 0] = np.uint8((accent[0] + phase * 11) % 256)
            frame[:stripe_height, :, 1] = np.uint8((accent[1] + len(self._prompt)) % 256)
            frame[:stripe_height, :, 2] = np.uint8((accent[2] + phase * 13) % 256)
            frames.append(frame)

        self._last_chunk = np.stack(frames, axis=0)
        self._chunk_index += 1
        return frames

    def _require_pipeline(self):
        if self._pipeline is None:
            raise HeliosEngineError("pipeline is not initialized")
        return self._pipeline


def _normalize_prompt(prompt: str | None, default_prompt: str) -> str:
    normalized = (prompt or "").strip()
    if normalized:
        return normalized
    return default_prompt.strip() or default_prompt


def _normalize_pipeline_frames(
    raw_frames: Any,
    *,
    expected_height: int,
    expected_width: int,
) -> list[np.ndarray]:
    if isinstance(raw_frames, tuple):
        if len(raw_frames) != 1:
            raise HeliosEngineError(f"unsupported Helios pipeline tuple output length: {len(raw_frames)}")
        raw_frames = raw_frames[0]

    if hasattr(raw_frames, "detach"):
        raw_frames = raw_frames.detach().cpu().numpy()

    if isinstance(raw_frames, np.ndarray):
        frames = raw_frames
        if frames.ndim == 5:
            frames = frames[0]
        if frames.ndim != 4:
            raise HeliosEngineError(f"expected 4D/5D frame output, got {frames.ndim}D")
        if frames.shape[-1] != 3 and frames.shape[1] == 3:
            frames = np.transpose(frames, (0, 2, 3, 1))
        return [
            _normalize_frame(frame, expected_height=expected_height, expected_width=expected_width)
            for frame in frames
        ]

    if isinstance(raw_frames, list):
        frames = raw_frames
        if frames and isinstance(frames[0], list):
            frames = frames[0]
        return [
            _normalize_frame(
                np.asarray(frame),
                expected_height=expected_height,
                expected_width=expected_width,
            )
            for frame in frames
        ]

    raise HeliosEngineError(f"unsupported Helios pipeline output type: {type(raw_frames).__name__}")


def _normalize_frame(
    frame: np.ndarray,
    *,
    expected_height: int,
    expected_width: int,
) -> np.ndarray:
    normalized = np.asarray(frame)
    if normalized.ndim == 3 and normalized.shape[0] == 3 and normalized.shape[-1] != 3:
        normalized = np.transpose(normalized, (1, 2, 0))
    if normalized.shape != (expected_height, expected_width, 3):
        raise HeliosEngineError(
            "helios frame shape mismatch: expected "
            f"{(expected_height, expected_width, 3)}, got {tuple(normalized.shape)}"
        )
    if normalized.dtype != np.uint8:
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(normalized)
