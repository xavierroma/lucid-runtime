from __future__ import annotations

import asyncio
import atexit
import gc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .config import WaypointRuntimeConfig

@dataclass(frozen=True, slots=True)
class WaypointControlState:
    buttons: frozenset[int] = frozenset()
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    scroll_amount: int = 0


class WaypointEngine:
    def __init__(self, runtime_config: WaypointRuntimeConfig, logger) -> None:
        self._config = runtime_config
        self._logger = logger
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="waypoint-cuda")
        atexit.register(_shutdown_executor, self._executor)
        self._engine: Any | None = None
        self._ctrl_cls: Any | None = None
        self._seed_frame: np.ndarray | None = None
        self._last_frame: np.ndarray | None = None
        self._current_prompt = runtime_config.waypoint_default_prompt

    async def load(self, *, warmup: bool = True) -> None:
        if self._engine is not None:
            return
        start = perf_counter()
        try:
            await self._run_on_cuda_thread(self._load_engine_sync)
            self._logger.info(
                "waypoint.engine.load engine_ready elapsed_ms=%.1f model_source=%s frame_width=%s frame_height=%s warmup=%s",
                (perf_counter() - start) * 1000.0,
                self._config.waypoint_model_source,
                self._config.frame_width,
                self._config.frame_height,
                warmup,
            )
            self._seed_frame = self._load_seed_frame()
            self._logger.info(
                "waypoint.engine.load seed_frame_ready elapsed_ms=%.1f model_source=%s",
                (perf_counter() - start) * 1000.0,
                self._config.waypoint_model_source,
            )
            if warmup:
                await self._warmup()
                self._logger.info(
                    "waypoint.engine.load warmup_complete elapsed_ms=%.1f model_source=%s",
                    (perf_counter() - start) * 1000.0,
                    self._config.waypoint_model_source,
                )
            else:
                self._logger.info(
                    "waypoint.engine.load warmup_skipped elapsed_ms=%.1f model_source=%s reason=compiler_cache_hit",
                    (perf_counter() - start) * 1000.0,
                    self._config.waypoint_model_source,
                )
        except Exception as exc:
            self._logger.error(
                "waypoint.engine.load failed duration_ms=%.1f model_source=%s error_type=%s",
                (perf_counter() - start) * 1000.0,
                self._config.waypoint_model_source,
                exc.__class__.__name__,
            )
            raise
        self._logger.info(
            "waypoint.engine.load complete duration_ms=%.1f model_source=%s",
            (perf_counter() - start) * 1000.0,
            self._config.waypoint_model_source,
        )

    async def start_session(self, prompt: str) -> None:
        start = perf_counter()
        try:
            await self._run_on_cuda_thread(lambda: self._reset_session_sync(prompt))
        except Exception as exc:
            self._logger.error(
                "waypoint.engine.start_session failed duration_ms=%.1f prompt_chars=%s error_type=%s",
                (perf_counter() - start) * 1000.0,
                len(prompt),
                exc.__class__.__name__,
            )
            raise
        self._logger.info(
            "waypoint.engine.start_session complete duration_ms=%.1f prompt_chars=%s",
            (perf_counter() - start) * 1000.0,
            len(prompt),
        )

    async def update_prompt(self, prompt: str) -> None:
        if prompt == self._current_prompt:
            return
        start = perf_counter()
        try:
            await self._run_on_cuda_thread(lambda: self._set_prompt_sync(prompt))
        except Exception as exc:
            self._logger.error(
                "waypoint.engine.update_prompt failed duration_ms=%.1f prompt_chars=%s error_type=%s",
                (perf_counter() - start) * 1000.0,
                len(prompt),
                exc.__class__.__name__,
            )
            raise
        self._logger.info(
            "waypoint.engine.update_prompt complete duration_ms=%.1f prompt_chars=%s",
            (perf_counter() - start) * 1000.0,
            len(prompt),
        )

    async def generate_frame(
        self,
        controls: WaypointControlState,
    ) -> tuple[np.ndarray, float]:
        loop = asyncio.get_running_loop()
        start_s = loop.time()
        frame = await self._run_on_cuda_thread(lambda: self._generate_frame_sync(controls))
        inference_ms = (loop.time() - start_s) * 1000.0
        return frame, inference_ms

    async def end_session(self) -> None:
        if self._engine is None:
            return
        await self._run_on_cuda_thread(self._reset_engine_state_sync)

    async def _warmup(self) -> None:
        await self._run_on_cuda_thread(lambda: self._warmup_sync(self._current_prompt))

    async def _run_on_cuda_thread(self, fn):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn)

    def _load_engine_sync(self) -> None:
        import torch
        from world_engine import CtrlInput, WorldEngine

        start = perf_counter()
        _patch_world_engine_prompt_encoder()
        self._logger.info(
            "waypoint.engine.load_sync prompt_encoder_patch_ready elapsed_ms=%.1f model_source=%s",
            (perf_counter() - start) * 1000.0,
            self._config.waypoint_model_source,
        )
        model_source = self._config.waypoint_model_source
        self._logger.info("loading waypoint model source=%s", model_source)
        last_error: BaseException | None = None
        for dtype in (torch.bfloat16, torch.float16):
            try:
                self._logger.info(
                    "waypoint.engine.load_sync world_engine_ctor_start elapsed_ms=%.1f model_source=%s dtype=%s",
                    (perf_counter() - start) * 1000.0,
                    model_source,
                    dtype,
                )
                self._engine = WorldEngine(
                    model_source,
                    model_config_overrides={
                        "ae_uri": self._config.waypoint_ae_source,
                        "prompt_encoder_uri": self._config.waypoint_prompt_encoder_source,
                    },
                    device="cuda",
                    dtype=dtype,
                )
                self._ctrl_cls = CtrlInput
                self._logger.info("loaded waypoint model source=%s dtype=%s", model_source, dtype)
                self._logger.info(
                    "waypoint.engine.load_sync world_engine_ctor_complete elapsed_ms=%.1f model_source=%s dtype=%s",
                    (perf_counter() - start) * 1000.0,
                    model_source,
                    dtype,
                )
                self._logger.info(
                    "waypoint.engine.load_sync complete duration_ms=%.1f model_source=%s dtype=%s",
                    (perf_counter() - start) * 1000.0,
                    model_source,
                    dtype,
                )
                return
            except torch.OutOfMemoryError as exc:
                last_error = exc
                self._logger.warning(
                    "OOM while loading waypoint model source=%s dtype=%s; retrying",
                    model_source,
                    dtype,
                )
                self._logger.warning(
                    "waypoint.engine.load_sync world_engine_ctor_oom elapsed_ms=%.1f model_source=%s dtype=%s",
                    (perf_counter() - start) * 1000.0,
                    model_source,
                    dtype,
                )
                self._cleanup_cuda_sync()
            except Exception as exc:
                last_error = exc
                self._logger.error(
                    "waypoint.engine.load_sync world_engine_ctor_failed elapsed_ms=%.1f model_source=%s dtype=%s error_type=%s",
                    (perf_counter() - start) * 1000.0,
                    model_source,
                    dtype,
                    exc.__class__.__name__,
                )
                self._cleanup_cuda_sync()
                break
        if last_error is not None:
            self._logger.error(
                "waypoint.engine.load_sync failed duration_ms=%.1f model_source=%s error_type=%s",
                (perf_counter() - start) * 1000.0,
                model_source,
                last_error.__class__.__name__,
            )
        raise RuntimeError(f"failed to load waypoint model from {model_source}") from last_error

    def _reset_session_sync(self, prompt: str, seed_frame: np.ndarray | None = None) -> None:
        engine = self._require_engine()
        seed_frame = (
            np.ascontiguousarray(seed_frame)
            if seed_frame is not None
            else self._require_seed_frame()
        )
        import torch

        engine.reset()
        seed_tensor = torch.from_numpy(seed_frame).to(device="cuda", dtype=torch.uint8)
        engine.append_frame(seed_tensor)
        self._last_frame = seed_frame
        self._current_prompt = prompt
        if getattr(engine.model_cfg, "prompt_conditioning", None) is not None:
            engine.set_prompt(prompt)

    def _set_prompt_sync(self, prompt: str) -> None:
        engine = self._require_engine()
        if getattr(engine.model_cfg, "prompt_conditioning", None) is None:
            self._current_prompt = prompt
            return
        engine.set_prompt(prompt)
        self._current_prompt = prompt

    def _generate_frame_sync(self, controls: WaypointControlState) -> np.ndarray:
        import torch

        engine = self._require_engine()
        self._roll_session_if_needed_sync()
        ctrl = self._build_ctrl_input(controls)
        frame = engine.gen_frame(ctrl=ctrl)
        if frame.dtype != torch.uint8:
            frame = frame.clamp(0, 255).to(dtype=torch.uint8)
        if not frame.is_contiguous():
            frame = frame.contiguous()

        expected_shape = (
            int(self._config.frame_height),
            int(self._config.frame_width),
            3,
        )
        if tuple(frame.shape) != expected_shape:
            raise RuntimeError(
                f"waypoint frame shape mismatch: expected {expected_shape}, got {tuple(frame.shape)}"
            )
        frame_cpu = frame if frame.device.type == "cpu" else frame.to(device="cpu")
        frame_np = frame_cpu.numpy()
        self._last_frame = frame_np
        return frame_np

    def _warmup_sync(self, prompt: str) -> None:
        start = perf_counter()
        self._reset_session_sync(prompt)
        self._logger.info(
            "waypoint.engine.warmup session_reset elapsed_ms=%.1f prompt_chars=%s",
            (perf_counter() - start) * 1000.0,
            len(prompt),
        )
        self._generate_frame_sync(WaypointControlState())
        self._logger.info(
            "waypoint.engine.warmup first_frame_generated elapsed_ms=%.1f prompt_chars=%s",
            (perf_counter() - start) * 1000.0,
            len(prompt),
        )
        self._logger.info("waypoint warmup complete")
        self._logger.info(
            "waypoint.engine.warmup complete duration_ms=%.1f prompt_chars=%s",
            (perf_counter() - start) * 1000.0,
            len(prompt),
        )

    def _reset_engine_state_sync(self) -> None:
        engine = self._require_engine()
        engine.reset()

    def _roll_session_if_needed_sync(self) -> None:
        frame_limit = self._frame_history_limit_sync()
        if frame_limit is None:
            return

        frame_timestamp = self._current_frame_timestamp_sync()
        if frame_timestamp is None or frame_timestamp < frame_limit:
            return

        rollover_seed = self._last_frame if self._last_frame is not None else self._require_seed_frame()
        self._logger.warning(
            "waypoint.engine.generate_frame frame_history_limit_reached frame_timestamp=%s frame_limit=%s",
            frame_timestamp,
            frame_limit,
        )
        self._reset_session_sync(self._current_prompt, seed_frame=rollover_seed)

    def _frame_history_limit_sync(self) -> int | None:
        engine = self._require_engine()
        model_cfg = getattr(engine, "model_cfg", None)
        raw_limit = getattr(model_cfg, "n_frames", None)
        if raw_limit is None:
            return None
        try:
            frame_limit = int(raw_limit)
        except (TypeError, ValueError):
            return None
        return frame_limit if frame_limit > 0 else None

    def _current_frame_timestamp_sync(self) -> int | None:
        engine = self._require_engine()
        frame_ts = getattr(engine, "frame_ts", None)
        if frame_ts is None:
            return None
        try:
            return int(frame_ts.reshape(-1)[0].item())
        except Exception:
            try:
                return int(frame_ts)
            except Exception:
                return None

    def _load_seed_frame(self) -> np.ndarray:
        start = perf_counter()
        height = int(self._config.frame_height)
        width = int(self._config.frame_width)
        seed_path = self._config.waypoint_seed_image
        if seed_path is not None:
            frame = self._load_seed_image_from_path(seed_path, width=width, height=height)
            self._logger.info(
                "waypoint.engine.load_seed_frame complete duration_ms=%.1f source=file seed_path=%s",
                (perf_counter() - start) * 1000.0,
                seed_path,
            )
            return frame
        frame = _default_seed_frame(width=width, height=height)
        self._logger.info(
            "waypoint.engine.load_seed_frame complete duration_ms=%.1f source=generated_default",
            (perf_counter() - start) * 1000.0,
        )
        return frame

    @staticmethod
    def _load_seed_image_from_path(path: Path, *, width: int, height: int) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"waypoint seed image not found: {path}")
        from PIL import Image

        image = Image.open(path).convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.BILINEAR)
        return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))

    def _build_ctrl_input(self, controls: WaypointControlState):
        ctrl_cls = self._ctrl_cls
        if ctrl_cls is None:
            raise RuntimeError("waypoint control class is not loaded")
        return ctrl_cls(
            button=set(controls.buttons),
            mouse=(float(controls.mouse_dx), float(controls.mouse_dy)),
            scroll_wheel=int(controls.scroll_amount),
        )

    def _require_engine(self):
        if self._engine is None:
            raise RuntimeError("waypoint engine has not been loaded")
        return self._engine

    def _require_seed_frame(self) -> np.ndarray:
        if self._seed_frame is None:
            raise RuntimeError("waypoint seed frame has not been initialized")
        return self._seed_frame

    @staticmethod
    def _cleanup_cuda_sync() -> None:
        try:
            import torch
        except Exception:
            return

        gc.collect()
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _default_seed_frame(*, width: int, height: int) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    horizon = max(height // 2, 1)

    for row in range(horizon):
        t = row / max(horizon - 1, 1)
        frame[row, :, 0] = np.uint8(30 + 45 * t)
        frame[row, :, 1] = np.uint8(80 + 70 * t)
        frame[row, :, 2] = np.uint8(135 + 90 * t)

    for row in range(horizon, height):
        t = (row - horizon) / max(height - horizon - 1, 1)
        frame[row, :, 0] = np.uint8(18 + 24 * t)
        frame[row, :, 1] = np.uint8(82 + 36 * t)
        frame[row, :, 2] = np.uint8(22 + 18 * t)

    path_half_width = max(width // 14, 8)
    for row in range(horizon, height):
        spread = int((row - horizon) * (width / max(height, 1)) * 0.18)
        center = width // 2
        left = max(center - path_half_width - spread, 0)
        right = min(center + path_half_width + spread, width)
        frame[row, left:right, :] = np.array([110, 95, 78], dtype=np.uint8)

    return frame


def _shutdown_executor(executor: ThreadPoolExecutor) -> None:
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


def _patch_world_engine_prompt_encoder() -> None:
    import os

    import torch
    from torch import nn
    from transformers import AutoTokenizer, UMT5EncoderModel
    try:
        from world_engine.model import PromptEncoder as prompt_encoder_cls
    except Exception:
        from world_engine.model.world_model import PromptEncoder as prompt_encoder_cls

    if getattr(prompt_encoder_cls, "_lucid_prompt_encoder_patched", False):
        return

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    def _patched_init(self, model_id="google/umt5-xl", dtype=torch.bfloat16):
        nn.Module.__init__(self)
        self.dtype = dtype
        try:
            self.tok = AutoTokenizer.from_pretrained(
                model_id,
                fix_mistral_regex=False,
            )
        except TypeError:
            self.tok = AutoTokenizer.from_pretrained(model_id)
        try:
            self.encoder = UMT5EncoderModel.from_pretrained(
                model_id,
                dtype=dtype,
            ).eval()
        except TypeError:
            self.encoder = UMT5EncoderModel.from_pretrained(
                model_id,
                torch_dtype=dtype,
            ).eval()

    prompt_encoder_cls.__init__ = _patched_init
    prompt_encoder_cls._lucid_prompt_encoder_patched = True
