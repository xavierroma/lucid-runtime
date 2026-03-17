from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Annotated

from pydantic import Field

from lucid import (
    InputFile,
    LoadContext,
    LucidModel,
    LucidSession,
    SessionContext,
    hold,
    image_input,
    input,
    pointer,
    publish,
    wheel,
)

from .config import (
    WAYPOINT_FRAME_HEIGHT,
    WAYPOINT_FRAME_WIDTH,
    WAYPOINT_OUTPUT_FPS,
    WaypointRuntimeConfig,
)
from .engine import WaypointControlState, WaypointEngine

_VK_LBUTTON = 0x01
_VK_RBUTTON = 0x02
_VK_SPACE = 0x20
_VK_A = 0x41
_VK_D = 0x44
_VK_S = 0x53
_VK_W = 0x57
_VK_LSHIFT = 0xA0
_VK_LCONTROL = 0xA2


@dataclass(slots=True)
class _TransientWaypointInput:
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    scroll_amount: int = 0

    def drain(self) -> tuple[float, float, int]:
        drained = (self.mouse_dx, self.mouse_dy, self.scroll_amount)
        self.mouse_dx = 0.0
        self.mouse_dy = 0.0
        self.scroll_amount = 0
        return drained


class WaypointSession(LucidSession["WaypointLucidModel"]):
    def __init__(self, model: "WaypointLucidModel", ctx: SessionContext) -> None:
        super().__init__(model, ctx)
        self.prompt = model.config.waypoint_default_prompt
        self._buttons: set[int] = set()
        self._transient = _TransientWaypointInput()
        self._pending_initial_frame: InputFile | None = None

    @input(description="Update the text prompt used by Waypoint.", paused=True)
    def set_prompt(
        self,
        prompt: Annotated[str, Field(..., min_length=1)],
    ) -> None:
        self.prompt = prompt.strip() or self.model.config.waypoint_default_prompt

    @input(description="Set the initial frame used to seed Waypoint.", paused=True)
    def set_initial_frame(
        self,
        image: InputFile = image_input(size=(WAYPOINT_FRAME_WIDTH, WAYPOINT_FRAME_HEIGHT)),
    ) -> None:
        self._pending_initial_frame = image

    @input(
        description="Move forward.",
        binding=hold(keys=("KeyW", "ArrowUp")),
    )
    def forward(self, pressed: bool) -> None:
        self._set_button(_VK_W, pressed)

    @input(
        description="Move backward.",
        binding=hold(keys=("KeyS", "ArrowDown")),
    )
    def backward(self, pressed: bool) -> None:
        self._set_button(_VK_S, pressed)

    @input(
        description="Strafe left.",
        binding=hold(keys=("KeyA", "ArrowLeft")),
    )
    def left(self, pressed: bool) -> None:
        self._set_button(_VK_A, pressed)

    @input(
        description="Strafe right.",
        binding=hold(keys=("KeyD", "ArrowRight")),
    )
    def right(self, pressed: bool) -> None:
        self._set_button(_VK_D, pressed)

    @input(
        description="Jump.",
        binding=hold(keys=("Space",)),
    )
    def jump(self, pressed: bool) -> None:
        self._set_button(_VK_SPACE, pressed)

    @input(
        description="Sprint.",
        binding=hold(keys=("ShiftLeft", "ShiftRight")),
    )
    def sprint(self, pressed: bool) -> None:
        self._set_button(_VK_LSHIFT, pressed)

    @input(
        description="Crouch.",
        binding=hold(keys=("ControlLeft", "ControlRight")),
    )
    def crouch(self, pressed: bool) -> None:
        self._set_button(_VK_LCONTROL, pressed)

    @input(
        description="Primary fire.",
        binding=hold(keys=("KeyJ",), mouse_buttons=(0,)),
    )
    def primary_fire(self, pressed: bool) -> None:
        self._set_button(_VK_LBUTTON, pressed)

    @input(
        description="Secondary fire.",
        binding=hold(keys=("KeyK",), mouse_buttons=(2,)),
    )
    def secondary_fire(self, pressed: bool) -> None:
        self._set_button(_VK_RBUTTON, pressed)

    @input(
        description="Look around.",
        binding=pointer(),
    )
    def look(self, dx: float, dy: float) -> None:
        self._transient.mouse_dx += float(dx)
        self._transient.mouse_dy += float(dy)

    @input(
        description="Scroll the wheel.",
        binding=wheel(),
    )
    def scroll(self, delta: float) -> None:
        self._transient.scroll_amount += int(delta)

    async def run(self) -> None:
        engine = self.model.require_engine()
        start = perf_counter()
        initial_prompt = self.prompt.strip() or self.model.config.waypoint_default_prompt
        try:
            await engine.start_session(
                initial_prompt,
                initial_frame_path=self._take_pending_initial_frame_path(),
            )
        except Exception as exc:
            self.ctx.logger.error(
                "waypoint.session.run failed duration_ms=%.1f session_id=%s error_type=%s",
                (perf_counter() - start) * 1000.0,
                self.ctx.session_id,
                exc.__class__.__name__,
            )
            raise
        self.ctx.logger.info(
            "waypoint.session.run engine_session_started elapsed_ms=%.1f session_id=%s prompt_chars=%s fps=%s frame_width=%s frame_height=%s",
            (perf_counter() - start) * 1000.0,
            self.ctx.session_id,
            len(self.prompt),
            WAYPOINT_OUTPUT_FPS,
            WAYPOINT_FRAME_WIDTH,
            WAYPOINT_FRAME_HEIGHT,
        )
        last_prompt = initial_prompt
        frame_index = 0
        try:
            while self.ctx.running:
                await self.ctx.wait_if_paused()
                if not self.ctx.running:
                    break
                prompt = self.prompt.strip() or self.model.config.waypoint_default_prompt
                initial_frame_path = self._take_pending_initial_frame_path()
                if initial_frame_path is not None:
                    await engine.set_initial_frame(prompt, initial_frame_path)
                    last_prompt = prompt
                    self.ctx.logger.info(
                        "waypoint.session.run initial_frame_updated elapsed_ms=%.1f session_id=%s frame_index=%s prompt_chars=%s",
                        (perf_counter() - start) * 1000.0,
                        self.ctx.session_id,
                        frame_index,
                        len(prompt),
                    )
                elif prompt != last_prompt:
                    await engine.update_prompt(prompt)
                    last_prompt = prompt
                    self.ctx.logger.info(
                        "waypoint.session.run prompt_updated elapsed_ms=%.1f session_id=%s frame_index=%s prompt_chars=%s",
                        (perf_counter() - start) * 1000.0,
                        self.ctx.session_id,
                        frame_index,
                        len(prompt),
                    )

                mouse_dx, mouse_dy, scroll_amount = self._transient.drain()
                controls = WaypointControlState(
                    buttons=frozenset(self._buttons),
                    mouse_dx=mouse_dx,
                    mouse_dy=mouse_dy,
                    scroll_amount=scroll_amount,
                )
                frame, inference_ms = await engine.generate_frame(controls)
                await self.ctx.wait_if_paused()
                if not self.ctx.running:
                    break
                self.ctx.record_inference_ms(inference_ms)
                frame_index += 1
                if frame_index == 1:
                    self.ctx.logger.info(
                        "waypoint.session.run first_frame_ready elapsed_ms=%.1f session_id=%s inference_ms=%.1f",
                        (perf_counter() - start) * 1000.0,
                        self.ctx.session_id,
                        inference_ms,
                    )
                await self.ctx.publish("main_video", frame)
                if frame_index == 1:
                    self.ctx.logger.info(
                        "waypoint.session.run first_frame_published elapsed_ms=%.1f session_id=%s",
                        (perf_counter() - start) * 1000.0,
                        self.ctx.session_id,
                    )
                    if (
                        not self.model._compiler_cache_committed
                        and self.model.compiler_cache_commit_hook is not None
                    ):
                        self.model._compiler_cache_committed = await asyncio.to_thread(
                            self.model.compiler_cache_commit_hook,
                            self.ctx.logger,
                            "first_frame",
                        )
        except Exception as exc:
            self.ctx.logger.error(
                "waypoint.session.run failed duration_ms=%.1f session_id=%s frames_generated=%s inference_ms_p50=%.1f error_type=%s",
                (perf_counter() - start) * 1000.0,
                self.ctx.session_id,
                frame_index,
                self.ctx.inference_ms_p50(),
                exc.__class__.__name__,
            )
            raise
        self.ctx.logger.info(
            "waypoint.session.run complete duration_ms=%.1f session_id=%s frames_generated=%s inference_ms_p50=%.1f",
            (perf_counter() - start) * 1000.0,
            self.ctx.session_id,
            frame_index,
            self.ctx.inference_ms_p50(),
        )

    async def close(self) -> None:
        try:
            engine = self.model._engine
            if engine is not None:
                await engine.end_session()
        finally:
            logger = self.model.logger or self.ctx.logger
            if logger is not None and self.model.compiler_cache_commit_hook is not None:
                self.model._compiler_cache_committed = (
                    await asyncio.to_thread(
                        self.model.compiler_cache_commit_hook,
                        logger,
                        "session_end",
                    )
                ) or self.model._compiler_cache_committed

    def _set_button(self, button_id: int, pressed: bool) -> None:
        if pressed:
            self._buttons.add(button_id)
            return
        self._buttons.discard(button_id)

    def _take_pending_initial_frame_path(self) -> Path | None:
        pending = self._pending_initial_frame
        self._pending_initial_frame = None
        if pending is None:
            return None
        return pending.path


class WaypointLucidModel(LucidModel[WaypointRuntimeConfig]):
    name = "waypoint"
    description = "Realtime Waypoint world model runtime"
    config_cls = WaypointRuntimeConfig
    session_cls = WaypointSession
    outputs = (
        publish.video(
            name="main_video",
            width=WAYPOINT_FRAME_WIDTH,
            height=WAYPOINT_FRAME_HEIGHT,
            fps=WAYPOINT_OUTPUT_FPS,
            pixel_format="rgb24",
        ),
    )
    def __init__(self, config: WaypointRuntimeConfig) -> None:
        super().__init__(config)
        self._engine: WaypointEngine | None = None
        self._compiler_cache_committed = False
        self.compiler_cache_commit_hook: Callable[[object, str], bool] | None = None

    async def load(self, ctx: LoadContext) -> None:
        if self._engine is not None:
            return
        if self.logger is None:
            raise RuntimeError("logger must be bound before loading the model")
        engine_config = self.config
        start = perf_counter()
        warmup_required = not self._has_compiled_cache()
        if warmup_required:
            self.logger.info("waypoint.model.load compiler_cache_miss warmup_required=true")
        else:
            self.logger.info("waypoint.model.load compiler_cache_hit warmup_required=false")
        self._engine = WaypointEngine(engine_config, self.logger)
        try:
            await self._engine.load(warmup=warmup_required)
        except Exception as exc:
            self.logger.error(
                "waypoint.model.load failed duration_ms=%.1f frame_width=%s frame_height=%s fps=%s error_type=%s",
                (perf_counter() - start) * 1000.0,
                WAYPOINT_FRAME_WIDTH,
                WAYPOINT_FRAME_HEIGHT,
                WAYPOINT_OUTPUT_FPS,
                exc.__class__.__name__,
            )
            raise
        if warmup_required:
            self._write_compiled_cache_marker()
            if self.compiler_cache_commit_hook is not None:
                self._compiler_cache_committed = await asyncio.to_thread(
                    self.compiler_cache_commit_hook,
                    self.logger,
                    "post_warmup",
                )
        self.logger.info(
            "waypoint.model.load complete duration_ms=%.1f frame_width=%s frame_height=%s fps=%s",
            (perf_counter() - start) * 1000.0,
            WAYPOINT_FRAME_WIDTH,
            WAYPOINT_FRAME_HEIGHT,
            WAYPOINT_OUTPUT_FPS,
        )

    def create_session(self, ctx: SessionContext) -> WaypointSession:
        return WaypointSession(self, ctx)

    def require_engine(self) -> WaypointEngine:
        if self._engine is None:
            raise RuntimeError("model must be loaded before starting a session")
        return self._engine

    def _has_compiled_cache(self) -> bool:
        marker_path = self._compiled_cache_marker_path()
        if marker_path is None or not marker_path.exists():
            return False
        try:
            payload = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return payload == self._compiled_cache_metadata()

    def _write_compiled_cache_marker(self) -> None:
        marker_path = self._compiled_cache_marker_path()
        if marker_path is None:
            return
        try:
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text(
                json.dumps(self._compiled_cache_metadata(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            logger = self.logger
            if logger is not None:
                logger.warning(
                    "waypoint.model.load compiler_cache_marker_write_failed path=%s error_type=%s",
                    marker_path,
                    exc.__class__.__name__,
                )

    def _compiled_cache_marker_path(self) -> Path | None:
        cache_root = os.getenv("MODAL_COMPILER_CACHE_ROOT", "").strip()
        if not cache_root:
            return None
        return Path(cache_root) / ".waypoint_compile_cache.json"

    def _compiled_cache_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "torch_version": _torch_version(),
            "gpu_type": os.getenv("MODAL_GPU", "").strip() or None,
            "world_engine_commit": os.getenv("WORLD_ENGINE_COMMIT", "").strip() or None,
            "model_source": self.config.waypoint_model_source,
            "ae_source": self.config.waypoint_ae_source,
            "prompt_encoder_source": self.config.waypoint_prompt_encoder_source,
            "frame_width": WAYPOINT_FRAME_WIDTH,
            "frame_height": WAYPOINT_FRAME_HEIGHT,
            "output_fps": WAYPOINT_OUTPUT_FPS,
        }


def _torch_version() -> str | None:
    try:
        import torch
    except Exception:
        return None
    return str(torch.__version__)
