from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Annotated

from pydantic import Field

from lucid import (
    LoadContext,
    LucidModel,
    LucidSession,
    SessionContext,
    hold,
    input,
    pointer,
    publish,
    wheel,
)

from .config import WaypointRuntimeConfig
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

    @input(description="Update the text prompt used by Waypoint.")
    def set_prompt(
        self,
        prompt: Annotated[str, Field(..., min_length=1)],
    ) -> None:
        self.prompt = prompt.strip() or self.model.config.waypoint_default_prompt

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
        target_interval_s = 1.0 / max(int(self.model.config.target_fps), 1)
        start = perf_counter()
        try:
            await engine.start_session(self.prompt)
        except Exception as exc:
            self.ctx.logger.error(
                "waypoint.session.run failed duration_ms=%.1f session_id=%s error_type=%s",
                (perf_counter() - start) * 1000.0,
                self.ctx.session_id,
                exc.__class__.__name__,
            )
            raise
        self.ctx.logger.info(
            "waypoint.session.run engine_session_started elapsed_ms=%.1f session_id=%s prompt_chars=%s target_fps=%s frame_width=%s frame_height=%s",
            (perf_counter() - start) * 1000.0,
            self.ctx.session_id,
            len(self.prompt),
            self.model.config.target_fps,
            self.model.config.frame_width,
            self.model.config.frame_height,
        )
        last_prompt = self.prompt
        frame_index = 0
        try:
            while self.ctx.running:
                await self.ctx.wait_if_paused()
                if not self.ctx.running:
                    break
                loop_start_s = asyncio.get_running_loop().time()
                prompt = self.prompt.strip() or self.model.config.waypoint_default_prompt
                if prompt != last_prompt:
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

                elapsed_s = asyncio.get_running_loop().time() - loop_start_s
                # if elapsed_s < target_interval_s:
                #     await asyncio.sleep(target_interval_s - elapsed_s)
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


class WaypointLucidModel(LucidModel[WaypointRuntimeConfig]):
    name = "waypoint"
    description = "Realtime Waypoint world model runtime"
    config_cls = WaypointRuntimeConfig
    outputs = (
        publish.video(
            name="main_video",
            width=640,
            height=360,
            fps=20,
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
        self._engine = WaypointEngine(engine_config, self.logger)
        try:
            await self._engine.load()
        except Exception as exc:
            self.logger.error(
                "waypoint.model.load failed duration_ms=%.1f frame_width=%s frame_height=%s target_fps=%s error_type=%s",
                (perf_counter() - start) * 1000.0,
                engine_config.frame_width,
                engine_config.frame_height,
                engine_config.target_fps,
                exc.__class__.__name__,
            )
            raise
        if engine_config.waypoint_warmup_on_load and self.compiler_cache_commit_hook is not None:
            self._compiler_cache_committed = await asyncio.to_thread(
                self.compiler_cache_commit_hook,
                self.logger,
                "post_warmup",
            )
        self.logger.info(
            "waypoint.model.load complete duration_ms=%.1f frame_width=%s frame_height=%s target_fps=%s",
            (perf_counter() - start) * 1000.0,
            engine_config.frame_width,
            engine_config.frame_height,
            engine_config.target_fps,
        )

    def create_session(self, ctx: SessionContext) -> WaypointSession:
        return WaypointSession(self, ctx)

    def require_engine(self) -> WaypointEngine:
        if self._engine is None:
            raise RuntimeError("model must be loaded before starting a session")
        return self._engine
