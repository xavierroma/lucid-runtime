from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Annotated

from pydantic import Field

from lucid import SessionContext, VideoModel, action, model, publish

from .config import WaypointRuntimeConfig, build_runtime_config
from .engine import WaypointControlState, WaypointEngine


@model(
    name="waypoint",
    config="configs/waypoint.yaml",
    description="Realtime Waypoint world model runtime",
)
class WaypointLucidModel(VideoModel):
    main_video = publish.video(
        name="main_video",
        width=640,
        height=360,
        fps=20,
        pixel_format="rgb24",
    )

    def __init__(self, config: dict[str, object]) -> None:
        super().__init__(config)
        self._engine: WaypointEngine | None = None
        self._compiler_cache_committed = False
        self._transient_inputs: dict[str, _TransientWaypointInput] = {}
        self._transient_inputs_lock = asyncio.Lock()

    def resolve_outputs(self, outputs):
        runtime_config = self.runtime_config
        if not isinstance(runtime_config, WaypointRuntimeConfig):
            raise RuntimeError("expected WaypointRuntimeConfig to be bound")
        return (
            publish.video(
                name="main_video",
                width=int(runtime_config.frame_width),
                height=int(runtime_config.frame_height),
                fps=int(runtime_config.target_fps),
                pixel_format="rgb24",
            ),
        )

    async def load(self) -> None:
        if self._engine is not None:
            return
        if self.runtime_config is None or self.logger is None:
            raise RuntimeError("runtime config must be bound before loading the model")
        if not isinstance(self.runtime_config, WaypointRuntimeConfig):
            raise RuntimeError("expected WaypointRuntimeConfig to be bound")
        start = perf_counter()
        self._engine = WaypointEngine(self.runtime_config, self.logger)
        try:
            await self._engine.load()
        except Exception as exc:
            self.logger.error(
                "waypoint.model.load failed duration_ms=%.1f frame_width=%s frame_height=%s target_fps=%s error_type=%s",
                (perf_counter() - start) * 1000.0,
                self.runtime_config.frame_width,
                self.runtime_config.frame_height,
                self.runtime_config.target_fps,
                exc.__class__.__name__,
            )
            raise
        if self.runtime_config.waypoint_warmup_on_load:
            self._compiler_cache_committed = await asyncio.to_thread(
                _commit_compiler_cache_volume,
                self.logger,
                "post_warmup",
            )
        self.logger.info(
            "waypoint.model.load complete duration_ms=%.1f frame_width=%s frame_height=%s target_fps=%s",
            (perf_counter() - start) * 1000.0,
            self.runtime_config.frame_width,
            self.runtime_config.frame_height,
            self.runtime_config.target_fps,
        )

    @action(
        name="set_prompt",
        description="Update the text prompt used by Waypoint.",
        mode="state",
    )
    def set_prompt(
        self,
        prompt: Annotated[str, Field(..., min_length=1)],
    ) -> None:
        _ = prompt

    @action(
        name="set_buttons",
        description="Set the held button IDs for Waypoint.",
        mode="state",
    )
    def set_buttons(
        self,
        buttons: list[Annotated[int, Field(ge=0, le=255)]] = Field(default_factory=list),
    ) -> None:
        _ = buttons

    @action(
        name="mouse_move",
        description="Apply a relative mouse movement delta for the next Waypoint frame.",
        mode="command",
    )
    async def mouse_move(
        self,
        ctx: SessionContext,
        dx: float = 0.0,
        dy: float = 0.0,
    ) -> None:
        await self._append_transient_input(
            ctx.session_id,
            dx=float(dx),
            dy=float(dy),
        )

    @action(
        name="scroll",
        description="Apply a relative scroll wheel delta for the next Waypoint frame.",
        mode="command",
    )
    async def scroll(
        self,
        ctx: SessionContext,
        amount: int = 0,
    ) -> None:
        await self._append_transient_input(
            ctx.session_id,
            scroll_amount=int(amount),
        )

    async def start_session(self, ctx: SessionContext) -> None:
        if self._engine is None:
            raise RuntimeError("model must be loaded before starting a session")
        runtime_config = self.runtime_config
        if not isinstance(runtime_config, WaypointRuntimeConfig):
            raise RuntimeError("expected WaypointRuntimeConfig to be bound")

        target_interval_s = 1.0 / max(int(runtime_config.target_fps), 1)
        prompt = _resolve_prompt(ctx, runtime_config.waypoint_default_prompt)
        start = perf_counter()
        try:
            await self._engine.start_session(prompt)
        except Exception as exc:
            ctx.logger.error(
                "waypoint.model.start_session failed duration_ms=%.1f session_id=%s error_type=%s",
                (perf_counter() - start) * 1000.0,
                ctx.session_id,
                exc.__class__.__name__,
            )
            raise
        ctx.logger.info(
            "waypoint.model.start_session engine_session_started elapsed_ms=%.1f session_id=%s prompt_chars=%s target_fps=%s frame_width=%s frame_height=%s",
            (perf_counter() - start) * 1000.0,
            ctx.session_id,
            len(prompt),
            runtime_config.target_fps,
            runtime_config.frame_width,
            runtime_config.frame_height,
        )
        last_prompt = prompt
        frame_index = 0
        try:
            while ctx.running:
                if ctx.paused:
                    await asyncio.sleep(0.05)
                    continue

                loop_start_s = asyncio.get_running_loop().time()
                prompt = _resolve_prompt(ctx, runtime_config.waypoint_default_prompt)
                if prompt != last_prompt:
                    await self._engine.update_prompt(prompt)
                    last_prompt = prompt
                    ctx.logger.info(
                        "waypoint.model.start_session prompt_updated elapsed_ms=%.1f session_id=%s frame_index=%s prompt_chars=%s",
                        (perf_counter() - start) * 1000.0,
                        ctx.session_id,
                        frame_index,
                        len(prompt),
                    )

                controls = WaypointControlState(
                    buttons=_resolve_buttons(ctx),
                    **(await self._drain_transient_input(ctx.session_id)).as_control_kwargs(),
                )
                frame, inference_ms = await self._engine.generate_frame(controls)
                ctx.record_inference_ms(inference_ms)
                frame_index += 1
                if frame_index == 1:
                    ctx.logger.info(
                        "waypoint.model.start_session first_frame_ready elapsed_ms=%.1f session_id=%s inference_ms=%.1f",
                        (perf_counter() - start) * 1000.0,
                        ctx.session_id,
                        inference_ms,
                    )
                await ctx.publish("main_video", frame)
                if frame_index == 1:
                    ctx.logger.info(
                        "waypoint.model.start_session first_frame_published elapsed_ms=%.1f session_id=%s",
                        (perf_counter() - start) * 1000.0,
                        ctx.session_id,
                    )
                    if not self._compiler_cache_committed:
                        self._compiler_cache_committed = await asyncio.to_thread(
                            _commit_compiler_cache_volume,
                            ctx.logger,
                            "first_frame",
                        )

                elapsed_s = asyncio.get_running_loop().time() - loop_start_s
                if elapsed_s < target_interval_s:
                    await asyncio.sleep(target_interval_s - elapsed_s)
        except Exception as exc:
            ctx.logger.error(
                "waypoint.model.start_session failed duration_ms=%.1f session_id=%s frames_generated=%s inference_ms_p50=%.1f error_type=%s",
                (perf_counter() - start) * 1000.0,
                ctx.session_id,
                frame_index,
                ctx.inference_ms_p50(),
                exc.__class__.__name__,
            )
            raise
        ctx.logger.info(
            "waypoint.model.start_session complete duration_ms=%.1f session_id=%s frames_generated=%s inference_ms_p50=%.1f",
            (perf_counter() - start) * 1000.0,
            ctx.session_id,
            frame_index,
            ctx.inference_ms_p50(),
        )

    async def end_session(self, ctx: SessionContext) -> None:
        try:
            if self._engine is not None:
                await self._engine.end_session()
        finally:
            await self._clear_transient_input(ctx.session_id)
            logger = self.logger or ctx.logger
            if logger is not None:
                self._compiler_cache_committed = (
                    await asyncio.to_thread(_commit_compiler_cache_volume, logger, "session_end")
                ) or self._compiler_cache_committed

    async def _append_transient_input(
        self,
        session_id: str,
        *,
        dx: float = 0.0,
        dy: float = 0.0,
        scroll_amount: int = 0,
    ) -> None:
        async with self._transient_inputs_lock:
            current = self._transient_inputs.setdefault(session_id, _TransientWaypointInput())
            current.mouse_dx += dx
            current.mouse_dy += dy
            current.scroll_amount += scroll_amount

    async def _drain_transient_input(self, session_id: str) -> "_TransientWaypointInput":
        async with self._transient_inputs_lock:
            current = self._transient_inputs.setdefault(session_id, _TransientWaypointInput())
            drained = _TransientWaypointInput(
                mouse_dx=current.mouse_dx,
                mouse_dy=current.mouse_dy,
                scroll_amount=current.scroll_amount,
            )
            current.mouse_dx = 0.0
            current.mouse_dy = 0.0
            current.scroll_amount = 0
            return drained

    async def _clear_transient_input(self, session_id: str) -> None:
        async with self._transient_inputs_lock:
            self._transient_inputs.pop(session_id, None)


@dataclass(slots=True)
class _TransientWaypointInput:
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    scroll_amount: int = 0

    def as_control_kwargs(self) -> dict[str, float | int]:
        return {
            "mouse_dx": self.mouse_dx,
            "mouse_dy": self.mouse_dy,
            "scroll_amount": self.scroll_amount,
        }


def _resolve_prompt(ctx: SessionContext, default_prompt: str) -> str:
    prompt_state = ctx.state.get("set_prompt")
    prompt = default_prompt.strip() or default_prompt
    if prompt_state is not None:
        prompt = str(getattr(prompt_state, "prompt", default_prompt)).strip() or default_prompt
    return prompt


def _resolve_buttons(ctx: SessionContext) -> frozenset[int]:
    button_state = ctx.state.get("set_buttons")
    if button_state is None:
        return frozenset()
    raw_buttons = getattr(button_state, "buttons", [])
    return frozenset(int(button) for button in raw_buttons)


def _commit_compiler_cache_volume(logger, reason: str = "session_end") -> bool:
    volume_name = os.getenv("MODAL_HF_CACHE_VOLUME", "").strip()
    cache_root = os.getenv("MODAL_COMPILER_CACHE_ROOT", "").strip()
    if not volume_name or not cache_root or not Path(cache_root).exists():
        return False

    try:
        import modal
    except Exception:
        return False

    try:
        modal.Volume.from_name(volume_name).commit()
    except Exception as exc:
        logger.warning(
            "waypoint.model.compiler_cache_commit_failed volume=%s root=%s reason=%s error_type=%s",
            volume_name,
            cache_root,
            reason,
            exc.__class__.__name__,
        )
        return False

    logger.info(
        "waypoint.model.compiler_cache_committed volume=%s root=%s reason=%s",
        volume_name,
        cache_root,
        reason,
    )
    return True
