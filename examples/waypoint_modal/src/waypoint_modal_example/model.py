from __future__ import annotations

import asyncio
import os
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
        name="set_controls",
        description="Set held controls and look velocity for Waypoint.",
        mode="state",
    )
    def set_controls(
        self,
        forward: bool = False,
        backward: bool = False,
        left: bool = False,
        right: bool = False,
        jump: bool = False,
        sprint: bool = False,
        crouch: bool = False,
        primary_fire: bool = False,
        secondary_fire: bool = False,
        mouse_x: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0,
        mouse_y: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0,
        scroll_wheel: Annotated[int, Field(ge=-1, le=1)] = 0,
    ) -> None:
        _ = (
            forward,
            backward,
            left,
            right,
            jump,
            sprint,
            crouch,
            primary_fire,
            secondary_fire,
            mouse_x,
            mouse_y,
            scroll_wheel,
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

                controls = _resolve_controls(ctx)
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
            logger = self.logger or ctx.logger
            if logger is not None:
                await asyncio.to_thread(_commit_compiler_cache_volume, logger)


def _resolve_prompt(ctx: SessionContext, default_prompt: str) -> str:
    prompt_state = ctx.state.get("set_prompt")
    prompt = default_prompt.strip() or default_prompt
    if prompt_state is not None:
        prompt = str(getattr(prompt_state, "prompt", default_prompt)).strip() or default_prompt
    return prompt


def _resolve_controls(ctx: SessionContext) -> WaypointControlState:
    control_state = ctx.state.get("set_controls")
    if control_state is None:
        return WaypointControlState()
    return WaypointControlState(
        forward=bool(getattr(control_state, "forward", False)),
        backward=bool(getattr(control_state, "backward", False)),
        left=bool(getattr(control_state, "left", False)),
        right=bool(getattr(control_state, "right", False)),
        jump=bool(getattr(control_state, "jump", False)),
        sprint=bool(getattr(control_state, "sprint", False)),
        crouch=bool(getattr(control_state, "crouch", False)),
        primary_fire=bool(getattr(control_state, "primary_fire", False)),
        secondary_fire=bool(getattr(control_state, "secondary_fire", False)),
        mouse_x=float(getattr(control_state, "mouse_x", 0.0)),
        mouse_y=float(getattr(control_state, "mouse_y", 0.0)),
        scroll_wheel=int(getattr(control_state, "scroll_wheel", 0)),
    )


def _commit_compiler_cache_volume(logger) -> None:
    volume_name = os.getenv("MODAL_HF_CACHE_VOLUME", "").strip()
    cache_root = os.getenv("MODAL_COMPILER_CACHE_ROOT", "").strip()
    if not volume_name or not cache_root or not Path(cache_root).exists():
        return

    try:
        import modal
    except Exception:
        return

    try:
        modal.Volume.from_name(volume_name).commit()
    except Exception as exc:
        logger.warning(
            "waypoint.model.end_session compiler_cache_commit_failed volume=%s root=%s error_type=%s",
            volume_name,
            cache_root,
            exc.__class__.__name__,
        )
        return

    logger.info(
        "waypoint.model.end_session compiler_cache_committed volume=%s root=%s",
        volume_name,
        cache_root,
    )
