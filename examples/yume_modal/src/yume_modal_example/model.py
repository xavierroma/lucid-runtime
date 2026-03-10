from __future__ import annotations

import asyncio
from typing import Annotated

from pydantic import Field

from lucid import SessionContext, VideoModel, action, model, publish

from .config import YumeRuntimeConfig, build_runtime_config
from .engine import YumeEngine


@model(
    name="yume",
    config="configs/yume.yaml",
    description="Realtime Yume world model runtime",
)
class YumeLucidModel(VideoModel):
    main_video = publish.video(
        name="main_video",
        width=1280,
        height=720,
        fps=2,
        pixel_format="rgb24",
    )

    def __init__(self, config: dict[str, object]) -> None:
        super().__init__(config)
        self._engine: YumeEngine | None = None

    def resolve_outputs(self, outputs):
        return (
            publish.video(
                name="main_video",
                width=int(self.runtime_config.frame_width),
                height=int(self.runtime_config.frame_height),
                fps=int(self.runtime_config.target_fps),
                pixel_format="rgb24",
            ),
        )

    async def load(self) -> None:
        if self._engine is not None:
            return
        if self.runtime_config is None or self.logger is None:
            raise RuntimeError("runtime config must be bound before loading the model")
        if not isinstance(self.runtime_config, YumeRuntimeConfig):
            raise RuntimeError("expected YumeRuntimeConfig to be bound")
        self._engine = YumeEngine(self.runtime_config, self.logger)
        await self._engine.load()

    @action(
        name="set_prompt",
        description="Update the scene prompt used by Yume.",
        mode="state",
    )
    def set_prompt(
        self,
        prompt: Annotated[str, Field(..., min_length=1)],
    ) -> None:
        _ = prompt

    async def start_session(self, ctx: SessionContext) -> None:
        if self._engine is None:
            raise RuntimeError("model must be loaded before starting a session")
        runtime_config = self.runtime_config
        if not isinstance(runtime_config, YumeRuntimeConfig):
            raise RuntimeError("expected YumeRuntimeConfig to be bound")
        frame_interval_s = 1.0 / max(int(runtime_config.target_fps), 1)
        prompt = _resolve_prompt(ctx, runtime_config.yume_base_prompt)
        await self._engine.start_session(prompt)
        last_prompt = prompt
        while ctx.running:
            if ctx.paused:
                await asyncio.sleep(0.05)
                continue
            last_prompt, _ = await _sync_prompt(
                self._engine,
                ctx,
                runtime_config.yume_base_prompt,
                last_prompt,
            )
            chunk = await self._engine.generate_chunk()
            last_prompt, prompt_changed = await _sync_prompt(
                self._engine,
                ctx,
                runtime_config.yume_base_prompt,
                last_prompt,
            )
            if prompt_changed:
                self.logger.info(
                    "dropping stale yume chunk after prompt update chunk_frames=%s",
                    len(chunk.frames),
                )
                await asyncio.sleep(0)
                continue
            ctx.record_inference_ms(chunk.inference_ms)
            enqueued_frames = 0
            prompt_changed_during_publish = False
            for idx, frame in enumerate(chunk.frames):
                if not ctx.running:
                    break
                last_prompt, prompt_changed = await _sync_prompt(
                    self._engine,
                    ctx,
                    runtime_config.yume_base_prompt,
                    last_prompt,
                )
                if prompt_changed:
                    prompt_changed_during_publish = True
                    break
                await ctx.publish("main_video", frame)
                enqueued_frames += 1
                if idx + 1 < len(chunk.frames):
                    await asyncio.sleep(frame_interval_s)
            if prompt_changed_during_publish:
                self.logger.info(
                    "stopped yume chunk publish after prompt update published_frames=%s chunk_frames=%s",
                    enqueued_frames,
                    len(chunk.frames),
                )
            output_metrics = ctx.output_metrics()
            self.logger.info(
                (
                    "generated yume chunk chunk_ms=%.2f chunk_frames=%s "
                    "effective_gen_fps=%.2f queue_depth=%s dropped_frames=%s"
                ),
                chunk.chunk_ms,
                enqueued_frames,
                (enqueued_frames * 1000.0) / max(chunk.chunk_ms, 1e-6),
                int(output_metrics.get("queue_depth", 0)),
                int(output_metrics.get("dropped_frames", 0)),
            )
            await asyncio.sleep(0)

    async def end_session(self, ctx: SessionContext) -> None:
        _ = ctx
        if self._engine is not None:
            await self._engine.end_session()


def _resolve_prompt(ctx: SessionContext, default_prompt: str) -> str:
    prompt_state = ctx.state.get("set_prompt")
    prompt = default_prompt.strip() or default_prompt
    if prompt_state is not None:
        prompt = str(getattr(prompt_state, "prompt", default_prompt)).strip() or default_prompt
    return prompt


async def _sync_prompt(
    engine: YumeEngine,
    ctx: SessionContext,
    default_prompt: str,
    last_prompt: str,
) -> tuple[str, bool]:
    prompt = _resolve_prompt(ctx, default_prompt)
    if prompt == last_prompt:
        return last_prompt, False
    await engine.update_prompt(prompt)
    return prompt, True
