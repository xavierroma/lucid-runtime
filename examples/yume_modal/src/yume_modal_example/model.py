from __future__ import annotations

import asyncio
from typing import Annotated

from pydantic import Field

from lucid import (
    LoadContext,
    LucidModel,
    LucidSession,
    SessionContext,
    input,
    publish,
)

from .config import (
    YUME_FRAME_HEIGHT,
    YUME_FRAME_WIDTH,
    YUME_OUTPUT_FPS,
    YumeRuntimeConfig,
)
from .engine import YumeEngine


class YumeSession(LucidSession["YumeLucidModel"]):
    def __init__(self, model: "YumeLucidModel", ctx: SessionContext) -> None:
        super().__init__(model, ctx)
        self.prompt = model.config.yume_base_prompt

    @input(description="Update the scene prompt used by Yume.", paused=True)
    def set_prompt(
        self,
        prompt: Annotated[str, Field(..., min_length=1)],
    ) -> None:
        self.prompt = prompt.strip() or self.model.config.yume_base_prompt

    async def run(self) -> None:
        engine = self.model.require_engine()
        frame_interval_s = 1.0 / YUME_OUTPUT_FPS
        await engine.start_session(self.prompt)
        last_prompt = self.prompt
        pending_chunk = None
        pending_chunk_frame_index = 0
        pending_chunk_locked_by_pause = False
        pending_chunk_published_frames = 0
        while self.ctx.running:
            await self.ctx.wait_if_paused()
            if not self.ctx.running:
                break
            if pending_chunk is None:
                last_prompt, _ = await _sync_prompt(engine, self, last_prompt)
                pending_chunk = await engine.generate_chunk()
                pending_chunk_frame_index = 0
                pending_chunk_published_frames = 0
                pending_chunk_locked_by_pause = self.ctx.is_paused()
                await self.ctx.wait_if_paused()
                if not self.ctx.running:
                    break
                if not pending_chunk_locked_by_pause:
                    last_prompt, prompt_changed = await _sync_prompt(engine, self, last_prompt)
                    if prompt_changed:
                        logger = self.model.logger
                        if logger is not None:
                            logger.info(
                                "dropping stale yume chunk after prompt update chunk_frames=%s",
                                len(pending_chunk.frames),
                            )
                        pending_chunk = None
                        await asyncio.sleep(0)
                        continue
                self.ctx.record_inference_ms(pending_chunk.inference_ms)
            prompt_changed_during_publish = False
            while pending_chunk is not None and pending_chunk_frame_index < len(pending_chunk.frames):
                if not self.ctx.running:
                    break
                await self.ctx.wait_if_paused()
                if not self.ctx.running:
                    break
                if not pending_chunk_locked_by_pause:
                    last_prompt, prompt_changed = await _sync_prompt(engine, self, last_prompt)
                    if prompt_changed:
                        prompt_changed_during_publish = True
                        break
                frame = pending_chunk.frames[pending_chunk_frame_index]
                await self.ctx.publish("main_video", frame)
                pending_chunk_published_frames += 1
                pending_chunk_frame_index += 1
                if self.ctx.is_paused():
                    pending_chunk_locked_by_pause = True
                    continue
                if pending_chunk_frame_index < len(pending_chunk.frames):
                    await asyncio.sleep(frame_interval_s)
            if prompt_changed_during_publish and self.model.logger is not None:
                self.model.logger.info(
                    "stopped yume chunk publish after prompt update published_frames=%s chunk_frames=%s",
                    pending_chunk_published_frames,
                    len(pending_chunk.frames) if pending_chunk is not None else 0,
                )
                pending_chunk = None
                pending_chunk_frame_index = 0
                pending_chunk_published_frames = 0
                pending_chunk_locked_by_pause = False
                await asyncio.sleep(0)
                continue
            if pending_chunk is None:
                continue
            if pending_chunk_frame_index < len(pending_chunk.frames):
                await asyncio.sleep(0)
                continue
            if self.model.logger is not None:
                self.model.logger.info(
                    (
                        "generated yume chunk chunk_ms=%.2f chunk_frames=%s effective_gen_fps=%.2f"
                    ),
                    pending_chunk.chunk_ms,
                    pending_chunk_published_frames,
                    (pending_chunk_published_frames * 1000.0)
                    / max(pending_chunk.chunk_ms, 1e-6),
                )
            pending_chunk = None
            pending_chunk_frame_index = 0
            pending_chunk_published_frames = 0
            pending_chunk_locked_by_pause = False
            await asyncio.sleep(0)

    async def close(self) -> None:
        engine = self.model._engine
        if engine is not None:
            await engine.end_session()


class YumeLucidModel(LucidModel[YumeRuntimeConfig]):
    name = "yume"
    description = "Realtime Yume world model runtime"
    config_cls = YumeRuntimeConfig
    session_cls = YumeSession
    outputs = (
        publish.video(
            name="main_video",
            width=YUME_FRAME_WIDTH,
            height=YUME_FRAME_HEIGHT,
            fps=YUME_OUTPUT_FPS,
            pixel_format="rgb24",
        ),
    )
    def __init__(self, config: YumeRuntimeConfig) -> None:
        super().__init__(config)
        self._engine: YumeEngine | None = None

    async def load(self, ctx: LoadContext) -> None:
        if self._engine is not None:
            return
        if self.logger is None:
            raise RuntimeError("logger must be bound before loading the model")
        self._engine = YumeEngine(self.config, self.logger)
        await self._engine.load()

    def create_session(self, ctx: SessionContext) -> YumeSession:
        return YumeSession(self, ctx)

    def require_engine(self) -> YumeEngine:
        if self._engine is None:
            raise RuntimeError("model must be loaded before starting a session")
        return self._engine


async def _sync_prompt(
    engine: YumeEngine,
    session: YumeSession,
    last_prompt: str,
) -> tuple[str, bool]:
    prompt = session.prompt.strip() or session.model.config.yume_base_prompt
    if prompt == last_prompt:
        return last_prompt, False
    await engine.update_prompt(prompt)
    return prompt, True
