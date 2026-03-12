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

from .config import YumeRuntimeConfig
from .engine import YumeEngine


class YumeSession(LucidSession["YumeLucidModel"]):
    def __init__(self, model: "YumeLucidModel", ctx: SessionContext) -> None:
        super().__init__(model, ctx)
        self.prompt = model.config.yume_base_prompt

    @input(description="Update the scene prompt used by Yume.")
    def set_prompt(
        self,
        prompt: Annotated[str, Field(..., min_length=1)],
    ) -> None:
        self.prompt = prompt.strip() or self.model.config.yume_base_prompt

    async def run(self) -> None:
        engine = self.model.require_engine()
        frame_interval_s = 1.0 / max(int(self.model.config.target_fps), 1)
        await engine.start_session(self.prompt)
        last_prompt = self.prompt
        while self.ctx.running:
            last_prompt, _ = await _sync_prompt(engine, self, last_prompt)
            chunk = await engine.generate_chunk()
            last_prompt, prompt_changed = await _sync_prompt(engine, self, last_prompt)
            if prompt_changed:
                logger = self.model.logger
                if logger is not None:
                    logger.info(
                        "dropping stale yume chunk after prompt update chunk_frames=%s",
                        len(chunk.frames),
                    )
                await asyncio.sleep(0)
                continue
            self.ctx.record_inference_ms(chunk.inference_ms)
            enqueued_frames = 0
            prompt_changed_during_publish = False
            for idx, frame in enumerate(chunk.frames):
                if not self.ctx.running:
                    break
                last_prompt, prompt_changed = await _sync_prompt(engine, self, last_prompt)
                if prompt_changed:
                    prompt_changed_during_publish = True
                    break
                await self.ctx.publish("main_video", frame)
                enqueued_frames += 1
                if idx + 1 < len(chunk.frames):
                    await asyncio.sleep(frame_interval_s)
            if prompt_changed_during_publish and self.model.logger is not None:
                self.model.logger.info(
                    "stopped yume chunk publish after prompt update published_frames=%s chunk_frames=%s",
                    enqueued_frames,
                    len(chunk.frames),
                )
            if self.model.logger is not None:
                output_metrics = self.ctx.output_metrics()
                self.model.logger.info(
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

    async def close(self) -> None:
        engine = self.model._engine
        if engine is not None:
            await engine.end_session()


class YumeLucidModel(LucidModel[YumeRuntimeConfig]):
    name = "yume"
    description = "Realtime Yume world model runtime"
    config_cls = YumeRuntimeConfig
    outputs = (
        publish.video(
            name="main_video",
            width=1280,
            height=720,
            fps=2,
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
