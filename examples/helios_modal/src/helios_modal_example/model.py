from __future__ import annotations

import asyncio
from time import perf_counter
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

from .config import HeliosRuntimeConfig
from .engine import HeliosEngine


class HeliosSession(LucidSession["HeliosLucidModel"]):
    def __init__(self, model: "HeliosLucidModel", ctx: SessionContext) -> None:
        super().__init__(model, ctx)
        self.prompt = model.config.helios_default_prompt

    @input(description="Update the scene prompt used by Helios.")
    def set_prompt(
        self,
        prompt: Annotated[str, Field(..., min_length=1)],
    ) -> None:
        self.prompt = prompt.strip() or self.model.config.helios_default_prompt

    async def run(self) -> None:
        engine = self.model.require_engine()
        frame_interval_s = 1.0 / max(int(self.model.config.output_fps), 1)
        await engine.start_session(self.prompt)
        last_prompt = self.prompt
        chunk_index = 0
        started = perf_counter()

        while self.ctx.running:
            await self.ctx.wait_if_paused()
            if not self.ctx.running:
                break

            prompt = self.prompt.strip() or self.model.config.helios_default_prompt
            if prompt != last_prompt:
                await engine.update_prompt(prompt)
                last_prompt = prompt
                if self.model.logger is not None:
                    self.model.logger.info(
                        "helios.session.prompt_updated session_id=%s chunk_index=%s prompt_len=%s",
                        self.ctx.session_id,
                        chunk_index,
                        len(prompt),
                    )

            chunk = await engine.generate_chunk()
            chunk_index += 1
            self.ctx.record_inference_ms(chunk.inference_ms)

            if self.model.logger is not None:
                self.model.logger.info(
                    "helios.session.chunk_ready session_id=%s chunk_index=%s chunk_frames=%s chunk_ms=%.2f prompt_len=%s",
                    self.ctx.session_id,
                    chunk_index,
                    len(chunk.frames),
                    chunk.chunk_ms,
                    len(last_prompt),
                )

            for frame_index, frame in enumerate(chunk.frames):
                await self.ctx.wait_if_paused()
                if not self.ctx.running:
                    break
                await self.ctx.publish("main_video", frame)
                if frame_index + 1 < len(chunk.frames):
                    await asyncio.sleep(frame_interval_s)

        if self.model.logger is not None:
            self.model.logger.info(
                "helios.session.complete duration_ms=%.1f session_id=%s chunks_generated=%s inference_ms_p50=%.1f",
                (perf_counter() - started) * 1000.0,
                self.ctx.session_id,
                chunk_index,
                self.ctx.inference_ms_p50(),
            )

    async def close(self) -> None:
        engine = self.model._engine
        if engine is not None:
            await engine.end_session()


class HeliosLucidModel(LucidModel[HeliosRuntimeConfig]):
    name = "helios"
    description = "Realtime Helios video generation runtime"
    config_cls = HeliosRuntimeConfig
    outputs = (
        publish.video(
            name="main_video",
            width=640,
            height=384,
            fps=24,
            pixel_format="rgb24",
        ),
    )

    def __init__(self, config: HeliosRuntimeConfig) -> None:
        super().__init__(config)
        self._engine: HeliosEngine | None = None

    async def load(self, ctx: LoadContext) -> None:
        if self._engine is not None:
            return
        if self.logger is None:
            raise RuntimeError("logger must be bound before loading the model")
        self._engine = HeliosEngine(self.config, self.logger)
        await self._engine.load()

    def create_session(self, ctx: SessionContext) -> HeliosSession:
        return HeliosSession(self, ctx)

    def require_engine(self) -> HeliosEngine:
        if self._engine is None:
            raise RuntimeError("model must be loaded before starting a session")
        return self._engine
