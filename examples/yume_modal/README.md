# Yume on Modal

This example shows the intended Lucid port boundary for a real model.

- The reusable runtime and hosting code lives in [`packages/lucid`](../../packages/lucid).
- The Yume-specific port lives in [`src/yume_modal_example`](src/yume_modal_example).

## What is the Lucid port

The Lucid-specific code is intentionally small:

- [`model.py`](src/yume_modal_example/model.py) declares the model with `@model`, declares `main_video`, exposes `set_prompt`, and runs the session loop with `await ctx.publish(...)`.
- [`config.py`](src/yume_modal_example/config.py) adapts the generic Lucid host config into the Yume-specific runtime config.
- [`modal_app.py`](src/yume_modal_example/modal_app.py) defines the Modal image and calls `lucid.modal.create_app(...)`.

That is the part you rewrite when you port a new model into Lucid.

## What is just the model

These files are ordinary Yume/Torch code, not Lucid concepts:

- [`engine.py`](src/yume_modal_example/engine.py) owns prompt state and chunk generation.
- [`single_gpu_runtime.py`](src/yume_modal_example/single_gpu_runtime.py) is a thin chunked-session adapter on top of upstream `wan23.Yume`.
- [`configs/yume.yaml`](src/yume_modal_example/configs/yume.yaml) and the `YUME_*` env vars are just model/runtime configuration.

If you were porting a different Torch model, most of the work would happen in equivalents of `engine.py` and `single_gpu_runtime.py`. Lucid should not force you to rewrite that logic.

## What it takes to add a Lucid port

For an existing model codebase, the minimum Lucid port usually looks like this:

1. Wrap the model in a `@model(...)` class.
2. Declare its outputs with `publish.video(...)`, `publish.audio(...)`, `publish.json(...)`, or `publish.bytes(...)`.
3. Expose the supported controls as typed `@action(...)` methods.
4. In `start_session(ctx)`, read Lucid state from `ctx.state`, call into your existing inference code, and publish outputs with `ctx.publish(...)`.
5. Add a `modal_app.py` only if you want a production deployment target on Modal.

The current Yume port is intentionally prompt-only because that is the only control the deployed model actually supports.

The port boundary in practice is:

- Lucid port code:
  [`model.py`](src/yume_modal_example/model.py),
  [`config.py`](src/yume_modal_example/config.py),
  [`modal_app.py`](src/yume_modal_example/modal_app.py)
- Plain Yume/Torch code:
  [`engine.py`](src/yume_modal_example/engine.py),
  [`single_gpu_runtime.py`](src/yume_modal_example/single_gpu_runtime.py),
  upstream `wan23` modules cloned by Modal,
  checkpoint files in `YUME_MODEL_DIR`

## Current Yume Lucid model

```python
import asyncio
from typing import Annotated

from pydantic import Field

from lucid import SessionContext, VideoModel, action, model, publish

from yume_modal_example.config import YumeRuntimeConfig
from yume_modal_example.engine import YumeEngine


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

    async def load(self) -> None:
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
        runtime_config = self.runtime_config
        assert isinstance(runtime_config, YumeRuntimeConfig)
        frame_interval_s = 1.0 / max(int(runtime_config.target_fps), 1)
        prompt = runtime_config.yume_base_prompt
        await self._engine.start_session(prompt)
        last_prompt = prompt
        while ctx.running:
            prompt = _resolve_prompt(ctx, runtime_config.yume_base_prompt)
            if prompt != last_prompt:
                await self._engine.update_prompt(prompt)
                last_prompt = prompt
            chunk = await self._engine.generate_chunk()
            prompt = _resolve_prompt(ctx, runtime_config.yume_base_prompt)
            if prompt != last_prompt:
                await self._engine.update_prompt(prompt)
                last_prompt = prompt
                continue
            for index, frame in enumerate(chunk.frames):
                prompt = _resolve_prompt(ctx, runtime_config.yume_base_prompt)
                if prompt != last_prompt:
                    await self._engine.update_prompt(prompt)
                    last_prompt = prompt
                    break
                await ctx.publish("main_video", frame)
                if index + 1 < len(chunk.frames):
                    await asyncio.sleep(frame_interval_s)
```

The full implementation is in [`model.py`](src/yume_modal_example/model.py).

## Local iteration

Install the example and test extras:

```bash
uv sync --project examples/yume_modal --extra test
```

Run the example tests:

```bash
uv run --project examples/yume_modal --extra test pytest examples/yume_modal/tests -q
```

## Deploy on Modal

This example owns its own Modal deployment module at
[`modal_app.py`](src/yume_modal_example/modal_app.py).

To deploy it:

```bash
cp deploy/modal/.env.example deploy/modal/.env
deploy/modal/create-volumes.sh
deploy/modal/download-model.sh -- --repo-id stdstu123/Yume-5B-720P
WM_MODEL_MODULE=yume_modal_example.model WM_MODEL_NAME=yume deploy/modal/deploy.sh
```

Required env for the deployed example:

```bash
WM_MODEL_MODULE=yume_modal_example.model
WM_MODEL_NAME=yume
WM_ENGINE=yume
YUME_MODEL_DIR=/models/Yume-5B-720P
```

The first session after a fresh deploy is a cold start. In the current setup, loading the full
Yume stack onto an `A100` takes about 3 minutes before the session transitions from `STARTING`
to `RUNNING`. Subsequent sessions reuse the warm container and start much faster.

## Run through the demo

Start the local coordinator with the deployed Modal dispatch URL and your LiveKit credentials,
then run the demo app:

```bash
cargo run -p coordinator
cd apps/demo && bun run dev
```

The demo calls `POST /api/sessions`, receives the prompt-only Yume manifest, joins the returned
LiveKit room, and renders:

- one `set_prompt` form with `Apply`
- the built-in Lucid runtime controls
- the `main_video` output as the primary track

For a freshly deployed worker, expect the first demo session to sit in `STARTING` until the
model finishes loading. That is normal.

## Session flow

The demo and clients interact with the coordinator, not the Modal worker directly:

1. `POST /sessions`
2. Join the returned LiveKit room with `client_access_token`
3. Subscribe to the `main_video` track
4. Publish action envelopes to `wm.control`

Example `set_prompt` control message:

```json
{
  "type": "action",
  "seq": 1,
  "ts_ms": 1741315200000,
  "session_id": "3acb0b65-7b3c-4ebb-8e98-9e18dbf7403f",
  "payload": {
    "name": "set_prompt",
    "args": {
      "prompt": "A rainy neon alley at dusk with reflective puddles"
    }
  }
}
```
