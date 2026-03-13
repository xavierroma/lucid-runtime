# Helios-Distilled on Modal

This example shows how to port a prompt-driven video model to the Lucid runtime using the same package split as the existing examples:

- [`packages/lucid`](../../packages/lucid) owns the reusable runtime contract.
- [`packages/lucid-modal`](../../packages/lucid-modal) owns the Modal worker lifecycle and helper CLI.
- [`src/helios_modal_example`](src/helios_modal_example) owns the Helios-specific config, engine, model, and Modal wrapper.

## Port boundary

The Helios-specific Lucid layer is intentionally small:

- [`model.py`](src/helios_modal_example/model.py) defines `HeliosLucidModel` and `HeliosSession`.
- [`config.py`](src/helios_modal_example/config.py) defines the env-backed model config.
- [`modal_app.py`](src/helios_modal_example/modal_app.py) defines only the Helios-specific Modal image, env, volumes, and `download_model`, then delegates worker wiring to `lucid_modal.create_app(...)`.

[`engine.py`](src/helios_modal_example/engine.py) is ordinary model-serving code. It wraps the Hugging Face `HeliosPyramidPipeline`, generates one chunk per call, and feeds the previously emitted chunk back in as `video=` conditioning so Lucid can keep a long-running session alive.

## Runtime contract

The Helios example exposes:

- one manual input: `set_prompt`
- one output: `main_video`

The current output size is `640x384` at `24 fps`.

Prompt changes are chunk-boundary updates. If the user changes the prompt while a chunk is generating or publishing, the current chunk finishes first and the new prompt is used for the next chunk.

## Local iteration

Install the example and test extras:

```bash
uv sync --project examples/helios_modal --extra test
```

Run the example tests:

```bash
uv run --project examples/helios_modal --extra test pytest examples/helios_modal/tests -q
```

## Deploy on Modal

From the example directory, copy the env file, then create volumes, download the checkpoint, and deploy:

```bash
cd examples/helios_modal
cp modal.env.example .env.helios
uv sync --extra test
uv run lucid-modal create-volumes --env-file .env.helios
uv run lucid-modal download-model --env-file .env.helios
uv run lucid-modal deploy --env-file .env.helios
```

The default deploy target is `MODAL_GPU=H100`. The model card for [`BestWishYsh/Helios-Distilled`](https://huggingface.co/BestWishYsh/Helios-Distilled) documents both the standard `HeliosPyramidPipeline` path and an optional group-offloading mode. This example keeps group offloading disabled by default and exposes it through:

```bash
HELIOS_ENABLE_GROUP_OFFLOADING=1
HELIOS_GROUP_OFFLOADING_TYPE=leaf_level
```

## Run through the demo

Start the local coordinator with the deployed Modal dispatch URL and your LiveKit credentials, then run the demo app:

```bash
cargo run -p coordinator
cd apps/demo && bun run dev
```

Choose `Helios` in the model picker, create a session, and update the environment prompt from the existing prompt editor. The next generated chunk should reflect the updated prompt.
