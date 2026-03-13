# Yume on Modal

This example shows the same split as the Waypoint port:

- [`packages/lucid`](../../packages/lucid) owns the reusable runtime contract.
- [`packages/lucid-modal`](../../packages/lucid-modal) owns the Modal worker, dispatch API, and shared Modal boilerplate.
- [`src/yume_modal_example`](src/yume_modal_example) owns the Yume-specific model code and the thin Modal wrapper.

## Port boundary

The example-specific Lucid code is intentionally small:

- [`model.py`](src/yume_modal_example/model.py) defines `YumeLucidModel` and `YumeSession`.
- [`config.py`](src/yume_modal_example/config.py) defines the model config loaded by the example.
- [`modal_app.py`](src/yume_modal_example/modal_app.py) defines only the Yume-specific Modal image, env, volumes, and helper functions, then delegates worker wiring to `lucid_modal.create_app(...)`.

The remaining files are ordinary model-serving code:

- [`engine.py`](src/yume_modal_example/engine.py) owns prompt state and chunk generation.
- [`single_gpu_runtime.py`](src/yume_modal_example/single_gpu_runtime.py) is a thin chunked-session adapter on top of upstream Yume code.

## Runtime contract

The current Yume example is intentionally simple:

- one manual input: `set_prompt`
- one output: `main_video`

That keeps the port small while still exercising the Lucid session lifecycle, manifest generation, and transport protocol.

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

This example keeps its Modal entrypoint in
[`modal_app.py`](src/yume_modal_example/modal_app.py), but the shared Modal runtime now lives in `lucid-modal`.

From the example directory, copy the env file, then create volumes, download the checkpoint, and deploy:

```bash
cd examples/yume_modal
cp modal.env.example .env.yume
uv sync --extra test
uv run lucid-modal create-volumes --env-file .env.yume
uv run lucid-modal download-model --env-file .env.yume -- --repo-id stdstu123/Yume-5B-720P
uv run lucid-modal deploy --env-file .env.yume
```

The required example-owned env is now just the Modal and model/runtime configuration:

```bash
MODAL_APP_ENTRYPOINT=yume_modal_example.modal_app
MODAL_APP_NAME=lucid-runtime-worker
MODAL_GPU=A100
MODAL_MODEL_VOLUME=lucid-yume-models
MODAL_HF_CACHE_VOLUME=lucid-hf-cache
WM_ENGINE=yume
YUME_MODEL_DIR=/models/Yume-5B-720P
YUME_CHUNK_FRAMES=8
YUME_BASE_PROMPT=POV of a character walking in a minecraft scene
```

The first session after a fresh deploy is a cold start. In the current setup, loading the full Yume stack onto an `A100` takes about 3 minutes before the session transitions from `STARTING` to `READY`. After that, the demo sends the selected prompt plus `resume`, and the worker enters `RUNNING`. Subsequent sessions reuse the warm container and start much faster.

## Run through the demo

Start the local coordinator with the deployed Modal dispatch URL and your LiveKit credentials, then run the demo app:

```bash
cargo run -p coordinator
cd apps/demo && bun run dev
```

The demo calls `POST /sessions`, receives the prompt-only Yume manifest, joins the returned LiveKit room, and renders the `main_video` output plus the generated prompt form. For a freshly deployed worker, expect the first demo session to sit in `STARTING` until the model finishes loading, then briefly in `READY` while the client sends the prompt and `resume`.
