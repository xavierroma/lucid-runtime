# Waypoint on Modal

This example uses the single-package Lucid layout:

- [`packages/lucid`](../../packages/lucid) owns the reusable runtime contract, LiveKit host, Modal adapter, and dev helpers.
- [`src/waypoint_modal_example`](src/waypoint_modal_example) owns the Waypoint-specific model code and the thin Modal wrapper.

## Port boundary

The example-specific Lucid code is intentionally small:

- [`model.py`](src/waypoint_modal_example/model.py) defines `WaypointLucidModel` and `WaypointSession`.
- [`config.py`](src/waypoint_modal_example/config.py) defines the model config loaded by the example.
- [`modal_app.py`](src/waypoint_modal_example/modal_app.py) defines only the Waypoint-specific Modal image, env, volumes, and hooks, then delegates worker wiring to `lucid.modal.create_app(...)`.

[`engine.py`](src/waypoint_modal_example/engine.py) is ordinary model-serving code. It wraps the upstream `world_engine.WorldEngine`, keeps CUDA work on a dedicated thread, seeds the model with a starter frame, and converts generated frames into Lucid-ready `numpy.uint8` RGB frames.

## Runtime contract

The Waypoint example exposes:

- one manual input: `set_prompt`
- hold bindings for movement, jump, sprint, crouch, and mouse buttons
- a pointer binding for relative look deltas
- a wheel binding for scroll input
- one output: `main_video`

The default output size is `640x360`, which matches the current Waypoint 1.x legacy-model path used by Overworld's Biome server.

## Local iteration

Install the example and test extras:

```bash
uv sync --project examples/waypoint_modal --extra test
```

Run the example tests:

```bash
uv run --project examples/waypoint_modal --extra test pytest examples/waypoint_modal/tests -q
```

## Deploy on Modal

This example keeps its Modal entrypoint in
[`modal_app.py`](src/waypoint_modal_example/modal_app.py), but the shared Modal runtime now lives in `lucid.modal`.

From the example directory, copy the env file, then create volumes, download the checkpoints, and deploy:

```bash
cd examples/waypoint_modal
cp modal.env.example .env.waypoint
uv sync --extra test
uv run lucid-modal create-volumes --env-file .env.waypoint
uv run lucid-modal download-model --env-file .env.waypoint
uv run lucid-modal deploy --env-file .env.waypoint
```

The required example-owned env is now the model/runtime config itself:

```bash
MODAL_APP_ENTRYPOINT=waypoint_modal_example.modal_app
MODAL_APP_NAME=lucid-waypoint-worker
MODAL_GPU=RTX-PRO-6000
MODAL_MODEL_VOLUME=lucid-waypoint-models
MODAL_HF_CACHE_VOLUME=lucid-hf-cache
MODAL_STARTUP_TIMEOUT_SECS=2400
MODAL_COMPILER_CACHE_ROOT=/cache/huggingface/compiler/waypoint/rtx-pro-6000
WAYPOINT_MODEL_SOURCE=/models/Waypoint-1.1-Small
WAYPOINT_AE_SOURCE=/models/owl_vae_f16_c16_distill_v0_nogan
WAYPOINT_PROMPT_ENCODER_SOURCE=/models/google-umt5-xl
```

The Waypoint model card recommends an RTX 5090 for roughly `20-30 FPS` or an RTX 6000 Pro Blackwell for roughly `35 FPS` when running locally. This example defaults to `MODAL_GPU=RTX-PRO-6000`, hardcodes the `640x360@20` output contract in the model, and keeps compiler caches under a GPU-specific directory so RTX PRO 6000 containers reuse their own `torch.compile` artifacts instead of mixing them with another accelerator type. If you want a cheaper starting point, try `L40S` first and only move up if startup time or frame rate is not acceptable.

`world_engine` compiles and autotunes kernels during load before the worker becomes interactive. After a successful warmup, the example writes a cache marker under `MODAL_COMPILER_CACHE_ROOT` and commits the mounted cache volume immediately, so later containers can skip warmup when that marker matches the current runtime signature (GPU type, torch version, world-engine revision, model sources, and frame size). If the marker is missing or stale, the worker warms again and refreshes the cached compile. The first successful interactive frame still keeps the existing commit fallback, and session teardown commits again so cache updates are not lost.

For debugging a `device-side assert`, set `CUDA_LAUNCH_BLOCKING=1` in the env file, redeploy, and rerun the session. That forces CUDA failures to surface at the kernel that triggered them instead of later during teardown.

## Run through the demo

Start the local coordinator with the deployed Modal dispatch URL and your LiveKit credentials, then run the demo app:

```bash
cargo run -p coordinator
cd apps/demo && bun run dev
```

The demo renders the generated manifest, subscribes to `main_video`, and registers only the keyboard and mouse listeners declared by the Waypoint input bindings.
