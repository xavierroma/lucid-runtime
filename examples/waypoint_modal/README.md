# Waypoint on Modal

This example shows a Lucid port of [Overworld/Waypoint-1.1-Small](https://huggingface.co/Overworld/Waypoint-1.1-Small) deployed on Modal.

- The reusable runtime and hosting code lives in [`packages/lucid`](../../packages/lucid).
- The Waypoint-specific port lives in [`src/waypoint_modal_example`](src/waypoint_modal_example).

## What is the Lucid port

The Lucid-specific code is intentionally small:

- [`model.py`](src/waypoint_modal_example/model.py) declares the model, publishes `main_video`, exposes prompt + control actions, and runs the session loop.
- [`config.py`](src/waypoint_modal_example/config.py) adapts generic Lucid host config into Waypoint-specific runtime config.
- [`modal_app.py`](src/waypoint_modal_example/modal_app.py) defines the Modal image and calls `lucid.modal.create_app(...)`.

That is the part you rewrite when you port a new model into Lucid.

## What is just the model

[`engine.py`](src/waypoint_modal_example/engine.py) is ordinary model-serving code. It wraps the official `world_engine.WorldEngine` API, keeps CUDA work on a dedicated thread, seeds the model with a starter frame, and converts generated frames into Lucid-ready `numpy.uint8` RGB frames.

## Current Waypoint Lucid model

The example exposes:

- `set_prompt`: persistent text conditioning
- `set_controls`: persistent held buttons plus mouse velocity and scroll input
- `main_video`: the generated RGB video stream

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

This example owns its own Modal deployment module at
[`modal_app.py`](src/waypoint_modal_example/modal_app.py).

Copy the example env file, then create Modal volumes, download the model, and deploy:

```bash
cp examples/waypoint_modal/modal.env.example deploy/modal/waypoint.env
deploy/modal/create-volumes.sh --env-file deploy/modal/waypoint.env
deploy/modal/download-model.sh --env-file deploy/modal/waypoint.env
deploy/modal/deploy.sh --env-file deploy/modal/waypoint.env
```

Required env for the deployed example:

```bash
MODAL_PROJECT_PATH=examples/waypoint_modal
MODAL_PROJECT_SRC=examples/waypoint_modal/src
MODAL_APP_ENTRYPOINT=examples/waypoint_modal/src/waypoint_modal_example/modal_app.py
MODAL_STARTUP_TIMEOUT_SECS=2400
MODAL_COMPILER_CACHE_ROOT=/cache/huggingface/compiler/waypoint/rtx-pro-6000
WM_MODEL_MODULE=waypoint_modal_example.model
WM_MODEL_NAME=waypoint
WM_ENGINE=waypoint
WAYPOINT_MODEL_SOURCE=/models/Waypoint-1.1-Small
WAYPOINT_WARMUP_ON_LOAD=0
```

The Waypoint model card recommends an RTX 5090 for roughly `20-30 FPS` or an RTX 6000 Pro Blackwell for roughly `35 FPS` when running locally. This example defaults to `MODAL_GPU=RTX-PRO-6000` and keeps the compiler caches under a GPU-specific directory so RTX PRO 6000 containers reuse their own `torch.compile` artifacts instead of mixing them with another accelerator type. If you want a cheaper starting point, try `L40S` first and only move up if startup time or frame rate is not acceptable.

`world_engine` compiles and autotunes kernels on the first generated frame. In this example, `WAYPOINT_WARMUP_ON_LOAD=0` skips that first-frame warmup during Modal container startup so the worker can come up inside Modal's startup deadline. The tradeoff is that the first generated frame of a fresh container will still pay the compile/autotune cost. The compiled Triton, Inductor, and CUDA driver caches are written under `MODAL_COMPILER_CACHE_ROOT` on the mounted cache volume, and the session teardown now commits that volume so later RTX PRO 6000 containers can reuse the first successful compile. The example timeout now defaults to 2400 seconds (40 minutes); if you want eager warm containers instead, set `WAYPOINT_WARMUP_ON_LOAD=1` and keep `MODAL_STARTUP_TIMEOUT_SECS` high enough to cover the one-time compile.

For debugging a `device-side assert`, set `CUDA_LAUNCH_BLOCKING=1` in the Modal env file, redeploy, and rerun the session. That forces CUDA failures to surface at the kernel that triggered them instead of later during `end_session()`.

## Run through the demo

Start the local coordinator with the deployed Modal dispatch URL and your LiveKit credentials,
then run the demo app:

```bash
cargo run -p coordinator
cd apps/demo && bun run dev
```

The demo renders the Lucid manifest and exposes the generated `main_video` output. For richer controls, send `set_controls` actions from your own client or a keyboard/mouse adapter.

Example `set_controls` control message:

```json
{
  "type": "action",
  "seq": 1,
  "ts_ms": 1741315200000,
  "session_id": "3acb0b65-7b3c-4ebb-8e98-9e18dbf7403f",
  "payload": {
    "name": "set_controls",
    "args": {
      "forward": true,
      "mouse_x": 0.18,
      "mouse_y": -0.04,
      "scroll_wheel": 0
    }
  }
}
```
