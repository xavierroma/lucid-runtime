# Modal Deployment

Request-based GPU execution is deployed via a model-specific Modal entrypoint such as:

- `examples/yume_modal/src/yume_modal_example/modal_app.py`
- `examples/waypoint_modal/src/waypoint_modal_example/modal_app.py`

See the example READMEs for end-to-end setup.

## Scripted workflow

All helper scripts live in `deploy/modal` and source `deploy/modal/.env` by default.

```bash
cp deploy/modal/.env.example deploy/modal/.env
```

Common scripts:

- `deploy/modal/create-volumes.sh`
- `deploy/modal/download-model.sh -- --repo-id stdstu123/Yume-5B-720P`
- `deploy/modal/flash-attn-smoke.sh` - validates CUDA, Torch, and FlashAttention inside the Modal GPU image before you try a full Yume session
- `deploy/modal/deploy.sh`
- `deploy/modal/serve.sh`
- `deploy/modal/logs.sh -- --timestamps`
- `deploy/modal/stop.sh`

The helper scripts now support a first-class model preset:

```bash
deploy/modal/deploy.sh --model yume
deploy/modal/deploy.sh --model waypoint
```

`--model` and `MODAL_MODEL` currently support:

- `yume`
- `waypoint`

The preset fills in the example project path, Modal entrypoint, app name, model volume, and
Lucid runtime defaults (`WM_MODEL_NAME`, `WM_MODEL_MODULE`, `WM_ENGINE`). You can still
override individual values in your env file.

If you want to wire a custom app manually, these env vars take precedence over the preset:

- `MODAL_PROJECT_PATH`
- `MODAL_PROJECT_SRC`
- `MODAL_APP_ENTRYPOINT`
- `MODAL_APP_NAME`
- `MODAL_MODEL_VOLUME`
- `MODAL_HF_CACHE_VOLUME`
- `WM_MODEL_NAME`
- `WM_MODEL_MODULE`
- `WM_ENGINE`

To use a different env file:

```bash
deploy/modal/deploy.sh --model waypoint --env-file /path/to/modal.env
```

## 1) Create Modal resources

```bash
deploy/modal/create-volumes.sh --model yume
deploy/modal/create-volumes.sh --model waypoint
```

## 2) Upload model files to the model volume

```bash
deploy/modal/download-model.sh --model yume
deploy/modal/download-model.sh --model waypoint
deploy/modal/download-model.sh --model waypoint -- --repo-id <huggingface-model-id>
```

## 3) Configure secrets/env for Modal app

Minimum required:

- `MODAL_DISPATCH_TOKEN` (shared with coordinator `MODAL_DISPATCH_TOKEN`)
- `LIVEKIT_URL`
- `WM_LIVEKIT_MODE=real`
- `HF_HOME=/cache/huggingface`

Example-specific env vars such as `YUME_MODEL_DIR` or `WAYPOINT_MODEL_SOURCE` depend on the
selected model preset. For the built-in presets, `WM_MODEL_NAME`, `WM_MODEL_MODULE`, and
`WM_ENGINE` are filled automatically.

Optional:

- `MODAL_GPU` to override the example default GPU
- `MODAL_APP_NAME` to override the preset app name
- `MODAL_MODEL_VOLUME` to override the preset model volume
- `MODAL_HF_CACHE_VOLUME` (default `lucid-hf-cache`)
- `MODAL_STARTUP_TIMEOUT_SECS` to raise Modal's container startup deadline for models that do heavy eager initialization

## 4) Deploy

```bash
deploy/modal/deploy.sh --model yume
deploy/modal/deploy.sh --model waypoint
```

Modal serves:

- `POST /launch`
- `POST /cancel`

Both routes require `Authorization: Bearer <MODAL_DISPATCH_TOKEN>`.

## Yume-only smoke test

`deploy/modal/flash-attn-smoke.sh` is still Yume-specific because the Waypoint image does not
install FlashAttention:

```bash
deploy/modal/flash-attn-smoke.sh --model yume
```

## 5) Wire coordinator

Set coordinator env:

- `MODAL_DISPATCH_BASE_URL=<modal-dispatch-base-url>`
- `MODAL_DISPATCH_TOKEN=<same-token>`
- `COORDINATOR_CALLBACK_BASE_URL=<public coordinator base URL>`
- `WM_MODEL_NAME=<same model preset, e.g. waypoint>`

## 6) Smoke test

1. `POST /sessions` should return `202`.
2. The first session after a fresh deploy can stay in `STARTING` while the worker loads the model
   onto the GPU.
3. `GET /sessions/{id}` should move `STARTING -> RUNNING -> ENDED` once the load completes.
4. `POST /sessions/{id}:end` should return `200` and trigger best-effort cancel.
