# Modal Deployment

Request-based GPU execution is deployed via
`examples/yume_modal/src/yume_modal_example/modal_app.py`.

See `examples/yume_modal/README.md` for a concrete end-to-end example using the Yume
Lucid model.

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

The helper scripts run the example project directly and prepend
`examples/yume_modal/src` to `PYTHONPATH`, so setting
`WM_MODEL_MODULE=yume_modal_example.model` is enough to deploy the example model.

To use a different env file:

```bash
deploy/modal/deploy.sh --env-file /path/to/modal.env
```

## 1) Create Modal resources

```bash
deploy/modal/create-volumes.sh
```

## 2) Upload Yume model files to model volume

```bash
deploy/modal/download-model.sh -- --repo-id stdstu123/Yume-5B-720P
```

## 3) Configure secrets/env for Modal app

Minimum required:

- `MODAL_DISPATCH_TOKEN` (shared with coordinator `MODAL_DISPATCH_TOKEN`)
- `LIVEKIT_URL`
- `WM_MODEL_NAME=yume`
- `WM_MODEL_MODULE=yume_modal_example.model`
- `YUME_MODEL_DIR=/models/Yume-5B-720P`
- `WM_ENGINE=yume`
- `WM_LIVEKIT_MODE=real`
- `HF_HOME=/cache/huggingface`

Optional:

- `MODAL_GPU` (default `A100`)
- `MODAL_APP_NAME` (default `lucid-runtime-worker`)
- `MODAL_MODEL_VOLUME` (default `lucid-yume-models`)
- `MODAL_HF_CACHE_VOLUME` (default `lucid-hf-cache`)

## 4) Deploy

```bash
deploy/modal/deploy.sh
```

Modal serves:

- `POST /launch`
- `POST /cancel`

Both routes require `Authorization: Bearer <MODAL_DISPATCH_TOKEN>`.

## 5) Wire coordinator

Set coordinator env:

- `MODAL_DISPATCH_BASE_URL=<modal-dispatch-base-url>`
- `MODAL_DISPATCH_TOKEN=<same-token>`
- `COORDINATOR_CALLBACK_BASE_URL=<public coordinator base URL>`

## 6) Smoke test

1. `POST /sessions` should return `202`.
2. The first session after a fresh deploy can stay in `STARTING` for roughly 3 minutes while the
   worker loads the Yume model onto the GPU.
3. `GET /sessions/{id}` should move `STARTING -> RUNNING -> ENDED` once the load completes.
4. `POST /sessions/{id}:end` should return `200` and trigger best-effort cancel.
