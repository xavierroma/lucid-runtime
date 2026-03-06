# Modal Deployment

Request-based GPU execution is deployed via `wm_worker.modal_app`.

## 1) Create Modal resources

```bash
modal volume create lucid-yume-models
modal volume create lucid-hf-cache
```

## 2) Upload Yume model files to model volume

```bash
modal volume put lucid-yume-models /local/path/to/Yume-5B-720P /models/Yume-5B-720P
```

## 3) Configure secrets/env for Modal app

Minimum required:

- `MODAL_DISPATCH_TOKEN` (shared with coordinator `MODAL_DISPATCH_TOKEN`)
- `LIVEKIT_URL`
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
modal deploy apps/worker/src/wm_worker/modal_app.py
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

1. `POST /v1/sessions` should return `202`.
2. `GET /v1/sessions/{id}` should move `CREATED -> RUNNING -> ENDED`.
3. `POST /v1/sessions/{id}:end` should return `200` and trigger best-effort cancel.
