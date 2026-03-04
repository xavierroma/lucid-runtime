# Worker (Python)

Single-process GPU worker runtime.

## Responsibilities

- Register to coordinator and wait for assignment
- Join assigned LiveKit room as bot participant
- Receive action commands over data messages
- Run inference loop and publish `main_video`
- Cleanup and return to IDLE on end/error

## Required environment

- `COORDINATOR_BASE_URL`
- `WORKER_INTERNAL_TOKEN`
- `LIVEKIT_URL`
- `YUME_MODEL_DIR`

## Optional environment

- `WORKER_ID` (default `wm-worker-1`)
- `WM_ENGINE` (`fake` or `yume`, default `fake`)
- `WM_LIVEKIT_MODE` (`fake` or `real`, default `fake`)
- `WORKER_HEALTH_PORT` (default `8090`)
- `WM_FRAME_WIDTH` (default `1280`)
- `WM_FRAME_HEIGHT` (default `720`)
- `WM_TARGET_FPS` (default `16`)
- `YUME_CHUNK_FRAMES` (default `8`)
- `YUME_MAX_QUEUE_FRAMES` (default `32`)

## Local run (fake mode)

```bash
uv sync --project apps/worker
COORDINATOR_BASE_URL=http://localhost:8080 \
WORKER_INTERNAL_TOKEN=replace-me \
LIVEKIT_URL=wss://example.livekit.cloud \
YUME_MODEL_DIR=/tmp/yume-model \
WM_ENGINE=fake WM_LIVEKIT_MODE=fake \
uv run --project apps/worker wm-worker --worker-id wm-worker-1
```

## Test

```bash
uv run --project apps/worker pytest apps/worker/tests
```
