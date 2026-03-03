# Worker (Python)

Single-process GPU worker runtime.

## Planned responsibilities

- Register to coordinator and wait for assignment
- Join assigned LiveKit room as bot participant
- Receive action commands over data messages
- Run inference loop and publish `main_video`
- Cleanup and return to IDLE on end/error

## Local run

```bash
uv sync --project apps/worker
uv run --project apps/worker wm-worker --worker-id wm-worker-1
```
