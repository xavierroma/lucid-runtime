# Coordinator (Rust)

CPU control-plane service for request-based session lifecycle + Modal dispatch.

## Public API

- `GET /healthz`
- `POST /sessions` (async; returns `202 Accepted`)
- `GET /sessions/{session_id}`
- `POST /sessions/{session_id}:end`

Public API uses bearer auth:

- `Authorization: Bearer <API_KEY>`

## Internal runtime callback API

- `POST /internal/sessions/{session_id}/ready`
- `POST /internal/sessions/{session_id}/running`
- `POST /internal/sessions/{session_id}/paused`
- `POST /internal/sessions/{session_id}/heartbeat`
- `POST /internal/sessions/{session_id}/ended`

Internal API uses bearer auth:

- `Authorization: Bearer <WORKER_INTERNAL_TOKEN>`

## Runtime behavior

- In-memory session state only (`STARTING -> READY -> RUNNING <-> PAUSED -> CANCELING -> ENDED|FAILED`)
- Multiple concurrent sessions are tracked in memory
- Session create dispatches to Modal and stores `function_call_id`
- Workers mark sessions `READY` once they have joined LiveKit and are ready for control
- Sessions stay `READY` until a client sends `resume`
- Running sessions can move to `PAUSED` and later `resume` without losing model state
- Public end requests are idempotent and move sessions into `CANCELING`
- Background reconciliation polls Modal status, consumes worker heartbeats, and escalates cancel requests before failing stuck sessions

## Configuration

Required environment variables:

- `API_KEY`
- `WORKER_INTERNAL_TOKEN`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `MODAL_DISPATCH_BASE_URL`
- `MODAL_DISPATCH_TOKEN`
- `COORDINATOR_CALLBACK_BASE_URL`

Optional environment variables:

- `WM_MODEL_NAME` (default: `yume`; should match the deployed Modal worker model)
- `COORDINATOR_BIND_ADDR` (default: `0.0.0.0:8080`)
- `WORKER_ID` (default: `wm-worker-1`)
- `SESSION_STARTUP_TIMEOUT_SECS` (default: `120`)
- `SESSION_MAX_DURATION_SECS` (default: `3600`)
- `SESSION_CANCEL_GRACE_SECS` (default: `30`)
- `WORKER_HEARTBEAT_TIMEOUT_SECS` (default: `15`)

`POST /sessions` also accepts an optional JSON body:

```json
{ "model_name": "waypoint" }
```

If `model_name` is provided, it must match `WM_MODEL_NAME`. The coordinator is still a
single-model deployment, so this is a guardrail for clients such as the demo app rather than
multi-model routing. `POST /sessions` no longer returns `409` for runtime capacity; `409` is
reserved for requested model mismatches.

## Local run

```bash
cargo run --manifest-path apps/coordinator/Cargo.toml
```

## Test

```bash
cargo test --manifest-path apps/coordinator/Cargo.toml
```
