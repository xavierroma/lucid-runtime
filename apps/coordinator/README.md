# Coordinator (Rust)

CPU-only control-plane service for V1 session lifecycle + worker assignment.

## Public API

- `GET /healthz`
- `POST /v1/sessions`
- `GET /v1/sessions/{session_id}`
- `POST /v1/sessions/{session_id}:end`

Public API uses bearer auth:

- `Authorization: Bearer <API_KEY>`

## Internal worker API

- `POST /internal/v1/worker/register`
- `POST /internal/v1/worker/heartbeat`
- `GET /internal/v1/worker/assignment?worker_id=...`
- `POST /internal/v1/sessions/{session_id}/running`
- `POST /internal/v1/sessions/{session_id}/ended`

Internal API uses bearer auth:

- `Authorization: Bearer <WORKER_INTERNAL_TOKEN>`

## Runtime behavior

- In-memory session state only (`CREATED -> ASSIGNED -> RUNNING -> ENDED`)
- Single active session invariant (`ASSIGNED`/`RUNNING` only one at a time)
- Worker liveness tracked with heartbeat TTL
- Missing heartbeat auto-ends active session with `WORKER_DISCONNECTED`

## Configuration

Required environment variables:

- `API_KEY`
- `WORKER_INTERNAL_TOKEN`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`

Optional environment variables:

- `COORDINATOR_BIND_ADDR` (default: `0.0.0.0:8080`)
- `WORKER_ID` (default: `wm-worker-1`)
- `HEARTBEAT_TTL_SECS` (default: `15`)

## Local run

```bash
cargo run --manifest-path apps/coordinator/Cargo.toml
```

## Test

```bash
cargo test --manifest-path apps/coordinator/Cargo.toml
```
