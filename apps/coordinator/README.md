# Coordinator (Rust)

CPU control-plane service for request-based V1 session lifecycle + Modal dispatch.

## Public API

- `GET /healthz`
- `POST /v1/sessions` (async; returns `202 Accepted`)
- `GET /v1/sessions/{session_id}`
- `POST /v1/sessions/{session_id}:end`

Public API uses bearer auth:

- `Authorization: Bearer <API_KEY>`

## Internal runtime callback API

- `POST /internal/v1/sessions/{session_id}/running`
- `POST /internal/v1/sessions/{session_id}/ended`

Internal API uses bearer auth:

- `Authorization: Bearer <WORKER_INTERNAL_TOKEN>`

## Runtime behavior

- In-memory session state only (`CREATED -> RUNNING -> ENDED`)
- Single active session invariant
- Session create dispatches to Modal and stores `function_call_id`
- Public end requests are idempotent and trigger best-effort Modal cancel
- Background reconciliation auto-ends stuck sessions (`STARTUP_TIMEOUT`, `SESSION_TIMEOUT`, `CANCEL_TIMEOUT`)

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

- `COORDINATOR_BIND_ADDR` (default: `0.0.0.0:8080`)
- `WORKER_ID` (default: `wm-worker-1`)
- `SESSION_STARTUP_TIMEOUT_SECS` (default: `120`)
- `SESSION_MAX_DURATION_SECS` (default: `3600`)
- `SESSION_CANCEL_GRACE_SECS` (default: `30`)

## Local run

```bash
cargo run --manifest-path apps/coordinator/Cargo.toml
```

## Test

```bash
cargo test --manifest-path apps/coordinator/Cargo.toml
```
