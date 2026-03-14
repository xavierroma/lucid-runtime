# Coordinator (Rust)

CPU control-plane service for request-based session lifecycle + Modal dispatch.

## Public API

- `GET /healthz`
- `GET /models`
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
- Session create dispatches to the selected model backend and stores `function_call_id`
- Workers mark sessions `READY` once they have joined LiveKit and are ready for control
- Sessions stay `READY` until a client sends `resume`
- Running sessions can move to `PAUSED` and later `resume` without losing model state
- Public end requests are idempotent and move sessions into `CANCELING`
- Background reconciliation polls the selected model backend, consumes worker heartbeats, and escalates cancel requests before failing stuck sessions

## Configuration

Required environment variables:

- `API_KEY`
- `WORKER_INTERNAL_TOKEN`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `COORDINATOR_CALLBACK_BASE_URL`
- `COORDINATOR_MODELS_FILE`

Optional environment variables:

- `COORDINATOR_BIND_ADDR` (default: `0.0.0.0:8080`)

`COORDINATOR_MODELS_FILE` points at a JSON registry describing every supported model:

- `id`
- `display_name`
- `manifest_path`
- `backend`
- `timeouts`

Only `modal` backends are implemented in this version. The coordinator loads each manifest at
startup and precomputes the capabilities that will be returned for sessions of that model.

`POST /sessions` requires a JSON body:

```json
{ "model_name": "helios" }
```

`GET /models` returns the configured model ids and display names in registry order. `POST /sessions`
returns `400` when `model_name` is missing or unsupported.

## Local run

```bash
cargo run --manifest-path apps/coordinator/Cargo.toml
```

## Test

```bash
cargo test --manifest-path apps/coordinator/Cargo.toml
```
