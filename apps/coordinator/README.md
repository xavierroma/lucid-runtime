# Coordinator (Rust)

CPU-only control-plane service.

## Planned responsibilities

- Expose session lifecycle API (`/v1/sessions`, `:end`, `GET session`)
- Mint LiveKit tokens for client and worker
- Track in-memory session + worker state (`IDLE/BUSY`, `CREATED/ASSIGNED/RUNNING/ENDED`)
- Enforce single active session

## Local run

```bash
cargo run --manifest-path apps/coordinator/Cargo.toml
```
