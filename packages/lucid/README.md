# Lucid

Python runtime library for declaring realtime world models with:

- `@lucid.model`
- `@lucid.action`
- `lucid.publish.*`
- `SessionContext`
- manifest generation and action/output validation
- `lucid.worker`
- `lucid.research_server`
- `lucid.livekit`
- `lucid.modal`

The library lives here. Example model ports live under `examples/`.

Runtime-managed sessions follow `STARTING -> READY -> RUNNING`.
Workers enter `READY` after allocation/load/connect, and only begin generation after
`lucid.runtime.start`.

The installable distribution name is `lucid-runtime`; model code imports `lucid`.
