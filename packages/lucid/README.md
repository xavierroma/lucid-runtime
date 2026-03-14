# Lucid

Single package for Lucid model authoring and runtime adapters.

Core authoring API:

- `LucidModel`
- `LucidSession`
- `@input`
- `publish.*`
- `SessionContext`
- manifest generation and input/output validation

Subpackages:

- `lucid.core`: model/session/spec/runtime internals
- `lucid.livekit`: realtime runtime host and LiveKit transport
- `lucid.controlplane`: worker lifecycle reporting clients
- `lucid.modal`: Modal adapter and CLI

The installable distribution name is `lucid-runtime`; model code imports `lucid`.
