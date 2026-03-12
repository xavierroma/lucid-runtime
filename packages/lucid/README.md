# Lucid

Core runtime library for declaring realtime world models with:

- `LucidModel`
- `LucidSession`
- `@input`
- `lucid.publish.*`
- `SessionContext`
- manifest generation and input/output validation
- `SessionRunner`
- protocol, LiveKit, and coordinator glue

Provider adapters and local dev servers live outside of this package.

The installable distribution name is `lucid-runtime`; model code imports `lucid`.
