# Contracts

Shared API contracts and generated Lucid artifacts used by all apps.

- `generated/lucid_manifest.json`: generated Yume Lucid model manifest embedded by the coordinator when `WM_MODEL_NAME=yume`.
- `generated/lucid_manifest.waypoint.json`: generated Waypoint Lucid model manifest embedded by the coordinator when `WM_MODEL_NAME=waypoint`.
- `generated/lucid_manifest.helios.json`: generated Helios Lucid model manifest embedded by the coordinator when `WM_MODEL_NAME=helios`.
- `../apps/demo/src/lib/generated/lucid.ts`: generated Yume action envelope TypeScript helpers for the demo.
- `../apps/demo/src/lib/generated/lucid.waypoint.ts`: generated Waypoint action envelope TypeScript helpers for the demo.
- `../apps/demo/src/lib/generated/lucid.helios.ts`: generated Helios action envelope TypeScript helpers for the demo.
- `openapi/session-api.yaml`: coordinator API contract draft.
- `openapi/worker-internal-api.yaml`: coordinator internal session callback API contract draft.
