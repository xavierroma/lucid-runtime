# Demo App

React + Vite frontend for creating a coordinator session, joining the returned LiveKit room,
authoring saved world environments, and sending Lucid control actions.

## Environment

Copy [`.env.example`](/Users/xavierroma/projects/lucid-runtime/apps/demo/.env.example) to
`.env` and set:

- `VITE_LIVEKIT_URL`: same LiveKit URL used by the coordinator and Modal worker
- `VITE_DEFAULT_MODEL`: `yume` or `waypoint`; this should match the deployed coordinator and
  Modal worker

For local development, prefer the Vite proxy:

- `VITE_COORDINATOR_PROXY_TARGET=http://127.0.0.1:8080`
- `COORDINATOR_API_KEY=<same API_KEY used by the coordinator>`

For a deployed demo, use direct mode only if the coordinator is exposed behind the same origin
or a gateway that handles CORS:

- `VITE_COORDINATOR_BASE_URL=https://coordinator.example.com`
- `VITE_COORDINATOR_API_KEY=<same API_KEY used by the coordinator>`

## Run

```bash
bun install
bun run dev
```

## Environment flow

- Open `/environments` in the demo to create saved world environments backed by browser
  `localStorage`.
- Each environment stores a title, a prompt, and a first-frame seed image.
- Return to the main console and choose one saved environment before starting a session.

## Session flow

- Choose a saved environment before powering the session on.
- Press power to allocate the worker and join the room.
- When the worker reaches `READY`, the demo sends `set_prompt` with the selected environment
  prompt, sends `set_initial_frame` when the manifest exposes that image input, and only then
  sends `resume`.
- The worker stays in `READY` until that resume signal arrives, then transitions to `RUNNING`.
- While `RUNNING`, the toolbar pause control sends `pause`; while `PAUSED`, it sends `resume`.
- While `PAUSED`, prompt updates plus hold/axis inputs still flow, while press/pointer/wheel
  inputs are dropped.
- When Waypoint is active, the demo mounts its control deck after the session becomes live and
  keeps transient look/scroll controls disabled while paused.

## Build

```bash
bun run build
```
