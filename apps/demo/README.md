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
- Each environment currently stores a name plus a text prompt.
- Return to the main console and choose one saved environment before starting a session.

## Session flow

- Press power to allocate the worker and join the room.
- Wait for `READY`, then choose the environment you want to launch.
- Press `Start`; the demo sends `set_prompt` with the selected environment prompt first, then
  sends `lucid.runtime.start`.
- When Waypoint is active, the demo mounts a control deck that publishes persistent `set_buttons`
  state plus transient `mouse_move` and `scroll` actions after the room connects.

## Build

```bash
bun run build
```
