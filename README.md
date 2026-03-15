# Lucid Runtime

Lucid is a slim runtime for making realtime world models usable. The library gives you a small
model + session contract, a generated manifest, and runtime adapters for LiveKit and Modal. The
coordinator and demo are separate apps that consume that contract.

The installable Python distribution is `lucid-runtime`; model code imports `lucid`.

## Minimal Lucid port

```python
from lucid import LucidModel, LucidSession, SessionContext, hold, input, publish


class MySession(LucidSession["MyModel"]):
    def __init__(self, model: "MyModel", ctx: SessionContext) -> None:
        super().__init__(model, ctx)
        self.prompt = "..."
        self.active_inputs: set[str] = set()

    @input(description="Update the prompt.", paused=True)
    def set_prompt(self, prompt: str) -> None:
        self.prompt = prompt

    @input(binding=hold(keys=("KeyW",)))
    def move_fwd(self, pressed: bool) -> None:
        self.active_inputs.add("fwd") if pressed else self.active_inputs.discard("fwd")

    # move_l/move_bwd/move_r are the same as move_fwd.

    async def run(self) -> None:
        while self.ctx.running:
            frame = self.model.render_frame(self.prompt, frozenset(self.active_inputs))
            await self.ctx.publish("main_video", frame)


class MyModel(LucidModel):
    name = "my-model"
    session_cls = MySession
    outputs = (publish.video(name="main_video", width=640, height=360, fps=20),)

    def create_session(self, ctx: SessionContext) -> MySession:
        return MySession(self, ctx)
```

That is basically the port: define a session, give inputs real Python type annotations, declare
the outputs, and return the session from `create_session()`. Lucid derives the manifest, input
bindings, and runtime plumbing from that contract; your actual model loading and inference code
stays yours.

The four `hold(...)` bindings above wire `W`, `A`, `S`, and `D` to movement. The Waypoint example
uses the same pattern, then adds arrow-key aliases plus jump, sprint, crouch, mouse buttons,
relative look, and wheel input.

## Repo layout

### `packages/lucid`

The main Python package. It owns the authoring contract and the runtime adapters.

- `lucid`
  - Top-level authoring API: `LucidModel`, `LucidSession`, `@input`, `publish.*`,
    `SessionContext`, manifest generation, and input/output bindings.
- `lucid.core`
  - The small core runtime: model/session primitives, spec compilation, manifest generation,
    output validation, and `LucidRuntime`.
- `lucid.livekit`
  - The realtime host: LiveKit access tokens, capabilities payloads, and `SessionRunner` for
    running a Lucid session over WebRTC transport.
- `lucid.controlplane`
  - Optional worker lifecycle reporting clients. This is where coordinator callbacks live now;
    it is not part of the LiveKit transport layer.
- `lucid.modal`
  - Modal adapter and CLI: app wiring, dispatch API helpers, runtime config loading, and worker
    launch/cancel helpers.

### `packages/contracts`

Generated manifests consumed by the coordinator and demo. These are the serialized Lucid model
contracts, not handwritten app code.

## Apps

### `apps/coordinator`

Rust control-plane service. It does not run the model. It owns:

- model registry loading
- public session API
- worker dispatch
- session lifecycle state
- LiveKit token minting

It talks to Lucid workers through the internal callback API and uses model manifests to expose
capabilities to clients.

### `apps/demo`

React + Vite frontend for exercising a Lucid model end-to-end:

- lists models from the coordinator
- creates and ends sessions
- joins the returned LiveKit room
- sends Lucid control/input messages
- renders the model outputs

For local development it usually talks to `apps/coordinator` through the Vite proxy.

## Examples

The examples are the real documentation for porting a model into Lucid. Each one keeps the
model-specific code outside the library and uses Lucid only for the runtime contract.

### `examples/waypoint_modal`

Waypoint model port with a Modal worker, coordinator integration, and demo flow. Start here if
you want the most complete end-to-end example. See
[examples/waypoint_modal/README.md](/Users/xavierroma/projects/lucid-runtime/examples/waypoint_modal/README.md)
for the full local run sequence.

### `examples/helios_modal`

Helios distilled image model port. Good reference for a smaller Modal-backed model setup.

### `examples/yume_modal`

Yume model port with tests that focus on engine/runtime behavior.
