# Lucid Runtime

A realtime world-model runtime with:

- a Python-first Lucid library in [`packages/lucid`](packages/lucid)
- a concrete Yume-on-Modal example in [`examples/yume_modal`](examples/yume_modal)
- a concrete Waypoint-on-Modal example in [`examples/waypoint_modal`](examples/waypoint_modal)

The installable Python distribution is `lucid-runtime`; the import package is `lucid`.

The intended porting model is:

- Lucid code declares the model, typed actions, outputs, and session loop.
- Your model code stays your model code: Torch modules, sampling code, checkpoint loading, and deployment image details remain outside the Lucid library.
- The demo and coordinator consume the Lucid manifest and do not need Yume-specific UI logic.

The examples are the reference for that boundary and document what a minimal Lucid port looks like.
