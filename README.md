# Lucid Runtime

A realtime world-model runtime with:

- Lucid library in [`packages/lucid`](packages/lucid)
- Lucid modal deploy helpers in [`packages/lucid-modal`](packages/lucid-modal)
- Lucid dev in [`packages/lucid-dev`](packages/lucid-dev)

The intended porting model is:

- Lucid code declares the model, typed actions, outputs, and session loop.
- Your model code stays your model code: Torch modules, sampling code, checkpoint loading, and deployment image details remain outside the Lucid library.
- The demo and coordinator consume the Lucid manifest and do not need specific UI logic.


The examples are documentation of what a minimal Lucid port looks like:
- Yume example in [`examples/yume_modal`](examples/yume_modal)
- Waypoint example in [`examples/waypoint_modal`](examples/waypoint_modal)
- Helios example in [`examples/helios_modal`](examples/helios_modal)

The installable Python distribution is `lucid-runtime`; the import package is `lucid`.


