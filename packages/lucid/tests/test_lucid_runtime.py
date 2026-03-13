from __future__ import annotations

import asyncio
import importlib.util
import logging
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from lucid import LucidRuntime, SessionContext, publish
from lucid.capabilities import manifest as load_manifest
from lucid.config import RuntimeConfig
import lucid.discovery as discovery
from yume_modal_example.model import YumeLucidModel


def _runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        livekit_url="wss://example.livekit.invalid",
        frame_width=64,
        frame_height=64,
        max_queue_frames=8,
        livekit_mode="fake",
    )


@pytest.mark.asyncio
async def test_manifest_includes_model_inputs_outputs() -> None:
    manifest = load_manifest("yume_modal_example.model:YumeLucidModel")

    assert manifest["model"]["name"] == "yume"
    assert manifest["inputs"] == [
        {
            "name": "set_prompt",
            "description": "Update the scene prompt used by Yume.",
            "args_schema": {
                "additionalProperties": False,
                "properties": {
                    "prompt": {
                        "minLength": 1,
                        "title": "Prompt",
                        "type": "string",
                    }
                },
                "required": ["prompt"],
                "title": "SetPromptInputArgs",
                "type": "object",
            },
        }
    ]
    assert manifest["outputs"] == [
        {
            "name": "main_video",
            "kind": "video",
            "width": 1280,
            "height": 720,
            "fps": 2,
            "pixel_format": "rgb24",
        }
    ]


@pytest.mark.asyncio
async def test_runtime_dispatches_inputs_to_session() -> None:
    runtime = LucidRuntime.load_model(
        runtime_config=_runtime_config(),
        logger=logging.getLogger("tests.lucid_runtime"),
        model=YumeLucidModel,
    )
    session_ctx = runtime.create_session_context(
        session_id="s1",
        room_name="wm-s1",
        publish_fn=_noop_publish,
    )
    session = cast(object, runtime._sessions["s1"])

    await runtime.dispatch_input(session_ctx, "set_prompt", {"prompt": "new prompt"})

    assert getattr(session, "prompt") == "new prompt"


@pytest.mark.asyncio
async def test_runtime_rejects_unknown_input() -> None:
    runtime = LucidRuntime.load_model(
        runtime_config=_runtime_config(),
        logger=logging.getLogger("tests.lucid_runtime"),
        model="yume_modal_example.model:YumeLucidModel",
    )
    session_ctx = runtime.create_session_context(
        session_id="s2",
        room_name="wm-s2",
        publish_fn=_noop_publish,
    )

    with pytest.raises(Exception, match="unknown input"):
        await runtime.dispatch_input(session_ctx, "missing", {})


@pytest.mark.asyncio
async def test_session_context_validates_video_json_and_bytes_outputs() -> None:
    seen: list[tuple[str, object]] = []

    async def publish_fn(name: str, payload: object, ts_ms: int | None) -> None:
        _ = ts_ms
        seen.append((name, payload))

    ctx = SessionContext(
        session_id="s1",
        room_name="wm-s1",
        outputs=(
            publish.video(name="video", width=4, height=4, fps=8),
            publish.json(name="state"),
            publish.bytes(name="blob"),
        ),
        publish_fn=publish_fn,
        logger=logging.getLogger("tests.lucid_runtime"),
    )

    await ctx.publish("video", np.zeros((4, 4, 3), dtype=np.uint8))
    await ctx.publish("state", {"ok": True})
    await ctx.publish("blob", b"abc")

    assert seen[0][0] == "video"
    assert seen[1] == ("state", b'{"ok":true}')
    assert seen[2] == ("blob", b"abc")


@pytest.mark.asyncio
async def test_session_context_waits_until_resumed() -> None:
    ctx = SessionContext(
        session_id="s1",
        room_name="wm-s1",
        outputs=(),
        publish_fn=_noop_publish,
        logger=logging.getLogger("tests.lucid_runtime"),
    )
    assert ctx.pause() is True

    waiter = asyncio.create_task(ctx.wait_if_paused())
    await asyncio.sleep(0.01)
    assert waiter.done() is False

    assert ctx.resume() is True
    await asyncio.wait_for(waiter, timeout=1.0)


def test_generated_artifacts_are_fresh() -> None:
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "generate_lucid_artifacts.py"
    spec = importlib.util.spec_from_file_location("generate_lucid_artifacts", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    manifest = load_manifest("yume_modal_example.model:YumeLucidModel")
    manifest_path = Path(__file__).resolve().parents[3] / "packages" / "contracts" / "generated" / "lucid_manifest.json"
    waypoint_manifest_path = Path(__file__).resolve().parents[3] / "packages" / "contracts" / "generated" / "lucid_manifest.waypoint.json"
    helios_manifest_path = Path(__file__).resolve().parents[3] / "packages" / "contracts" / "generated" / "lucid_manifest.helios.json"
    ts_path = Path(__file__).resolve().parents[3] / "apps" / "demo" / "src" / "lib" / "generated" / "lucid.ts"
    waypoint_ts_path = Path(__file__).resolve().parents[3] / "apps" / "demo" / "src" / "lib" / "generated" / "lucid.waypoint.ts"
    helios_ts_path = Path(__file__).resolve().parents[3] / "apps" / "demo" / "src" / "lib" / "generated" / "lucid.helios.ts"

    assert manifest_path.read_text(encoding="utf-8") == (
        module.json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    assert waypoint_manifest_path.read_text(encoding="utf-8") == (
        module.json.dumps(
            module._load_manifest(
                module_name="waypoint_modal_example.model",
                extra_path=Path(__file__).resolve().parents[3] / "examples" / "waypoint_modal" / "src",
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    assert helios_manifest_path.read_text(encoding="utf-8") == (
        module.json.dumps(
            module._load_manifest(
                module_name="helios_modal_example.model",
                extra_path=Path(__file__).resolve().parents[3] / "examples" / "helios_modal" / "src",
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    assert ts_path.read_text(encoding="utf-8") == module.render_ts(manifest)
    assert waypoint_ts_path.read_text(encoding="utf-8") == module.render_ts(
        module._load_manifest(
            module_name="waypoint_modal_example.model",
            extra_path=Path(__file__).resolve().parents[3] / "examples" / "waypoint_modal" / "src",
        )
    )
    assert helios_ts_path.read_text(encoding="utf-8") == module.render_ts(
        module._load_manifest(
            module_name="helios_modal_example.model",
            extra_path=Path(__file__).resolve().parents[3] / "examples" / "helios_modal" / "src",
        )
    )


def test_model_loader_can_import_example_model_module(
) -> None:
    discovery._loaded_modules.clear()
    discovery._loaded_model_classes.clear()

    assert discovery.load_model_module("yume_modal_example.model:YumeLucidModel").__name__ == "yume_modal_example.model"
    assert (
        discovery.resolve_model_class("yume_modal_example.model:YumeLucidModel").__name__
        == "YumeLucidModel"
    )


def test_model_loader_accepts_direct_model_class() -> None:
    discovery._loaded_modules.clear()
    discovery._loaded_model_classes.clear()

    assert discovery.resolve_model_class(YumeLucidModel) is YumeLucidModel


async def _noop_publish(_name: str, _payload: object, _ts_ms: int | None) -> None:
    return None
