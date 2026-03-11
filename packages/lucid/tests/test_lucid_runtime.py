from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import numpy as np
import pytest

from lucid import LucidRuntime, SessionContext, publish
from lucid.capabilities import manifest as load_manifest
from lucid.config import RuntimeConfig
import lucid.discovery as discovery
from yume_modal_example.config import YumeRuntimeConfig


@pytest.fixture
def worker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.invalid")
    monkeypatch.setenv("WM_ENGINE", "fake")
    monkeypatch.setenv("WM_LIVEKIT_MODE", "fake")
    monkeypatch.setenv("YUME_MODEL_DIR", "/tmp/yume-model")
    monkeypatch.setenv("YUME_CHUNK_FRAMES", "1")
    monkeypatch.setenv("WM_MAX_QUEUE_FRAMES", "8")
    monkeypatch.setenv("WM_FRAME_WIDTH", "64")
    monkeypatch.setenv("WM_FRAME_HEIGHT", "64")


@pytest.mark.asyncio
async def test_manifest_includes_model_actions_outputs(worker_env: None) -> None:
    manifest = load_manifest()

    assert manifest["model"]["name"] == "yume"
    assert any(action["name"] == "set_prompt" for action in manifest["actions"])
    assert any(action["name"] == "lucid.runtime.start" for action in manifest["actions"])
    assert any(action["name"] == "lucid.runtime.pause" for action in manifest["actions"])
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
async def test_runtime_dispatches_state_and_command_actions(worker_env: None) -> None:
    runtime = LucidRuntime.load_selected(
        runtime_config=RuntimeConfig.from_env(),
        logger=logging.getLogger("tests.lucid_runtime"),
    )
    session_ctx = runtime.create_session_context(
        session_id="s1",
        room_name="wm-s1",
        publish_fn=_noop_publish,
    )

    await runtime.dispatch_action(session_ctx, "set_prompt", {"prompt": "new prompt"})
    await runtime.dispatch_action(session_ctx, "lucid.runtime.pause", {})
    await runtime.dispatch_action(session_ctx, "lucid.runtime.resume", {})
    await runtime.dispatch_action(session_ctx, "lucid.runtime.start", {})
    await runtime.dispatch_action(session_ctx, "lucid.runtime.pause", {})
    await runtime.dispatch_action(session_ctx, "lucid.runtime.resume", {})
    await runtime.dispatch_action(
        session_ctx,
        "lucid.runtime.set_output_enabled",
        {"output": "main_video", "enabled": False},
    )
    await runtime.dispatch_action(
        session_ctx,
        "lucid.runtime.set_output_rate",
        {"output": "main_video", "max_rate_hz": 4},
    )

    assert session_ctx.state.set_prompt.prompt == "new prompt"
    assert session_ctx.started is True
    assert session_ctx.paused is False
    assert session_ctx._output_enabled["main_video"] is False
    assert session_ctx._output_rate_hz["main_video"] == 4


@pytest.mark.asyncio
async def test_pause_and_resume_are_noops_before_start(worker_env: None) -> None:
    runtime = LucidRuntime.load_selected(
        runtime_config=RuntimeConfig.from_env(),
        logger=logging.getLogger("tests.lucid_runtime"),
    )
    session_ctx = runtime.create_session_context(
        session_id="s2",
        room_name="wm-s2",
        publish_fn=_noop_publish,
    )

    await runtime.dispatch_action(session_ctx, "lucid.runtime.pause", {})
    assert session_ctx.started is False
    assert session_ctx.paused is False

    await runtime.dispatch_action(session_ctx, "lucid.runtime.resume", {})
    assert session_ctx.started is False
    assert session_ctx.paused is False


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


def test_generated_artifacts_are_fresh(worker_env: None) -> None:
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "generate_lucid_artifacts.py"
    spec = importlib.util.spec_from_file_location("generate_lucid_artifacts", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    manifest = load_manifest()
    manifest_path = Path(__file__).resolve().parents[3] / "packages" / "contracts" / "generated" / "lucid_manifest.json"
    waypoint_manifest_path = Path(__file__).resolve().parents[3] / "packages" / "contracts" / "generated" / "lucid_manifest.waypoint.json"
    ts_path = Path(__file__).resolve().parents[3] / "apps" / "demo" / "src" / "lib" / "generated" / "lucid.ts"

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
    assert ts_path.read_text(encoding="utf-8") == module.render_ts(manifest)


def test_model_loader_can_import_example_model_module(
    worker_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WM_MODEL_MODULE", "yume_modal_example.model")
    discovery._loaded_modules.clear()

    assert discovery.ensure_model_module_loaded() == "yume_modal_example.model"
    assert discovery.configured_model_packages() == ("yume_modal_example",)


def test_build_model_runtime_config_uses_model_module_builder(
    worker_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WM_MODEL_MODULE", "yume_modal_example.model")
    discovery._loaded_modules.clear()

    runtime_config = discovery.build_model_runtime_config(RuntimeConfig.from_env())

    assert isinstance(runtime_config, YumeRuntimeConfig)


async def _noop_publish(_name: str, _payload: object, _ts_ms: int | None) -> None:
    return None
