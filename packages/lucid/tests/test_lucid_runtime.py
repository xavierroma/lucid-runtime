from __future__ import annotations

import asyncio

import numpy as np
import pytest

from lucid import (
    InputFile,
    LucidModel,
    LucidSession,
    SessionContext,
    build_model_definition,
    image_input,
    input,
    manifest,
    publish,
)
from lucid.core import ManifestGenerationError
from lucid.core.runtime import ActionDispatchError, LucidRuntime
from lucid.livekit import RuntimeConfig, capabilities
from yume_modal_example.model import YumeLucidModel


class _MinimalSession(LucidSession):
    async def run(self) -> None:
        self.ctx.running = False


class _MissingOutputsModel(LucidModel):
    session_cls = _MinimalSession

    def create_session(self, ctx: SessionContext) -> _MinimalSession:
        return _MinimalSession(self, ctx)


class _MissingSessionModel(LucidModel):
    outputs = (publish.bytes(name="state"),)

    def create_session(self, ctx: SessionContext) -> _MinimalSession:
        return _MinimalSession(self, ctx)


class _MissingAnnotationSession(LucidSession):
    @input(description="Broken input")
    def set_prompt(self, prompt) -> None:
        _ = prompt

    async def run(self) -> None:
        self.ctx.running = False


class _MissingAnnotationModel(LucidModel):
    session_cls = _MissingAnnotationSession
    outputs = (publish.bytes(name="state"),)

    def create_session(self, ctx: SessionContext) -> _MissingAnnotationSession:
        return _MissingAnnotationSession(self, ctx)


class _InputFileSession(LucidSession):
    def __init__(self, model, ctx: SessionContext) -> None:
        super().__init__(model, ctx)
        self.initial_frame: InputFile | None = None

    @input(description="Set an initial frame.", paused=True)
    def set_initial_frame(self, image: InputFile = image_input(size=(64, 64))) -> None:
        self.initial_frame = image

    async def run(self) -> None:
        self.ctx.running = False


class _InputFileModel(LucidModel):
    session_cls = _InputFileSession
    outputs = (publish.bytes(name="state"),)

    def create_session(self, ctx: SessionContext) -> _InputFileSession:
        return _InputFileSession(self, ctx)


@pytest.mark.asyncio
async def test_manifest_includes_model_inputs_outputs() -> None:
    generated = manifest(YumeLucidModel)

    assert generated["model"]["name"] == "yume"
    assert generated["inputs"] == [
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
    assert generated["outputs"] == [
        {
            "name": "main_video",
            "kind": "video",
            "width": 1280,
            "height": 720,
            "fps": 2,
            "pixel_format": "rgb24",
        }
    ]


def test_build_model_definition_compiles_once() -> None:
    first = build_model_definition(YumeLucidModel)
    second = build_model_definition(YumeLucidModel)

    assert first is second


def test_build_model_definition_requires_explicit_outputs() -> None:
    with pytest.raises(ManifestGenerationError, match="explicit outputs"):
        build_model_definition(_MissingOutputsModel)


def test_build_model_definition_requires_explicit_session_cls() -> None:
    with pytest.raises(ManifestGenerationError, match="explicit session_cls"):
        build_model_definition(_MissingSessionModel)


def test_build_model_definition_requires_typed_input_parameters() -> None:
    with pytest.raises(ManifestGenerationError, match="must have a type annotation"):
        build_model_definition(_MissingAnnotationModel)


def test_runtime_and_capabilities_share_output_binding_source() -> None:
    runtime = LucidRuntime.load_model(
        runtime_config=RuntimeConfig(livekit_url="wss://example.livekit.invalid"),
        logger=_logger(),
        model=YumeLucidModel,
    )

    assert runtime.output_bindings() == capabilities(model=YumeLucidModel)["output_bindings"]


def test_manifest_includes_uploaded_file_metadata() -> None:
    generated = manifest(_InputFileModel)

    assert generated["inputs"] == [
        {
            "name": "set_initial_frame",
            "description": "Set an initial frame.",
            "args_schema": {
                "additionalProperties": False,
                "properties": {
                    "image": {
                        "title": "Image",
                        "type": "string",
                        "x-lucid-upload": {
                            "kind": "image",
                            "max_bytes": 1_000_000,
                            "mime_types": [
                                "image/jpeg",
                                "image/png",
                                "image/webp",
                            ],
                            "target_height": 64,
                            "target_width": 64,
                        },
                    }
                },
                "required": ["image"],
                "title": "SetInitialFrameInputArgs",
                "type": "object",
            },
        }
    ]


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
        logger=_logger(),
    )

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    await ctx.publish("video", frame)
    await ctx.publish("state", {"ok": True})
    await ctx.publish("blob", b"abc")

    assert seen[0] == ("video", frame)
    assert seen[1] == ("state", b'{"ok":true}')
    assert seen[2] == ("blob", b"abc")


@pytest.mark.asyncio
async def test_session_context_rejects_bad_video_frames() -> None:
    ctx = SessionContext(
        session_id="s1",
        room_name="wm-s1",
        outputs=(publish.video(name="video", width=4, height=4, fps=8),),
        publish_fn=_noop_publish,
        logger=_logger(),
    )

    with pytest.raises(Exception, match="uint8"):
        await ctx.publish("video", np.zeros((4, 4, 3), dtype=np.float32))

    with pytest.raises(Exception, match="expects frame shape"):
        await ctx.publish("video", np.zeros((4, 5, 3), dtype=np.uint8))

    with pytest.raises(Exception, match="C-contiguous"):
        await ctx.publish("video", np.zeros((4, 4, 3), dtype=np.uint8).transpose(1, 0, 2))


@pytest.mark.asyncio
async def test_session_context_waits_until_resumed() -> None:
    ctx = SessionContext(
        session_id="s1",
        room_name="wm-s1",
        outputs=(),
        publish_fn=_noop_publish,
        logger=_logger(),
    )
    assert ctx.pause() is True

    waiter = asyncio.create_task(ctx.wait_if_paused())
    await asyncio.sleep(0.01)
    assert waiter.done() is False

    assert ctx.resume() is True
    await asyncio.wait_for(waiter, timeout=1.0)


@pytest.mark.asyncio
async def test_runtime_resolves_uploaded_file_ids() -> None:
    runtime = LucidRuntime.load_model(
        runtime_config=RuntimeConfig(livekit_url="wss://example.livekit.invalid"),
        logger=_logger(),
        model=_InputFileModel,
    )
    resolved = InputFile(
        id="upload-1",
        filename="seed.png",
        mime_type="image/png",
        size_bytes=4,
        sha256="seed",
        data=b"seed",
    )

    def resolve_input_file(file_id: str) -> InputFile | None:
        if file_id == "upload-1":
            return resolved
        return None

    runtime_session = runtime.open_session(
        session_id="s1",
        room_name="wm-s1",
        publish_fn=_noop_publish,
        input_file_resolver=resolve_input_file,
    )

    await runtime_session.dispatch_input("set_initial_frame", {"image": "upload-1"})

    assert isinstance(runtime_session.session, _InputFileSession)
    assert runtime_session.session.initial_frame == resolved


@pytest.mark.asyncio
async def test_runtime_rejects_unknown_uploaded_file_ids() -> None:
    runtime = LucidRuntime.load_model(
        runtime_config=RuntimeConfig(livekit_url="wss://example.livekit.invalid"),
        logger=_logger(),
        model=_InputFileModel,
    )

    def resolve_input_file(_file_id: str) -> InputFile | None:
        return None

    runtime_session = runtime.open_session(
        session_id="s1",
        room_name="wm-s1",
        publish_fn=_noop_publish,
        input_file_resolver=resolve_input_file,
    )

    with pytest.raises(ActionDispatchError, match="unknown input file"):
        await runtime_session.dispatch_input("set_initial_frame", {"image": "missing"})


def _logger():
    import logging

    return logging.getLogger("tests.lucid_core")


async def _noop_publish(_name: str, _payload: object, _ts_ms: int | None) -> None:
    return None
