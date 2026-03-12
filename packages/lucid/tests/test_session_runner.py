from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import numpy as np
import pytest

from lucid import SessionContext, publish
from lucid.publish import OutputSpec

from lucid.config import RuntimeConfig, SessionConfig
from lucid.coordinator import CoordinatorClient
from lucid.host import SessionRunner
from lucid.types import Assignment


class StubLiveKitAdapter:
    def __init__(self, control_messages: list[bytes] | None = None, fail_connect: bool = False):
        self._control_messages = asyncio.Queue[bytes]()
        for message in control_messages or []:
            self._control_messages.put_nowait(message)
        self._fail_connect = fail_connect
        self.status_messages: list[bytes] = []
        self.connected = False
        self.outputs: tuple[OutputSpec, ...] = ()

    async def connect(self, assignment: Assignment, outputs: tuple[OutputSpec, ...]) -> None:
        if self._fail_connect:
            raise RuntimeError("connect failed")
        _ = assignment
        self.connected = True
        self.outputs = outputs

    async def disconnect(self) -> None:
        self.connected = False

    async def publish_video(self, output_name: str, frame: np.ndarray) -> None:
        _ = output_name
        _ = frame

    async def publish_audio(self, output_name: str, samples: np.ndarray) -> None:
        _ = output_name
        _ = samples

    async def publish_data(self, output_name: str, payload: bytes, *, reliable: bool = True) -> None:
        _ = output_name
        _ = payload
        _ = reliable

    async def recv_control(self, timeout_s: float) -> bytes | None:
        try:
            return await asyncio.wait_for(self._control_messages.get(), timeout=timeout_s)
        except TimeoutError:
            return None

    async def send_status(self, payload: bytes) -> None:
        self.status_messages.append(payload)


class _StubSession:
    def __init__(
        self,
        runtime: "StubRuntime",
        ctx: SessionContext,
    ) -> None:
        self.runtime = runtime
        self.ctx = ctx
        self.prompt = "default prompt"
        self.run_calls = 0
        self.close_calls = 0

    async def run(self) -> None:
        self.run_calls += 1
        if self.runtime.read_prompt_delay_s > 0:
            await asyncio.sleep(self.runtime.read_prompt_delay_s)
        self.runtime.observed_prompts.append(self.prompt)
        if self.runtime.block_until_stopped:
            while self.ctx.running:
                await asyncio.sleep(0.01)
        self.ctx.running = False

    async def close(self) -> None:
        self.close_calls += 1
        self.runtime.close_calls += 1


class StubRuntime:
    def __init__(
        self,
        *,
        outputs: tuple[OutputSpec, ...] = (),
        read_prompt_delay_s: float = 0.0,
        block_until_stopped: bool = False,
    ) -> None:
        self.definition = type(
            "Definition",
            (),
            {
                "name": "stub",
                "inputs": (type("InputDefinition", (), {"binding": None})(),),
            },
        )()
        self.outputs = outputs
        self.read_prompt_delay_s = read_prompt_delay_s
        self.block_until_stopped = block_until_stopped
        self.observed_prompts: list[str] = []
        self.close_calls = 0
        self.sessions: dict[str, _StubSession] = {}

    async def load(self) -> None:
        return None

    async def unload(self) -> None:
        return None

    def manifest(self) -> dict[str, object]:
        return {
            "model": {"name": "stub"},
            "inputs": [
                {
                    "name": "set_prompt",
                    "description": "Update prompt",
                    "args_schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"prompt": {"type": "string"}},
                        "required": ["prompt"],
                    },
                }
            ],
            "outputs": [output.to_manifest() for output in self.outputs],
        }

    def output_bindings(self) -> list[dict[str, object]]:
        bindings: list[dict[str, object]] = []
        for output in self.outputs:
            bindings.append(
                {
                    "name": output.name,
                    "kind": output.kind,
                    "track_name": output.name,
                }
            )
        return bindings

    def create_session_context(
        self,
        *,
        session_id: str,
        room_name: str,
        publish_fn: Callable[[str, Any, int | None], Awaitable[None]],
        metrics_fn=None,
    ) -> SessionContext:
        ctx = SessionContext(
            session_id=session_id,
            room_name=room_name,
            outputs=self.outputs,
            publish_fn=publish_fn,
            metrics_fn=metrics_fn,
            logger=logging.getLogger("tests.session_runner.stub_runtime"),
        )
        self.sessions[session_id] = _StubSession(self, ctx)
        return ctx

    async def dispatch_input(self, ctx: SessionContext, name: str, args: dict[str, Any]) -> None:
        if name != "set_prompt":
            raise RuntimeError(f"unexpected input: {name}")
        prompt = args.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise RuntimeError("prompt is required")
        self.sessions[ctx.session_id].prompt = prompt

    async def run_session(self, ctx: SessionContext) -> None:
        await self.sessions[ctx.session_id].run()

    async def close_session(self, ctx: SessionContext) -> None:
        session = self.sessions.pop(ctx.session_id, None)
        if session is not None:
            await session.close()


def _runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        livekit_url="wss://example.livekit.invalid",
        frame_width=64,
        frame_height=64,
        max_queue_frames=8,
        livekit_mode="fake",
    )


def _session_config() -> SessionConfig:
    return SessionConfig(
        worker_id="wm-worker-test",
        coordinator_base_url="http://coordinator",
        worker_internal_token="test-token",
    )


@pytest.mark.asyncio
async def test_session_runner_runs_session_after_input_grace_and_marks_running(
) -> None:
    calls: list[tuple[str, dict[str, Any] | None]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = None
        if request.content:
            body = json.loads(request.content.decode("utf-8"))
        calls.append((request.url.path, body))
        return httpx.Response(200, json={"status": "ok"})

    transport = httpx.MockTransport(handler)
    coordinator = CoordinatorClient(
        base_url="http://coordinator",
        worker_internal_token="test-token",
        transport=transport,
    )
    runtime = StubRuntime()
    adapter = StubLiveKitAdapter()
    runner = SessionRunner(
        _runtime_config(),
        _session_config(),
        logging.getLogger("tests.session_runner"),
        model="yume_modal_example.model:YumeLucidModel",
        coordinator=coordinator,
        livekit_factory=lambda: adapter,
        runtime=runtime,
    )

    await runner.run_session(
        Assignment(
            session_id="session-1",
            room_name="wm-session-1",
            worker_access_token="worker-token",
            control_topic="wm.control",
        )
    )
    await runner.close()

    paths = [path for path, _ in calls]
    assert paths[0] == "/internal/sessions/session-1/ready"
    assert "/internal/sessions/session-1/running" in paths
    assert paths[-1] == "/internal/sessions/session-1/ended"
    assert calls[-1][1] == {}
    assert runtime.observed_prompts == ["default prompt"]
    assert runtime.close_calls == 1
    assert any(b'"type":"started"' in status for status in adapter.status_messages)
    assert any(b'"type":"ended"' in status for status in adapter.status_messages)


@pytest.mark.asyncio
async def test_session_runner_applies_inputs_before_session_starts() -> None:
    calls: list[tuple[str, dict[str, Any] | None]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = None
        if request.content:
            body = json.loads(request.content.decode("utf-8"))
        calls.append((request.url.path, body))
        return httpx.Response(200, json={"status": "ok"})

    transport = httpx.MockTransport(handler)
    coordinator = CoordinatorClient(
        base_url="http://coordinator",
        worker_internal_token="test-token",
        transport=transport,
    )
    runtime = StubRuntime(read_prompt_delay_s=0.0)
    prompt_message = (
        b'{"type":"action","seq":1,"ts_ms":10,"session_id":"session-3",'
        b'"payload":{"name":"set_prompt","args":{"prompt":"new prompt"}}}'
    )
    runner = SessionRunner(
        _runtime_config(),
        _session_config(),
        logging.getLogger("tests.session_runner"),
        model="yume_modal_example.model:YumeLucidModel",
        coordinator=coordinator,
        livekit_factory=lambda: StubLiveKitAdapter(control_messages=[prompt_message]),
        runtime=runtime,
    )

    await runner.run_session(
        Assignment(
            session_id="session-3",
            room_name="wm-session-3",
            worker_access_token="worker-token",
            control_topic="wm.control",
        )
    )
    await runner.close()

    assert runtime.observed_prompts == ["new prompt"]
    paths = [path for path, _ in calls]
    ready_index = paths.index("/internal/sessions/session-3/ready")
    running_index = paths.index("/internal/sessions/session-3/running")
    assert ready_index < running_index
    assert calls[-1][0] == "/internal/sessions/session-3/ended"


@pytest.mark.asyncio
async def test_session_runner_ends_when_control_requests_stop() -> None:
    ended_payloads: list[dict[str, Any] | None] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = None
        if request.content:
            body = json.loads(request.content.decode("utf-8"))
        if request.url.path.endswith("/ended"):
            ended_payloads.append(body)
        return httpx.Response(200, json={"status": "ok"})

    transport = httpx.MockTransport(handler)
    coordinator = CoordinatorClient(
        base_url="http://coordinator",
        worker_internal_token="test-token",
        transport=transport,
    )
    runtime = StubRuntime(block_until_stopped=True)
    end_message = (
        b'{"type":"end","seq":1,"ts_ms":10,'
        b'"session_id":"session-2","payload":{}}'
    )
    runner = SessionRunner(
        _runtime_config(),
        _session_config(),
        logging.getLogger("tests.session_runner"),
        model="yume_modal_example.model:YumeLucidModel",
        coordinator=coordinator,
        livekit_factory=lambda: StubLiveKitAdapter(control_messages=[end_message]),
        runtime=runtime,
    )

    result = await runner.run_session(
        Assignment(
            session_id="session-2",
            room_name="wm-session-2",
            worker_access_token="worker-token",
            control_topic="wm.control",
        )
    )
    await runner.close()

    assert result.ended_by_control is True
    assert result.error_code is None
    assert ended_payloads == [{"end_reason": "CONTROL_REQUESTED"}]
    assert runtime.close_calls == 1


@pytest.mark.asyncio
async def test_session_runner_reports_error_when_connect_fails() -> None:
    ended_payloads: list[dict[str, Any] | None] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = None
        if request.content:
            body = json.loads(request.content.decode("utf-8"))
        if request.url.path.endswith("/ended"):
            ended_payloads.append(body)
        return httpx.Response(200, json={"status": "ok"})

    transport = httpx.MockTransport(handler)
    coordinator = CoordinatorClient(
        base_url="http://coordinator",
        worker_internal_token="test-token",
        transport=transport,
    )
    runner = SessionRunner(
        _runtime_config(),
        _session_config(),
        logging.getLogger("tests.session_runner"),
        model="yume_modal_example.model:YumeLucidModel",
        coordinator=coordinator,
        livekit_factory=lambda: StubLiveKitAdapter(fail_connect=True),
        runtime=StubRuntime(),
    )

    await runner.run_session(
        Assignment(
            session_id="session-4",
            room_name="wm-session-4",
            worker_access_token="worker-token",
            control_topic="wm.control",
        )
    )
    await runner.close()

    assert ended_payloads, "expected ended callback"
    assert ended_payloads[0] == {
        "error_code": "LIVEKIT_DISCONNECT",
        "end_reason": "WORKER_REPORTED_ERROR",
    }


@pytest.mark.asyncio
async def test_session_runner_exposes_manifest_and_output_bindings() -> None:
    runtime = StubRuntime(
        outputs=(publish.video(name="main_video", width=64, height=64, fps=8),)
    )
    runner = SessionRunner(
        _runtime_config(),
        _session_config(),
        logging.getLogger("tests.session_runner"),
        model="yume_modal_example.model:YumeLucidModel",
        livekit_factory=lambda: StubLiveKitAdapter(),
        runtime=runtime,
    )

    assert runner.manifest["model"]["name"] == "stub"
    assert runner.manifest["inputs"][0]["name"] == "set_prompt"
    assert runner.output_bindings == [
        {"name": "main_video", "kind": "video", "track_name": "main_video"}
    ]
    await runner.close()
