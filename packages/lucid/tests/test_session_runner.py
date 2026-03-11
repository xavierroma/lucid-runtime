from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
import numpy as np
from pydantic import BaseModel
import pytest

from lucid import SessionContext
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


class _PromptState(BaseModel):
    prompt: str


class StubModel:
    def __init__(self) -> None:
        self.start_calls = 0
        self.start_prompts: list[str | None] = []
        self.end_calls = 0

    async def start_session(self, ctx: SessionContext) -> None:
        self.start_calls += 1
        prompt_state = ctx.state.get("set_prompt")
        self.start_prompts.append(
            None if prompt_state is None else str(prompt_state.prompt)
        )
        ctx.running = False

    async def end_session(self, _ctx: SessionContext) -> None:
        self.end_calls += 1


class StubRuntime:
    def __init__(self) -> None:
        self.definition = type("Definition", (), {"name": "stub"})()
        self.model = StubModel()
        self.outputs: tuple[OutputSpec, ...] = ()

    async def load(self) -> None:
        return None

    def manifest(self) -> dict[str, object]:
        return {"model": {"name": "stub"}, "actions": [], "outputs": []}

    def output_bindings(self) -> list[dict[str, object]]:
        return []

    def create_session_context(
        self,
        *,
        session_id: str,
        room_name: str,
        publish_fn,
        metrics_fn=None,
    ) -> SessionContext:
        return SessionContext(
            session_id=session_id,
            room_name=room_name,
            outputs=self.outputs,
            publish_fn=publish_fn,
            metrics_fn=metrics_fn,
            logger=logging.getLogger("tests.session_runner.stub_runtime"),
        )

    async def dispatch_action(self, ctx: SessionContext, name: str, args: dict[str, Any]) -> None:
        if name == "set_prompt":
            ctx.state.set(name, _PromptState.model_validate(args))
            return
        if name == "lucid.runtime.start":
            ctx.mark_started()
            return
        if name == "lucid.runtime.pause":
            if ctx.started:
                ctx.paused = True
            return
        if name == "lucid.runtime.resume":
            if ctx.started:
                ctx.paused = False
            return
        raise RuntimeError(f"unexpected action: {name}")


@pytest.fixture
def worker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_BASE_URL", "http://coordinator")
    monkeypatch.setenv("WORKER_INTERNAL_TOKEN", "test-token")
    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.invalid")
    monkeypatch.setenv("WM_ENGINE", "fake")
    monkeypatch.setenv("WM_LIVEKIT_MODE", "fake")
    monkeypatch.setenv("YUME_MODEL_DIR", "/tmp/yume-model")
    monkeypatch.setenv("YUME_CHUNK_FRAMES", "1")
    monkeypatch.setenv("WM_MAX_QUEUE_FRAMES", "8")
    monkeypatch.setenv("WM_FRAME_WIDTH", "64")
    monkeypatch.setenv("WM_FRAME_HEIGHT", "64")


@pytest.mark.asyncio
async def test_session_runner_reaches_ready_without_starting_model(worker_env: None) -> None:
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
    runtime_config = RuntimeConfig.from_env()
    session_config = SessionConfig.from_env(worker_id_override="wm-worker-test")
    runtime = StubRuntime()
    end_message = (
        b'{"type":"end","seq":1,"ts_ms":10,'
        b'"session_id":"session-1","payload":{}}'
    )
    adapter = StubLiveKitAdapter(control_messages=[end_message])
    runner = SessionRunner(
        runtime_config,
        session_config,
        logging.getLogger("tests.session_runner"),
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
    assert "/internal/sessions/session-1/running" not in paths
    assert paths[-1] == "/internal/sessions/session-1/ended"
    assert calls[-1][1] == {"end_reason": "CONTROL_REQUESTED"}
    assert runtime.model.start_calls == 0
    assert runtime.model.end_calls == 1
    assert any(b'"type":"started"' in status for status in adapter.status_messages)
    assert any(b'"type":"ended"' in status for status in adapter.status_messages)
    assert [output.name for output in adapter.outputs] == []


@pytest.mark.asyncio
async def test_session_runner_applies_prestart_prompt_before_running(worker_env: None) -> None:
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
    prompt_message = (
        b'{"type":"action","seq":1,"ts_ms":10,"session_id":"session-3",'
        b'"payload":{"name":"set_prompt","args":{"prompt":"new prompt"}}}'
    )
    start_message = (
        b'{"type":"action","seq":2,"ts_ms":11,"session_id":"session-3",'
        b'"payload":{"name":"lucid.runtime.start","args":{}}}'
    )
    runner = SessionRunner(
        RuntimeConfig.from_env(),
        SessionConfig.from_env(worker_id_override="wm-worker-test"),
        logging.getLogger("tests.session_runner"),
        coordinator=coordinator,
        livekit_factory=lambda: StubLiveKitAdapter(control_messages=[prompt_message, start_message]),
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

    assert runtime.model.start_calls == 1
    assert runtime.model.start_prompts == ["new prompt"]
    paths = [path for path, _ in calls]
    ready_index = paths.index("/internal/sessions/session-3/ready")
    running_index = paths.index("/internal/sessions/session-3/running")
    assert ready_index < running_index
    assert calls[-1][0] == "/internal/sessions/session-3/ended"


@pytest.mark.asyncio
async def test_session_runner_reports_error_when_connect_fails(worker_env: None) -> None:
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
    runtime_config = RuntimeConfig.from_env()
    session_config = SessionConfig.from_env(worker_id_override="wm-worker-test")
    runner = SessionRunner(
        runtime_config,
        session_config,
        logging.getLogger("tests.session_runner"),
        coordinator=coordinator,
        livekit_factory=lambda: StubLiveKitAdapter(fail_connect=True),
    )

    await runner.run_session(
        Assignment(
            session_id="session-2",
            room_name="wm-session-2",
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
async def test_session_runner_exposes_manifest_and_output_bindings(worker_env: None) -> None:
    runner = SessionRunner(
        RuntimeConfig.from_env(),
        SessionConfig.from_env(worker_id_override="wm-worker-test"),
        logging.getLogger("tests.session_runner"),
        livekit_factory=lambda: StubLiveKitAdapter(),
    )

    assert runner.manifest["model"]["name"] == "yume"
    assert runner.output_bindings == [
        {"name": "main_video", "kind": "video", "track_name": "main_video"}
    ]
    await runner.close()
