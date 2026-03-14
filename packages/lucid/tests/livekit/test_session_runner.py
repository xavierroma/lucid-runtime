from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np
import pytest

from lucid import OutputSpec, SessionContext, publish
from lucid.livekit import Assignment
from lucid.livekit.config import RuntimeConfig, SessionConfig
from lucid.livekit.runner import SessionRunner


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
        if timeout_s <= 0:
            try:
                return self._control_messages.get_nowait()
            except asyncio.QueueEmpty:
                return None
        try:
            return await asyncio.wait_for(self._control_messages.get(), timeout=timeout_s)
        except TimeoutError:
            return None

    async def send_status(self, payload: bytes) -> None:
        self.status_messages.append(payload)

    async def inject_control(self, payload: bytes) -> None:
        await self._control_messages.put(payload)


class StubLifecycleReporter:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any] | None]] = []
        self.close_calls = 0

    async def ready(self, session_id: str) -> None:
        self.events.append(("ready", session_id, None))

    async def running(self, session_id: str) -> None:
        self.events.append(("running", session_id, None))

    async def paused(self, session_id: str) -> None:
        self.events.append(("paused", session_id, None))

    async def heartbeat(self, session_id: str) -> None:
        self.events.append(("heartbeat", session_id, None))

    async def ended(
        self,
        session_id: str,
        error_code: str | None,
        end_reason: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {}
        if error_code is not None:
            payload["error_code"] = error_code
        if end_reason is not None:
            payload["end_reason"] = end_reason
        self.events.append(("ended", session_id, payload))

    async def close(self) -> None:
        self.close_calls += 1


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
                await self.ctx.wait_if_paused()
                if not self.ctx.running:
                    break
                self.runtime.progress_count += 1
                await asyncio.sleep(0.01)
        self.ctx.running = False

    async def close(self) -> None:
        self.close_calls += 1
        self.runtime.close_calls += 1


class _StubRuntimeSession:
    def __init__(self, runtime: "StubRuntime", session: _StubSession, ctx: SessionContext) -> None:
        self._runtime = runtime
        self.session = session
        self.ctx = ctx

    async def dispatch_input(self, name: str, args: dict[str, Any]) -> None:
        if name != "set_prompt":
            raise RuntimeError(f"unexpected input: {name}")
        prompt = args.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise RuntimeError("prompt is required")
        self.session.prompt = prompt

    async def run(self) -> None:
        await self.session.run()

    async def close(self) -> None:
        await self.session.close()
        self._runtime.sessions.pop(self.ctx.session_id, None)

    def allows_input_while_paused(self, name: str) -> bool:
        return name == "set_prompt"


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
        self.progress_count = 0
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

    def open_session(
        self,
        *,
        session_id: str,
        room_name: str,
        publish_fn: Callable[[str, Any, int | None], Awaitable[None]],
        metrics_fn=None,
    ) -> _StubRuntimeSession:
        ctx = SessionContext(
            session_id=session_id,
            room_name=room_name,
            outputs=self.outputs,
            publish_fn=publish_fn,
            metrics_fn=metrics_fn,
            logger=logging.getLogger("tests.session_runner.stub_runtime"),
        )
        session = _StubSession(self, ctx)
        self.sessions[session_id] = session
        return _StubRuntimeSession(self, session, ctx)


def _runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        livekit_url="wss://example.livekit.invalid",
    )


def _session_config() -> SessionConfig:
    return SessionConfig(worker_id="wm-worker-test")


def _lifecycle_names(reporter: StubLifecycleReporter, session_id: str) -> list[str]:
    return [kind for kind, event_session_id, _ in reporter.events if event_session_id == session_id]


def _lifecycle_payloads(
    reporter: StubLifecycleReporter,
    session_id: str,
    kind: str,
) -> list[dict[str, Any] | None]:
    return [
        payload
        for event_kind, event_session_id, payload in reporter.events
        if event_kind == kind and event_session_id == session_id
    ]


async def _wait_for(
    predicate: Callable[[], bool],
    *,
    timeout_s: float = 1.0,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("timed out waiting for condition")


@pytest.mark.asyncio
async def test_session_runner_waits_for_resume_before_running() -> None:
    reporter = StubLifecycleReporter()
    runtime = StubRuntime()
    adapter = StubLiveKitAdapter()
    runner = SessionRunner(
        _runtime_config(),
        _session_config(),
        logging.getLogger("tests.session_runner"),
        model="yume_modal_example.model:YumeLucidModel",
        reporter=reporter,
        livekit_factory=lambda: adapter,
        runtime=runtime,
    )

    task = asyncio.create_task(
        runner.run_session(
            Assignment(
                session_id="session-1",
                room_name="wm-session-1",
                worker_access_token="worker-token",
                control_topic="wm.control",
            )
        )
    )
    await asyncio.sleep(0.05)

    names = _lifecycle_names(reporter, "session-1")
    assert names[0] == "ready"
    assert "running" not in names
    assert runtime.observed_prompts == []

    await adapter.inject_control(
        b'{"type":"resume","seq":1,"ts_ms":10,"session_id":"session-1","payload":{}}'
    )

    await task
    await runner.close()

    names = _lifecycle_names(reporter, "session-1")
    assert names[0] == "ready"
    assert "running" in names
    assert names[-1] == "ended"
    assert _lifecycle_payloads(reporter, "session-1", "ended") == [{}]
    assert runtime.observed_prompts == ["default prompt"]
    assert runtime.close_calls == 1
    assert reporter.close_calls == 1
    assert any(b'"type":"started"' in status for status in adapter.status_messages)
    assert any(b'"type":"ended"' in status for status in adapter.status_messages)


@pytest.mark.asyncio
async def test_session_runner_applies_inputs_before_resume() -> None:
    reporter = StubLifecycleReporter()
    runtime = StubRuntime(read_prompt_delay_s=0.0)
    prompt_message = (
        b'{"type":"action","seq":1,"ts_ms":10,"session_id":"session-3",'
        b'"payload":{"name":"set_prompt","args":{"prompt":"new prompt"}}}'
    )
    resume_message = (
        b'{"type":"resume","seq":2,"ts_ms":11,"session_id":"session-3","payload":{}}'
    )
    runner = SessionRunner(
        _runtime_config(),
        _session_config(),
        logging.getLogger("tests.session_runner"),
        model="yume_modal_example.model:YumeLucidModel",
        reporter=reporter,
        livekit_factory=lambda: StubLiveKitAdapter(
            control_messages=[prompt_message, resume_message]
        ),
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
    names = _lifecycle_names(reporter, "session-3")
    ready_index = names.index("ready")
    running_index = names.index("running")
    assert ready_index < running_index
    assert names[-1] == "ended"


@pytest.mark.asyncio
async def test_session_runner_pauses_and_resumes_without_losing_state() -> None:
    reporter = StubLifecycleReporter()
    runtime = StubRuntime(block_until_stopped=True)
    adapter = StubLiveKitAdapter()
    runner = SessionRunner(
        _runtime_config(),
        _session_config(),
        logging.getLogger("tests.session_runner"),
        model="yume_modal_example.model:YumeLucidModel",
        reporter=reporter,
        livekit_factory=lambda: adapter,
        runtime=runtime,
    )

    task = asyncio.create_task(
        runner.run_session(
            Assignment(
                session_id="session-pause",
                room_name="wm-session-pause",
                worker_access_token="worker-token",
                control_topic="wm.control",
            )
        )
    )

    await adapter.inject_control(
        b'{"type":"resume","seq":1,"ts_ms":10,"session_id":"session-pause","payload":{}}'
    )
    await _wait_for(
        lambda: _lifecycle_names(reporter, "session-pause").count("running") == 1
    )
    await _wait_for(lambda: runtime.progress_count > 0)

    await adapter.inject_control(
        b'{"type":"pause","seq":2,"ts_ms":11,"session_id":"session-pause","payload":{}}'
    )
    await _wait_for(lambda: "paused" in _lifecycle_names(reporter, "session-pause"))
    paused_progress = runtime.progress_count
    await asyncio.sleep(0.05)
    assert runtime.progress_count == paused_progress

    await adapter.inject_control(
        b'{"type":"resume","seq":3,"ts_ms":12,"session_id":"session-pause","payload":{}}'
    )
    await _wait_for(
        lambda: _lifecycle_names(reporter, "session-pause").count("running") == 2
    )
    await _wait_for(lambda: runtime.progress_count > paused_progress)

    runner.stop()
    await task
    await runner.close()

    names = _lifecycle_names(reporter, "session-pause")
    assert names[0] == "ready"
    assert names.count("running") == 2
    assert names.count("paused") == 1
    assert names[-1] == "ended"
    assert runtime.observed_prompts == ["default prompt"]
    assert runtime.close_calls == 1
    assert next(iter(runtime.sessions.values()), None) is None


@pytest.mark.asyncio
async def test_session_runner_ignores_duplicate_pause_and_resume_messages() -> None:
    reporter = StubLifecycleReporter()
    runtime = StubRuntime(block_until_stopped=True)
    adapter = StubLiveKitAdapter()
    runner = SessionRunner(
        _runtime_config(),
        _session_config(),
        logging.getLogger("tests.session_runner"),
        model="yume_modal_example.model:YumeLucidModel",
        reporter=reporter,
        livekit_factory=lambda: adapter,
        runtime=runtime,
    )

    task = asyncio.create_task(
        runner.run_session(
            Assignment(
                session_id="session-duplicate",
                room_name="wm-session-duplicate",
                worker_access_token="worker-token",
                control_topic="wm.control",
            )
        )
    )

    await adapter.inject_control(
        b'{"type":"resume","seq":1,"ts_ms":10,"session_id":"session-duplicate","payload":{}}'
    )
    await adapter.inject_control(
        b'{"type":"resume","seq":2,"ts_ms":11,"session_id":"session-duplicate","payload":{}}'
    )
    await _wait_for(
        lambda: _lifecycle_names(reporter, "session-duplicate").count("running") == 1
    )

    await adapter.inject_control(
        b'{"type":"pause","seq":3,"ts_ms":12,"session_id":"session-duplicate","payload":{}}'
    )
    await adapter.inject_control(
        b'{"type":"pause","seq":4,"ts_ms":13,"session_id":"session-duplicate","payload":{}}'
    )
    await _wait_for(
        lambda: _lifecycle_names(reporter, "session-duplicate").count("paused") == 1
    )

    await adapter.inject_control(
        b'{"type":"resume","seq":5,"ts_ms":14,"session_id":"session-duplicate","payload":{}}'
    )
    await adapter.inject_control(
        b'{"type":"resume","seq":6,"ts_ms":15,"session_id":"session-duplicate","payload":{}}'
    )
    await _wait_for(
        lambda: _lifecycle_names(reporter, "session-duplicate").count("running") == 2
    )

    runner.stop()
    await task
    await runner.close()

    names = _lifecycle_names(reporter, "session-duplicate")
    assert names.count("running") == 2
    assert names.count("paused") == 1


@pytest.mark.asyncio
async def test_session_runner_ends_when_control_requests_stop() -> None:
    reporter = StubLifecycleReporter()
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
        reporter=reporter,
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
    assert _lifecycle_payloads(reporter, "session-2", "ended") == [
        {"end_reason": "CONTROL_REQUESTED"}
    ]
    assert runtime.close_calls == 1


@pytest.mark.asyncio
async def test_session_runner_keeps_run_state_per_session() -> None:
    runtime = StubRuntime()
    first_adapter = StubLiveKitAdapter()
    second_adapter = StubLiveKitAdapter()
    adapters = [first_adapter, second_adapter]

    def next_adapter() -> StubLiveKitAdapter:
        return adapters.pop(0)

    runner = SessionRunner(
        _runtime_config(),
        None,
        logging.getLogger("tests.session_runner"),
        model="yume_modal_example.model:YumeLucidModel",
        livekit_factory=next_adapter,
        runtime=runtime,
    )

    first = asyncio.create_task(
        runner.run_session(
            Assignment(
                session_id="session-a",
                room_name="wm-session-a",
                worker_access_token="worker-token",
                control_topic="wm.control",
            )
        )
    )
    second = asyncio.create_task(
        runner.run_session(
            Assignment(
                session_id="session-b",
                room_name="wm-session-b",
                worker_access_token="worker-token",
                control_topic="wm.control",
            )
        )
    )

    await asyncio.sleep(0.05)
    await first_adapter.inject_control(
        b'{"type":"resume","seq":1,"ts_ms":10,"session_id":"session-a","payload":{}}'
    )
    await first

    assert runtime.observed_prompts == ["default prompt"]
    assert second.done() is False

    await second_adapter.inject_control(
        b'{"type":"resume","seq":1,"ts_ms":10,"session_id":"session-b","payload":{}}'
    )
    await second
    await runner.close()

    assert runtime.observed_prompts == ["default prompt", "default prompt"]


@pytest.mark.asyncio
async def test_session_runner_reports_error_when_connect_fails() -> None:
    reporter = StubLifecycleReporter()
    runner = SessionRunner(
        _runtime_config(),
        _session_config(),
        logging.getLogger("tests.session_runner"),
        model="yume_modal_example.model:YumeLucidModel",
        reporter=reporter,
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

    assert _lifecycle_payloads(reporter, "session-4", "ended") == [{
        "error_code": "LIVEKIT_DISCONNECT",
        "end_reason": "WORKER_REPORTED_ERROR",
    }]


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
