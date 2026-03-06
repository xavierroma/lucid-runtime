from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
import numpy as np
import pytest

from wm_worker.config import WorkerConfig
from wm_worker.coordinator_client import CoordinatorClient
from wm_worker.models import Assignment
from wm_worker.session_runner import SessionRunner


class StubLiveKitAdapter:
    def __init__(self, control_messages: list[bytes] | None = None, fail_connect: bool = False):
        self._control_messages = asyncio.Queue[bytes]()
        for message in control_messages or []:
            self._control_messages.put_nowait(message)
        self._fail_connect = fail_connect
        self.status_messages: list[bytes] = []
        self.connected = False

    async def connect_and_publish(self, assignment: Assignment) -> None:
        if self._fail_connect:
            raise RuntimeError("connect failed")
        _ = assignment
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def publish_frame(self, frame: np.ndarray) -> None:
        _ = frame

    async def recv_control(self, timeout_s: float) -> bytes | None:
        try:
            return await asyncio.wait_for(self._control_messages.get(), timeout=timeout_s)
        except TimeoutError:
            return None

    async def send_status(self, payload: bytes) -> None:
        self.status_messages.append(payload)


class StubEngine:
    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc

    async def load(self) -> None:
        return None

    async def start_session(self, prompt: str | None) -> None:
        _ = prompt

    async def update_snapshot(self, snapshot: Any) -> None:
        _ = snapshot

    async def generate_chunk(self) -> Any:
        if self._exc is not None:
            raise self._exc
        return type("Chunk", (), {"frames": [np.zeros((64, 64, 3), dtype=np.uint8)], "inference_ms": 1.0})()

    async def end_session(self) -> None:
        return None


@pytest.fixture
def worker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_BASE_URL", "http://coordinator")
    monkeypatch.setenv("WORKER_INTERNAL_TOKEN", "test-token")
    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.invalid")
    monkeypatch.setenv("WM_ENGINE", "fake")
    monkeypatch.setenv("WM_LIVEKIT_MODE", "fake")
    monkeypatch.setenv("YUME_MODEL_DIR", "/tmp/yume-model")
    monkeypatch.setenv("YUME_CHUNK_FRAMES", "1")
    monkeypatch.setenv("WM_FRAME_WIDTH", "64")
    monkeypatch.setenv("WM_FRAME_HEIGHT", "64")


@pytest.mark.asyncio
async def test_session_runner_calls_running_and_ended(worker_env: None) -> None:
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
    config = WorkerConfig.from_env(worker_id_override="wm-worker-test")
    end_message = (
        b'{"v":"v1","type":"end","seq":1,"ts_ms":10,'
        b'"session_id":"session-1","payload":{}}'
    )
    adapter = StubLiveKitAdapter(control_messages=[end_message])
    runner = SessionRunner(
        config,
        logging.getLogger("tests.session_runner"),
        coordinator=coordinator,
        livekit_factory=lambda: adapter,
    )

    await runner.run_session(
        Assignment(
            session_id="session-1",
            room_name="wm-session-1",
            worker_access_token="worker-token",
            video_track_name="main_video",
            control_topic="wm.control.v1",
        )
    )
    await runner.close()

    assert [path for path, _ in calls] == [
        "/internal/v1/sessions/session-1/running",
        "/internal/v1/sessions/session-1/ended",
    ]
    assert any(b'"type":"started"' in status for status in adapter.status_messages)
    assert any(b'"type":"ended"' in status for status in adapter.status_messages)


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
    config = WorkerConfig.from_env(worker_id_override="wm-worker-test")
    runner = SessionRunner(
        config,
        logging.getLogger("tests.session_runner"),
        coordinator=coordinator,
        livekit_factory=lambda: StubLiveKitAdapter(fail_connect=True),
    )

    await runner.run_session(
        Assignment(
            session_id="session-2",
            room_name="wm-session-2",
            worker_access_token="worker-token",
            video_track_name="main_video",
            control_topic="wm.control.v1",
        )
    )
    await runner.close()

    assert ended_payloads, "expected ended callback"
    assert ended_payloads[0] == {"error_code": "LIVEKIT_DISCONNECT"}


@pytest.mark.asyncio
async def test_session_runner_treats_modal_input_cancellation_as_clean_end(
    worker_env: None,
) -> None:
    InputCancellation = type(
        "InputCancellation",
        (Exception,),
        {"__module__": "modal.exception"},
    )
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
    config = WorkerConfig.from_env(worker_id_override="wm-worker-test")
    runner = SessionRunner(
        config,
        logging.getLogger("tests.session_runner"),
        coordinator=coordinator,
        livekit_factory=lambda: StubLiveKitAdapter(),
        engine=StubEngine(exc=InputCancellation("cancelled")),
    )

    result = await runner.run_session(
        Assignment(
            session_id="session-cancel",
            room_name="wm-session-cancel",
            worker_access_token="worker-token",
            video_track_name="main_video",
            control_topic="wm.control.v1",
        )
    )
    await runner.close()

    assert result.error_code is None
    assert [path for path, _ in calls] == [
        "/internal/v1/sessions/session-cancel/ended",
    ]
    assert calls[-1][1] == {}
