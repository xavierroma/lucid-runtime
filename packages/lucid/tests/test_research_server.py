from __future__ import annotations

import asyncio
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lucid.publish import OutputSpec

from lucid.config import RuntimeConfig
from lucid.research_server import ResearchSessionService, create_app
from lucid.types import Assignment


class StubLiveKitAdapter:
    def __init__(self) -> None:
        self.connected = False

    async def connect(self, assignment: Assignment, outputs: tuple[OutputSpec, ...]) -> None:
        _ = assignment
        _ = outputs
        self.connected = True

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
        await asyncio.sleep(min(timeout_s, 0.01))
        return None

    async def send_status(self, payload: bytes) -> None:
        _ = payload


@pytest.fixture
def worker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.invalid")
    monkeypatch.setenv("LIVEKIT_API_KEY", "api-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "api-secret")
    monkeypatch.setenv("WM_ENGINE", "fake")
    monkeypatch.setenv("WM_LIVEKIT_MODE", "fake")
    monkeypatch.setenv("YUME_MODEL_DIR", "/tmp/yume-model")
    monkeypatch.setenv("YUME_CHUNK_FRAMES", "1")
    monkeypatch.setenv("WM_MAX_QUEUE_FRAMES", "8")
    monkeypatch.setenv("WM_FRAME_WIDTH", "64")
    monkeypatch.setenv("WM_FRAME_HEIGHT", "64")


def test_research_server_exposes_same_session_shape(worker_env: None) -> None:
    service = ResearchSessionService(
        RuntimeConfig.from_env(),
        logging.getLogger("tests.research_server"),
        livekit_factory=lambda: StubLiveKitAdapter(),
    )
    app = create_app(service)

    with TestClient(app) as client:
        created = client.post("/sessions")
        assert created.status_code == 200
        created_payload = created.json()
        assert created_payload["session"]["state"] in {"STARTING", "RUNNING"}
        assert created_payload["client_access_token"]
        assert created_payload["capabilities"]["control_topic"] == "wm.control"
        session_id = created_payload["session"]["session_id"]

        fetched = client.get(f"/sessions/{session_id}")
        assert fetched.status_code == 200
        assert fetched.json()["session"]["session_id"] == session_id

        ended = client.post(f"/sessions/{session_id}:end")
        assert ended.status_code == 200
