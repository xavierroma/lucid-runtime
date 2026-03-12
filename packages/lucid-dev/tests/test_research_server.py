from __future__ import annotations

import asyncio
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lucid.publish import OutputSpec
from lucid.types import Assignment

from lucid_dev.research_server import ResearchSessionService, create_app


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


def test_research_server_exposes_same_session_shape() -> None:
    from lucid import RuntimeConfig

    service = ResearchSessionService(
        RuntimeConfig(livekit_url="wss://example.livekit.invalid", livekit_mode="fake"),
        logging.getLogger("tests.research_server"),
        model="yume_modal_example.model:YumeLucidModel",
        livekit_factory=lambda: StubLiveKitAdapter(),
    )
    app = create_app(service, api_key="api-key", api_secret="api-secret")

    with TestClient(app) as client:
        created = client.post("/sessions")
        assert created.status_code == 200
        created_payload = created.json()
        assert created_payload["session"]["state"] in {"STARTING", "READY", "RUNNING"}
        assert created_payload["client_access_token"]
        assert created_payload["capabilities"]["control_topic"] == "wm.control"
        session_id = created_payload["session"]["session_id"]

        fetched = client.get(f"/sessions/{session_id}")
        assert fetched.status_code == 200
        assert fetched.json()["session"]["session_id"] == session_id

        ended = client.post(f"/sessions/{session_id}:end")
        assert ended.status_code == 200
