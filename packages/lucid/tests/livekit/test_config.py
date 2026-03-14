from __future__ import annotations

import lucid.livekit as livekit
from lucid.livekit.config import Assignment, RuntimeConfig, SessionConfig, SessionResult


def test_runtime_config_is_plain_value_object() -> None:
    config = RuntimeConfig(
        livekit_url="wss://example.livekit.invalid",
        status_topic="wm.status",
    )

    assert config.livekit_url == "wss://example.livekit.invalid"
    assert config.status_topic == "wm.status"


def test_session_config_is_plain_value_object() -> None:
    config = SessionConfig(worker_id="wm-worker-1")

    assert config.worker_id == "wm-worker-1"


def test_assignment_is_plain_value_object() -> None:
    assignment = Assignment(
        session_id="session-1",
        room_name="wm-session-1",
        worker_access_token="worker-token",
        control_topic="wm.control",
    )

    assert assignment.session_id == "session-1"
    assert assignment.room_name == "wm-session-1"
    assert assignment.worker_access_token == "worker-token"
    assert assignment.control_topic == "wm.control"


def test_session_result_is_plain_value_object() -> None:
    result = SessionResult(error_code="MODEL_RUNTIME_ERROR", ended_by_control=True)

    assert result.error_code == "MODEL_RUNTIME_ERROR"
    assert result.ended_by_control is True


def test_fake_livekit_adapter_is_not_exported() -> None:
    assert hasattr(livekit, "SessionRunner")
    assert not hasattr(livekit, "CoordinatorClient")
    assert not hasattr(livekit, "FakeLiveKitAdapter")
