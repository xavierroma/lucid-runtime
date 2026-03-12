from __future__ import annotations

from lucid.config import RuntimeConfig, SessionConfig


def test_runtime_config_is_plain_value_object() -> None:
    config = RuntimeConfig(livekit_url="wss://example.livekit.invalid", frame_width=64, frame_height=64)

    assert config.livekit_url == "wss://example.livekit.invalid"
    assert config.frame_width == 64
    assert config.frame_height == 64


def test_session_config_is_plain_value_object() -> None:
    config = SessionConfig(
        worker_id="wm-worker-1",
        coordinator_base_url="https://coord.example.com",
        worker_internal_token="secret",
    )

    assert config.worker_id == "wm-worker-1"
    assert config.coordinator_base_url == "https://coord.example.com"
    assert config.worker_internal_token == "secret"
