from __future__ import annotations

import pytest

from wm_worker.config import ConfigError, RuntimeConfig, SessionConfig


def test_runtime_config_reads_only_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COORDINATOR_BASE_URL", raising=False)
    monkeypatch.delenv("WORKER_INTERNAL_TOKEN", raising=False)
    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.invalid")

    config = RuntimeConfig.from_env()

    assert config.livekit_url == "wss://example.livekit.invalid"


def test_session_config_from_values_normalizes_fields() -> None:
    config = SessionConfig.from_values(
        worker_id="  ",
        coordinator_base_url=" https://coord.example.com/ ",
        worker_internal_token=" secret ",
    )

    assert config.worker_id == "wm-worker-1"
    assert config.coordinator_base_url == "https://coord.example.com"
    assert config.worker_internal_token == "secret"


def test_session_config_requires_callback_values() -> None:
    with pytest.raises(ConfigError):
        SessionConfig.from_values(
            worker_id="wm-worker-1",
            coordinator_base_url="",
            worker_internal_token="secret",
        )
