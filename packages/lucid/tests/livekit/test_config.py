from __future__ import annotations

import lucid.livekit as livekit


def test_fake_livekit_adapter_is_not_exported() -> None:
    assert hasattr(livekit, "SessionRunner")
    assert not hasattr(livekit, "CoordinatorClient")
    assert not hasattr(livekit, "FakeLiveKitAdapter")
