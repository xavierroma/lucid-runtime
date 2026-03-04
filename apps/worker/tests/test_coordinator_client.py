from __future__ import annotations

import httpx
import pytest

from wm_worker.coordinator_client import CoordinatorClient


@pytest.mark.asyncio
async def test_heartbeat_parses_cancel_signal() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/internal/v1/worker/heartbeat":
            return httpx.Response(
                200,
                json={"status": "ok", "cancel_active_session": True},
            )
        return httpx.Response(404, json={"error": "not found"})

    transport = httpx.MockTransport(handler)
    client = CoordinatorClient(
        base_url="http://coordinator",
        worker_id="wm-worker-1",
        worker_internal_token="token",
        transport=transport,
    )
    result = await client.heartbeat_worker()
    await client.close()
    assert result.cancel_active_session is True
