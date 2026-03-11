from __future__ import annotations

import httpx
import pytest

from lucid.coordinator import CoordinatorAuthError, CoordinatorClient


@pytest.mark.asyncio
async def test_coordinator_client_posts_internal_session_paths() -> None:
    seen_paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        return httpx.Response(200, json={"status": "ok"})

    transport = httpx.MockTransport(handler)
    client = CoordinatorClient(
        base_url="http://coordinator",
        worker_internal_token="token",
        transport=transport,
    )

    await client.mark_ready("session-1")
    await client.mark_running("session-1")
    await client.mark_heartbeat("session-1")
    await client.mark_ended("session-1", "MODEL_RUNTIME_ERROR", "WORKER_REPORTED_ERROR")
    await client.close()

    assert seen_paths == [
        "/internal/sessions/session-1/ready",
        "/internal/sessions/session-1/running",
        "/internal/sessions/session-1/heartbeat",
        "/internal/sessions/session-1/ended",
    ]


@pytest.mark.asyncio
async def test_mark_running_raises_auth_error() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    transport = httpx.MockTransport(handler)
    client = CoordinatorClient(
        base_url="http://coordinator",
        worker_internal_token="token",
        transport=transport,
    )

    with pytest.raises(CoordinatorAuthError):
        await client.mark_running("session-1")
    await client.close()
