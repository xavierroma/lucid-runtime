from __future__ import annotations

import json
from typing import Protocol

import httpx

__all__ = [
    "CoordinatorAuthError",
    "CoordinatorClient",
    "CoordinatorError",
    "CoordinatorNotFoundError",
    "NoopSessionLifecycleReporter",
    "SessionLifecycleReporter",
]


class SessionLifecycleReporter(Protocol):
    async def ready(self, session_id: str) -> None: ...
    async def running(self, session_id: str) -> None: ...
    async def paused(self, session_id: str) -> None: ...
    async def heartbeat(self, session_id: str) -> None: ...
    async def ended(
        self,
        session_id: str,
        error_code: str | None,
        end_reason: str | None = None,
    ) -> None: ...
    async def close(self) -> None: ...


class NoopSessionLifecycleReporter:
    async def ready(self, session_id: str) -> None:
        _ = session_id

    async def running(self, session_id: str) -> None:
        _ = session_id

    async def paused(self, session_id: str) -> None:
        _ = session_id

    async def heartbeat(self, session_id: str) -> None:
        _ = session_id

    async def ended(
        self,
        session_id: str,
        error_code: str | None,
        end_reason: str | None = None,
    ) -> None:
        _ = session_id
        _ = error_code
        _ = end_reason

    async def close(self) -> None:
        return None


class CoordinatorError(RuntimeError):
    pass


class CoordinatorAuthError(CoordinatorError):
    pass


class CoordinatorNotFoundError(CoordinatorError):
    pass


class CoordinatorClient:
    def __init__(
        self,
        *,
        base_url: str,
        worker_internal_token: str,
        timeout_seconds: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._headers = {"Authorization": f"Bearer {worker_internal_token}"}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
            headers=self._headers,
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def ready(self, session_id: str) -> None:
        response = await self._client.post(f"/internal/sessions/{session_id}/ready")
        self._raise_for_error(response)

    async def running(self, session_id: str) -> None:
        response = await self._client.post(f"/internal/sessions/{session_id}/running")
        self._raise_for_error(response)

    async def paused(self, session_id: str) -> None:
        response = await self._client.post(f"/internal/sessions/{session_id}/paused")
        self._raise_for_error(response)

    async def heartbeat(self, session_id: str) -> None:
        response = await self._client.post(f"/internal/sessions/{session_id}/heartbeat")
        self._raise_for_error(response)

    async def ended(
        self,
        session_id: str,
        error_code: str | None,
        end_reason: str | None = None,
    ) -> None:
        payload: dict[str, object] = {}
        if error_code:
            payload["error_code"] = error_code
        if end_reason:
            payload["end_reason"] = end_reason
        response = await self._client.post(
            f"/internal/sessions/{session_id}/ended", json=payload
        )
        self._raise_for_error(response)

    def _raise_for_error(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        message = self._extract_error_message(response)
        if response.status_code == httpx.codes.UNAUTHORIZED:
            raise CoordinatorAuthError(message)
        if response.status_code == httpx.codes.NOT_FOUND:
            raise CoordinatorNotFoundError(message)
        raise CoordinatorError(message)

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        try:
            data = response.json()
            if isinstance(data, dict) and "error" in data:
                return str(data["error"])
        except json.JSONDecodeError:
            pass
        return f"coordinator request failed with status {response.status_code}"
