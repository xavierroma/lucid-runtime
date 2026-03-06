from __future__ import annotations

import json
from typing import Any

import httpx


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

    async def mark_running(self, session_id: str) -> None:
        response = await self._client.post(f"/internal/v1/sessions/{session_id}/running")
        self._raise_for_error(response)

    async def mark_ended(self, session_id: str, error_code: str | None) -> None:
        payload: dict[str, Any] = {}
        if error_code:
            payload["error_code"] = error_code
        response = await self._client.post(
            f"/internal/v1/sessions/{session_id}/ended", json=payload
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
