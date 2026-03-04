from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from wm_worker.models import Assignment


class CoordinatorError(RuntimeError):
    pass


class CoordinatorAuthError(CoordinatorError):
    pass


class CoordinatorNotFoundError(CoordinatorError):
    pass


@dataclass(slots=True)
class HeartbeatResult:
    cancel_active_session: bool


class CoordinatorClient:
    def __init__(
        self,
        *,
        base_url: str,
        worker_id: str,
        worker_internal_token: str,
        timeout_seconds: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._worker_id = worker_id
        self._headers = {"Authorization": f"Bearer {worker_internal_token}"}
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_seconds),
            headers=self._headers,
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def register_worker(self) -> None:
        payload = {"worker_id": self._worker_id}
        response = await self._client.post("/internal/v1/worker/register", json=payload)
        self._raise_for_error(response)

    async def heartbeat_worker(self) -> HeartbeatResult:
        payload = {"worker_id": self._worker_id}
        response = await self._client.post("/internal/v1/worker/heartbeat", json=payload)
        self._raise_for_error(response)
        body = self._parse_json(response)
        return HeartbeatResult(cancel_active_session=bool(body.get("cancel_active_session")))

    async def poll_assignment(self) -> Assignment | None:
        response = await self._client.get(
            "/internal/v1/worker/assignment", params={"worker_id": self._worker_id}
        )
        if response.status_code == httpx.codes.NO_CONTENT:
            return None
        if response.status_code == httpx.codes.CONFLICT:
            return None
        self._raise_for_error(response)
        body = self._parse_json(response)
        return Assignment(
            session_id=str(body["session_id"]),
            room_name=str(body["room_name"]),
            worker_access_token=str(body["worker_access_token"]),
            video_track_name=str(body["video_track_name"]),
            control_topic=str(body["control_topic"]),
        )

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
    def _parse_json(response: httpx.Response) -> dict[str, Any]:
        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise CoordinatorError(f"invalid JSON from coordinator: {exc}") from exc
        if not isinstance(body, dict):
            raise CoordinatorError("expected JSON object from coordinator")
        return body

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        try:
            data = response.json()
            if isinstance(data, dict) and "error" in data:
                return str(data["error"])
        except json.JSONDecodeError:
            pass
        return f"coordinator request failed with status {response.status_code}"
