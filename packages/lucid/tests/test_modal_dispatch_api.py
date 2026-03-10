from __future__ import annotations

from fastapi.testclient import TestClient

from lucid.modal import (
    FunctionCallStatus,
    LaunchRequest,
    SessionDispatcher,
    create_dispatch_api,
)


class StubDispatcher(SessionDispatcher):
    def __init__(self) -> None:
        self.launch_count = 0
        self.cancel_count = 0
        self.last_cancel_id: str | None = None
        self.last_cancel_force = False

    def launch(self, payload: LaunchRequest) -> str:
        self.launch_count += 1
        return f"call-{payload.session_id}"

    def cancel(self, function_call_id: str, *, force: bool = False) -> None:
        self.cancel_count += 1
        self.last_cancel_id = function_call_id
        self.last_cancel_force = force

    def status(self, function_call_id: str) -> FunctionCallStatus:
        _ = function_call_id
        return FunctionCallStatus.PENDING


def test_launch_returns_function_call_id() -> None:
    dispatcher = StubDispatcher()
    app = create_dispatch_api(dispatcher, "token-1")
    client = TestClient(app)

    response = client.post(
        "/launch",
        headers={"Authorization": "Bearer token-1"},
        json={
            "session_id": "session-1",
            "room_name": "wm-session-1",
            "worker_access_token": "worker-token",
            "control_topic": "wm.control",
            "coordinator_base_url": "https://coord.example.com",
            "coordinator_internal_token": "secret",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"function_call_id": "call-session-1"}
    assert dispatcher.launch_count == 1


def test_cancel_is_idempotent() -> None:
    dispatcher = StubDispatcher()
    app = create_dispatch_api(dispatcher, "token-1")
    client = TestClient(app)

    response = client.post(
        "/cancel",
        headers={"Authorization": "Bearer token-1"},
        json={"function_call_id": "call-1", "force": True},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert dispatcher.cancel_count == 1
    assert dispatcher.last_cancel_id == "call-1"
    assert dispatcher.last_cancel_force is True


def test_status_returns_dispatcher_status() -> None:
    dispatcher = StubDispatcher()
    app = create_dispatch_api(dispatcher, "token-1")
    client = TestClient(app)

    response = client.get(
        "/status/call-1",
        headers={"Authorization": "Bearer token-1"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "PENDING"}


def test_auth_rejection_for_launch_and_cancel() -> None:
    dispatcher = StubDispatcher()
    app = create_dispatch_api(dispatcher, "token-1")
    client = TestClient(app)

    launch = client.post(
        "/launch",
        headers={"Authorization": "Bearer wrong"},
        json={
            "session_id": "session-1",
            "room_name": "wm-session-1",
            "worker_access_token": "worker-token",
            "coordinator_base_url": "https://coord.example.com",
            "coordinator_internal_token": "secret",
        },
    )
    assert launch.status_code == 401

    cancel = client.post(
        "/cancel",
        headers={"Authorization": "Bearer wrong"},
        json={"function_call_id": "call-1"},
    )
    assert cancel.status_code == 401
