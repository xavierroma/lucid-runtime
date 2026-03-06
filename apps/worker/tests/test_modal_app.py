from __future__ import annotations

import base64
import json
from types import SimpleNamespace

from wm_worker.modal_app import ModalSessionDispatcher, _mint_worker_access_token
from wm_worker.modal_dispatch_api import FunctionCallStatus, LaunchRequest


def _decode_payload(token: str) -> dict[str, object]:
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))


def test_mint_worker_access_token_uses_camel_case_video_grant(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LIVEKIT_API_KEY", "api-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "api-secret")

    token = _mint_worker_access_token(
        room_name="wm-session-1",
        session_id="session-1",
        worker_id="wm-worker-1",
    )

    assert token is not None
    payload = _decode_payload(token)
    assert payload["iss"] == "api-key"
    assert payload["sub"] == "wm-worker-1-session-1"
    assert payload["video"] == {
        "roomJoin": True,
        "room": "wm-session-1",
    }


def test_modal_dispatcher_launch_uses_spawn_session(monkeypatch) -> None:
    calls: list[LaunchRequest] = []

    def fake_spawn(payload: LaunchRequest) -> str:
        calls.append(payload)
        return "fc-123"

    monkeypatch.setattr("wm_worker.modal_app._spawn_session", fake_spawn)

    dispatcher = ModalSessionDispatcher()
    function_call_id = dispatcher.launch(
        LaunchRequest(
            session_id="session-1",
            room_name="wm-session-1",
            worker_access_token="worker-token",
            coordinator_base_url="https://coord.example.com",
            coordinator_internal_token="secret",
        )
    )

    assert function_call_id == "fc-123"
    assert [call.session_id for call in calls] == ["session-1"]


def test_modal_dispatcher_cancel_passes_force(monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []

    class FakeFunctionCall:
        def cancel(self, *, terminate_containers: bool = False) -> None:
            calls.append(("cancel", terminate_containers))

    monkeypatch.setattr(
        "wm_worker.modal_app.modal.FunctionCall.from_id",
        lambda function_call_id: calls.append((function_call_id, False)) or FakeFunctionCall(),
    )

    dispatcher = ModalSessionDispatcher()
    dispatcher.cancel("fc-123", force=True)

    assert calls == [("fc-123", False), ("cancel", True)]


def test_modal_dispatcher_status_reads_call_graph(monkeypatch) -> None:
    class FakeFunctionCall:
        def get_call_graph(self):
            return [SimpleNamespace(status="InputStatus.FAILURE", parent_input_id=None)]

    monkeypatch.setattr(
        "wm_worker.modal_app.modal.FunctionCall.from_id",
        lambda _function_call_id: FakeFunctionCall(),
    )

    dispatcher = ModalSessionDispatcher()
    assert dispatcher.status("fc-123") == FunctionCallStatus.FAILURE
