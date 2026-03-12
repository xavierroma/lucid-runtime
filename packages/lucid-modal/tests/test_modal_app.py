from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from pathlib import Path

from lucid_modal import (
    FunctionCallStatus,
    LaunchRequest,
    ModalSessionDispatcher,
    mint_worker_access_token,
    spawn_session_call,
    with_lucid_runtime,
)


def _decode_payload(token: str) -> dict[str, object]:
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))


def test_mint_worker_access_token_uses_camel_case_video_grant(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LIVEKIT_API_KEY", "api-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "api-secret")

    token = mint_worker_access_token(
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

    dispatcher = ModalSessionDispatcher(fake_spawn)
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
        "lucid_modal.app.modal.FunctionCall.from_id",
        lambda function_call_id: calls.append((function_call_id, False)) or FakeFunctionCall(),
    )

    dispatcher = ModalSessionDispatcher(lambda _payload: "unused")
    dispatcher.cancel("fc-123", force=True)

    assert calls == [("fc-123", False), ("cancel", True)]


def test_modal_dispatcher_status_reads_call_graph(monkeypatch) -> None:
    class FakeFunctionCall:
        def get_call_graph(self):
            return [SimpleNamespace(status="InputStatus.FAILURE", parent_input_id=None)]

    monkeypatch.setattr(
        "lucid_modal.app.modal.FunctionCall.from_id",
        lambda _function_call_id: FakeFunctionCall(),
    )

    dispatcher = ModalSessionDispatcher(lambda _payload: "unused")
    assert dispatcher.status("fc-123") == FunctionCallStatus.FAILURE


def test_modal_dispatcher_status_prefers_matching_child_call(monkeypatch) -> None:
    class FakeFunctionCall:
        def get_call_graph(self):
            return [
                SimpleNamespace(
                    function_call_id="fc-root",
                    status="InputStatus.SUCCESS",
                    parent_input_id=None,
                    children=[
                        SimpleNamespace(
                            function_call_id="fc-target",
                            status="InputStatus.PENDING",
                            parent_input_id="in-root",
                            children=[],
                        )
                    ],
                )
            ]

    monkeypatch.setattr(
        "lucid_modal.app.modal.FunctionCall.from_id",
        lambda _function_call_id: FakeFunctionCall(),
    )

    dispatcher = ModalSessionDispatcher(lambda _payload: "unused")
    assert dispatcher.status("fc-target") == FunctionCallStatus.PENDING


def test_spawn_session_call_looks_up_worker_class(monkeypatch) -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    class FakeSpawnMethod:
        def spawn(self, payload: dict[str, object]):
            calls.append(("spawn", "unused", payload))
            return SimpleNamespace(object_id="fc-456")

    class FakeWorkerInstance:
        def __init__(self) -> None:
            self.run_session = FakeSpawnMethod()

    class FakeWorkerClass:
        def __call__(self) -> FakeWorkerInstance:
            calls.append(("call", "unused", {}))
            return FakeWorkerInstance()

    def fake_from_name(app_name: str, name: str):
        calls.append(("lookup", f"{app_name}:{name}", {}))
        return FakeWorkerClass()

    monkeypatch.setattr(
        "lucid_modal.app.modal",
        SimpleNamespace(Cls=SimpleNamespace(from_name=fake_from_name)),
    )

    function_call_id = spawn_session_call(
        app_name="app-1",
        worker_cls_name="WarmSessionWorker",
        payload=LaunchRequest(
            session_id="session-1",
            room_name="room-1",
            worker_access_token="worker-token",
            coordinator_base_url="https://coord.example.com",
            coordinator_internal_token="secret",
        ),
    )

    assert function_call_id == "fc-456"
    assert calls[0] == ("lookup", "app-1:WarmSessionWorker", {})
    assert calls[1][0] == "call"
    assert calls[2][0] == "spawn"


def test_with_lucid_runtime_uses_absolute_package_and_extra_dirs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    extra_dir = tmp_path / "demo"
    extra_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    class FakeImage:
        def __init__(self) -> None:
            self.local_dirs: list[tuple[str, str]] = []
            self.commands: list[str] = []

        def apt_install(self, *_args):
            return self

        def add_local_dir(self, src: str, dest: str, *, copy: bool, ignore):
            assert copy is True
            assert ignore is not None
            self.local_dirs.append((src, dest))
            return self

        def run_commands(self, *commands: str):
            self.commands.extend(commands)
            return self

    image = with_lucid_runtime(
        FakeImage(),
        extra_local_dirs=[("demo", "/workspace/demo")],
    )

    assert Path(image.local_dirs[0][0]).is_absolute()
    assert Path(image.local_dirs[0][0]).exists()
    assert Path(image.local_dirs[1][0]).is_absolute()
    assert Path(image.local_dirs[1][0]).exists()
    assert image.local_dirs[2] == (str(extra_dir.resolve()), "/workspace/demo")
    assert image.commands == [
        "python -m pip install '/workspace/packages/lucid[livekit]'",
        "python -m pip install /workspace/packages/lucid-modal",
    ]
