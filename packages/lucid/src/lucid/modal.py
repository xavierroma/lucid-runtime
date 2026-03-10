from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .config import RuntimeConfig, SessionConfig
from .discovery import build_model_runtime_config, ensure_model_module_loaded
from .host import SessionRunner
from .livekit import mint_access_token
from .runtime import LucidRuntime
from .types import Assignment

try:  # pragma: no cover - depends on optional dependency
    import modal  # type: ignore
except Exception:  # pragma: no cover - depends on optional dependency
    class _MissingFunctionCall:
        @staticmethod
        def from_id(_function_call_id: str):
            raise RuntimeError("modal package is missing; install lucid[modal]")

    class _MissingModal:
        FunctionCall = _MissingFunctionCall

        def __getattr__(self, _name: str) -> object:
            raise RuntimeError("modal package is missing; install lucid[modal]")

    modal = _MissingModal()  # type: ignore[assignment]


class LaunchRequest(BaseModel):
    session_id: str
    room_name: str
    worker_access_token: str
    control_topic: str = Field(default="wm.control")
    coordinator_base_url: str
    coordinator_internal_token: str
    worker_id: str = Field(default="wm-worker-1")


class LaunchResponse(BaseModel):
    function_call_id: str


class CancelRequest(BaseModel):
    function_call_id: str
    force: bool = False


class OkResponse(BaseModel):
    status: str = "ok"


class FunctionCallStatus(str, Enum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    INIT_FAILURE = "INIT_FAILURE"
    TERMINATED = "TERMINATED"
    TIMEOUT = "TIMEOUT"
    NOT_FOUND = "NOT_FOUND"


class StatusResponse(BaseModel):
    status: FunctionCallStatus


class SessionDispatcher(Protocol):
    def launch(self, payload: LaunchRequest) -> str: ...
    def cancel(self, function_call_id: str, *, force: bool = False) -> None: ...
    def status(self, function_call_id: str) -> FunctionCallStatus: ...


def create_dispatch_api(dispatcher: SessionDispatcher, dispatch_token: str) -> FastAPI:
    app = FastAPI()

    def _authorize(authorization: str | None = Header(default=None)) -> None:
        expected = dispatch_token.strip()
        if not expected:
            raise HTTPException(status_code=500, detail="dispatch token is not configured")
        token = None
        if authorization and authorization.startswith("Bearer "):
            token = authorization.removeprefix("Bearer ").strip()
        if token != expected:
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.post("/launch", response_model=LaunchResponse)
    def launch(payload: LaunchRequest, _auth: None = Depends(_authorize)) -> LaunchResponse:
        return LaunchResponse(function_call_id=dispatcher.launch(payload))

    @app.post("/cancel", response_model=OkResponse)
    def cancel(payload: CancelRequest, _auth: None = Depends(_authorize)) -> OkResponse:
        try:
            dispatcher.cancel(payload.function_call_id, force=payload.force)
        except Exception:
            pass
        return OkResponse()

    @app.get("/status/{function_call_id}", response_model=StatusResponse)
    def status(
        function_call_id: str,
        _auth: None = Depends(_authorize),
    ) -> StatusResponse:
        return StatusResponse(status=dispatcher.status(function_call_id))

    return app


def mint_worker_access_token(
    *,
    room_name: str,
    session_id: str,
    worker_id: str,
) -> str | None:
    api_key = os.getenv("LIVEKIT_API_KEY", "").strip()
    api_secret = os.getenv("LIVEKIT_API_SECRET", "").strip()
    if not api_key or not api_secret:
        return None
    return mint_access_token(
        api_key=api_key,
        api_secret=api_secret,
        identity=f"{worker_id}-{session_id}",
        room_name=room_name,
    )


def build_assignment(request: LaunchRequest) -> Assignment:
    return Assignment(
        session_id=request.session_id,
        room_name=request.room_name,
        worker_access_token=mint_worker_access_token(
            room_name=request.room_name,
            session_id=request.session_id,
            worker_id=request.worker_id,
        )
        or request.worker_access_token,
        control_topic=request.control_topic,
    )


def _normalize_function_call_status(raw_status: Any) -> FunctionCallStatus:
    normalized = getattr(raw_status, "name", None) or str(raw_status)
    normalized = normalized.split(".")[-1].upper()
    try:
        return FunctionCallStatus(normalized)
    except ValueError:
        return FunctionCallStatus.PENDING


def _find_call_graph_node(nodes: list[Any], function_call_id: str) -> Any | None:
    for node in nodes:
        if getattr(node, "function_call_id", None) == function_call_id:
            return node
        child = _find_call_graph_node(list(getattr(node, "children", [])), function_call_id)
        if child is not None:
            return child
    return None


class ModalSessionDispatcher:
    def __init__(self, spawn_fn) -> None:
        self._spawn_fn = spawn_fn

    def launch(self, payload: LaunchRequest) -> str:
        return self._spawn_fn(payload)

    def cancel(self, function_call_id: str, *, force: bool = False) -> None:
        modal.FunctionCall.from_id(function_call_id).cancel(terminate_containers=force)

    def status(self, function_call_id: str) -> FunctionCallStatus:
        try:
            graph = modal.FunctionCall.from_id(function_call_id).get_call_graph()
        except Exception as exc:
            if exc.__class__.__name__ == "NotFoundError":
                return FunctionCallStatus.NOT_FOUND
            raise
        if not graph:
            return FunctionCallStatus.PENDING
        matched = _find_call_graph_node(graph, function_call_id)
        if matched is not None:
            return _normalize_function_call_status(getattr(matched, "status", "PENDING"))
        root = next(
            (
                node
                for node in graph
                if getattr(node, "parent_input_id", None) in (None, "")
            ),
            graph[0],
        )
        return _normalize_function_call_status(getattr(root, "status", "PENDING"))


def spawn_session_call(
    *,
    app_name: str,
    worker_cls_name: str,
    payload: LaunchRequest,
) -> str:
    worker_cls = modal.Cls.from_name(app_name, worker_cls_name)
    function_call = worker_cls().run_session.spawn(payload.model_dump())
    function_call_id = getattr(function_call, "object_id", "")
    return function_call_id or str(function_call)


@dataclass(frozen=True, slots=True)
class ModalAppBundle:
    app: Any
    worker_cls: type[Any]
    dispatcher: ModalSessionDispatcher


def create_app(
    *,
    app_name: str,
    image,
    gpu: str,
    secrets: list[Any],
    min_containers: int = 1,
    max_containers: int = 1,
    scaledown_window_secs: int = 1200,
    timeout_seconds: int = 60 * 60 * 2,
    startup_timeout_seconds: int = 20 * 60,
    volumes: dict[str, Any] | None = None,
    dispatch_token: str = "",
    runtime_config_loader=RuntimeConfig.from_env,
    logger_name: str = "lucid.modal",
) -> ModalAppBundle:
    app = modal.App(app_name)
    volumes = volumes or {}

    def _build_logger() -> logging.Logger:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        return logger

    @app.cls(
        image=image,
        gpu=gpu,
        serialized=True,
        timeout=timeout_seconds,
        startup_timeout=startup_timeout_seconds,
        min_containers=min_containers,
        max_containers=max_containers,
        scaledown_window=scaledown_window_secs,
        volumes=volumes,
        secrets=secrets,
    )
    class WarmSessionWorker:
        @modal.enter()
        async def load(self) -> None:
            self._logger = _build_logger()
            self._host_config = runtime_config_loader()
            ensure_model_module_loaded()
            self._runtime_config = build_model_runtime_config(self._host_config)
            self._runtime = LucidRuntime.load_selected(
                runtime_config=self._runtime_config,
                logger=self._logger,
                model_name=os.getenv("WM_MODEL_NAME", "").strip() or None,
            )
            await self._runtime.load()

        @modal.method()
        async def run_session(self, payload: dict[str, Any]) -> None:
            request = LaunchRequest.model_validate(payload)
            session_config = SessionConfig.from_values(
                worker_id=request.worker_id,
                coordinator_base_url=request.coordinator_base_url,
                worker_internal_token=request.coordinator_internal_token,
            )
            runner = SessionRunner(
                self._host_config,
                session_config,
                self._logger,
                runtime_config=self._runtime_config,
                runtime=self._runtime,
            )
            try:
                await runner.run_session(build_assignment(request))
            finally:
                await runner.close()

    worker_cls_name = getattr(WarmSessionWorker, "_get_name", lambda: "WarmSessionWorker")()

    def _spawn_session(payload: LaunchRequest) -> str:
        return spawn_session_call(
            app_name=app_name,
            worker_cls_name=worker_cls_name,
            payload=payload,
        )

    dispatcher = ModalSessionDispatcher(_spawn_session)

    @app.function(image=image, secrets=secrets, serialized=True)
    @modal.asgi_app()
    def dispatch_api():
        return create_dispatch_api(
            dispatcher,
            os.getenv("MODAL_DISPATCH_TOKEN", dispatch_token),
        )

    return ModalAppBundle(app=app, worker_cls=WarmSessionWorker, dispatcher=dispatcher)
