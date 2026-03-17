from __future__ import annotations

import inspect
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Awaitable, Callable, Protocol, Self, TypeAlias

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ..controlplane import CoordinatorClient
from ..core.runtime import ModelConfigInput
from ..core.spec import ModelTarget, build_model_definition, resolve_model_definition
from ..livekit import (
    Assignment,
    LucidRuntime,
    RuntimeConfig,
    SessionConfig,
    SessionRunner,
    mint_access_token,
)

from .config import load_runtime_config_from_env

try:  # pragma: no cover - depends on optional dependency
    import modal  # type: ignore
except Exception:  # pragma: no cover - depends on optional dependency
    class _MissingFunctionCall:
        @staticmethod
        def from_id(_function_call_id: str):
            raise RuntimeError("modal package is missing; install lucid-runtime[modal]")

    class _MissingModal:
        FunctionCall = _MissingFunctionCall

        def __getattr__(self, _name: str) -> object:
            raise RuntimeError("modal package is missing; install lucid-runtime[modal]")

    modal = _MissingModal()  # type: ignore[assignment]

RuntimeSetup = Callable[[LucidRuntime, logging.Logger], Awaitable[None] | None]
ModelConfigLoader = Callable[[], ModelConfigInput]
RuntimeConfigLoader = Callable[[], RuntimeConfig]
SpawnSessionFn: TypeAlias = Callable[["LaunchRequest"], str]

_LOCAL_IGNORE_PARTS = {"__pycache__", ".pytest_cache", ".venv", "build", "dist"}


class ModalImage(Protocol):
    def apt_install(self, *packages: str) -> Self: ...
    def add_local_dir(
        self,
        src: str,
        dest: str,
        *,
        copy: bool,
        ignore: Callable[[Path], bool],
    ) -> Self: ...
    def run_commands(self, *commands: str) -> Self: ...


class _CallGraphNode(Protocol):
    function_call_id: str | None
    status: object
    parent_input_id: str | None
    children: list["_CallGraphNode"]


def ignore_local_artifacts(path: Path) -> bool:
    if any(part in _LOCAL_IGNORE_PARTS for part in path.parts):
        return True
    if any(part.endswith(".egg-info") for part in path.parts):
        return True
    return path.suffix in {".pyc", ".pyo"}


def _package_project_root(path: str | Path) -> Path:
    return Path(path).resolve().parents[2]


def _resolve_local_dir(path: str | Path) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    if candidate.exists():
        return str(candidate.resolve())
    repo_candidate = Path(__file__).resolve().parents[4] / candidate
    if repo_candidate.exists():
        return str(repo_candidate.resolve())
    return str(candidate.resolve())


def with_lucid_runtime(
    image: ModalImage,
    *,
    include_livekit: bool = True,
    extra_local_dirs: list[tuple[str, str]] | None = None,
) -> ModalImage:
    lucid_project_root = _package_project_root(inspect.getfile(build_model_definition))
    updated = (
        image.apt_install("ffmpeg", "ca-certificates")
        .add_local_dir(
            str(lucid_project_root),
            "/workspace/packages/lucid",
            copy=True,
            ignore=ignore_local_artifacts,
        )
    )
    for src, dest in extra_local_dirs or []:
        updated = updated.add_local_dir(
            _resolve_local_dir(src),
            dest,
            copy=True,
            ignore=ignore_local_artifacts,
        )
    if include_livekit:
        commands = ["python -m pip install '/workspace/packages/lucid[livekit,modal]'"]
    else:
        commands = ["python -m pip install '/workspace/packages/lucid[modal]'"]
    return updated.run_commands(*commands)


def env_secret(*names: str) -> object:
    payload: dict[str, str] = {}
    for name in names:
        value = os.getenv(name)
        if value:
            payload[name] = value
    return modal.Secret.from_dict(payload)


def build_modal_volume_commit_hook(
    *,
    volume_env: str = "MODAL_HF_CACHE_VOLUME",
    root_env: str = "MODAL_COMPILER_CACHE_ROOT",
) -> Callable[[logging.Logger, str], bool]:
    def _commit(logger: logging.Logger, reason: str) -> bool:
        volume_name = os.getenv(volume_env, "").strip()
        cache_root = os.getenv(root_env, "").strip()
        if not volume_name or not cache_root or not Path(cache_root).exists():
            return False
        try:
            modal.Volume.from_name(volume_name).commit()
        except Exception as exc:
            logger.warning(
                "modal.volume.commit_failed volume=%s root=%s reason=%s error_type=%s",
                volume_name,
                cache_root,
                reason,
                exc.__class__.__name__,
            )
            return False
        logger.info(
            "modal.volume.committed volume=%s root=%s reason=%s",
            volume_name,
            cache_root,
            reason,
        )
        return True

    return _commit


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


def create_dispatch_api(dispatcher: SessionDispatcher, dispatch_token: str, manifest: dict) -> FastAPI:
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

    @app.get("/manifest")
    def get_manifest(_auth: None = Depends(_authorize)) -> dict:
        return manifest

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


def _normalize_function_call_status(raw_status: object) -> FunctionCallStatus:
    normalized = getattr(raw_status, "name", None) or str(raw_status)
    normalized = normalized.split(".")[-1].upper()
    try:
        return FunctionCallStatus(normalized)
    except ValueError:
        return FunctionCallStatus.PENDING


def _find_call_graph_node(
    nodes: list[_CallGraphNode],
    function_call_id: str,
) -> _CallGraphNode | None:
    for node in nodes:
        if getattr(node, "function_call_id", None) == function_call_id:
            return node
        child = _find_call_graph_node(list(getattr(node, "children", [])), function_call_id)
        if child is not None:
            return child
    return None


class ModalSessionDispatcher:
    def __init__(self, spawn_fn: SpawnSessionFn) -> None:
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
    app: object
    worker_cls: type[object]
    dispatcher: ModalSessionDispatcher


def create_app(
    *,
    app_name: str,
    model: ModelTarget,
    image: ModalImage,
    gpu: str,
    secrets: list[object],
    model_config_loader: ModelConfigLoader | None = None,
    runtime_setup: RuntimeSetup | None = None,
    min_containers: int = 1,
    max_containers: int = 1,
    scaledown_window_secs: int = 1200,
    timeout_seconds: int = 60 * 60 * 2,
    startup_timeout_seconds: int = 20 * 60,
    volumes: dict[str, object] | None = None,
    dispatch_token: str = "",
    runtime_config_loader: RuntimeConfigLoader = load_runtime_config_from_env,
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
            start = perf_counter()
            try:
                self._host_config = runtime_config_loader()
                self._logger.info(
                    "modal.worker.load host_config_ready elapsed_ms=%.1f app_name=%s gpu=%s startup_timeout_seconds=%s",
                    (perf_counter() - start) * 1000.0,
                    app_name,
                    gpu,
                    startup_timeout_seconds,
                )
                self._model_config = model_config_loader() if model_config_loader is not None else None
                self._runtime = LucidRuntime.load_model(
                    runtime_config=self._host_config,
                    logger=self._logger,
                    model=model,
                    config=self._model_config,
                )
                if runtime_setup is not None:
                    result = runtime_setup(self._runtime, self._logger)
                    if inspect.isawaitable(result):
                        await result
                self._logger.info(
                    "modal.worker.load runtime_selected elapsed_ms=%.1f model=%s",
                    (perf_counter() - start) * 1000.0,
                    self._runtime.definition.name,
                )
                await self._runtime.load()
            except Exception as exc:
                self._logger.error(
                    "modal.worker.load failed duration_ms=%.1f error_type=%s",
                    (perf_counter() - start) * 1000.0,
                    exc.__class__.__name__,
                )
                raise
            self._logger.info(
                "modal.worker.load complete duration_ms=%.1f model=%s",
                (perf_counter() - start) * 1000.0,
                self._runtime.definition.name,
            )

        @modal.method()
        async def run_session(self, payload: dict[str, object]) -> None:
            request = LaunchRequest.model_validate(payload)
            session_config = SessionConfig(worker_id=request.worker_id)
            reporter = CoordinatorClient(
                base_url=request.coordinator_base_url,
                worker_internal_token=request.coordinator_internal_token,
            )
            runner = SessionRunner(
                self._host_config,
                session_config,
                self._logger,
                model=model,
                model_config=self._model_config,
                reporter=reporter,
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
    def dispatch_api() -> FastAPI:
        _manifest = resolve_model_definition(model).to_manifest()
        return create_dispatch_api(
            dispatcher,
            os.getenv("MODAL_DISPATCH_TOKEN", dispatch_token),
            _manifest,
        )

    return ModalAppBundle(app=app, worker_cls=WarmSessionWorker, dispatcher=dispatcher)
