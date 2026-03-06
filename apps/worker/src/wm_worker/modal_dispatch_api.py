from __future__ import annotations

from enum import Enum
from typing import Protocol

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field


class LaunchRequest(BaseModel):
    session_id: str
    room_name: str
    worker_access_token: str
    video_track_name: str = Field(default="main_video")
    control_topic: str = Field(default="wm.control.v1")
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


def create_app(dispatcher: SessionDispatcher, dispatch_token: str) -> FastAPI:
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
        function_call_id = dispatcher.launch(payload)
        return LaunchResponse(function_call_id=function_call_id)

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
