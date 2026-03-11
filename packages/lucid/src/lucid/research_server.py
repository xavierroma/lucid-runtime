from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .capabilities import DEFAULT_CONTROL_TOPIC, DEFAULT_STATUS_TOPIC, capabilities
from .config import ConfigError, RuntimeConfig
from .discovery import ensure_model_module_loaded
from .host import SessionLifecycleHooks, SessionRunner
from .livekit import mint_access_token
from .types import Assignment


class SessionState(str, Enum):
    STARTING = "STARTING"
    READY = "READY"
    RUNNING = "RUNNING"
    CANCELING = "CANCELING"
    ENDED = "ENDED"
    FAILED = "FAILED"


class SessionRecord(BaseModel):
    session_id: str
    room_name: str
    state: SessionState
    error_code: str | None = None
    end_reason: str | None = None


class SessionResponse(BaseModel):
    session: SessionRecord
    client_access_token: str | None = None
    capabilities: dict[str, Any]


@dataclass(slots=True)
class _SessionTask:
    record: SessionRecord
    task: asyncio.Task[None]
    client_requested_end: bool = False


class ResearchSessionService:
    def __init__(
        self,
        host_config: RuntimeConfig,
        logger: logging.Logger,
        *,
        livekit_factory=None,
    ) -> None:
        ensure_model_module_loaded()
        self._host_config = host_config
        self._logger = logger
        self._runner = SessionRunner(
            host_config,
            None,
            logger,
            livekit_factory=livekit_factory,
            lifecycle_hooks=SessionLifecycleHooks(
                on_ready=self._mark_ready,
                on_running=self._mark_running,
            ),
        )
        self._active: _SessionTask | None = None
        self._lock = asyncio.Lock()

    async def startup(self) -> None:
        await self._runner.load()

    async def shutdown(self) -> None:
        self._runner.stop()
        if self._active is not None:
            await asyncio.gather(self._active.task, return_exceptions=True)
        await self._runner.close()

    async def create_session(self) -> SessionResponse:
        async with self._lock:
            if self._active is not None and self._active.record.state not in {
                SessionState.ENDED,
                SessionState.FAILED,
            }:
                raise HTTPException(status_code=409, detail="active session in progress")

            api_key = os.getenv("LIVEKIT_API_KEY", "").strip()
            api_secret = os.getenv("LIVEKIT_API_SECRET", "").strip()
            if not api_key or not api_secret:
                raise HTTPException(
                    status_code=500,
                    detail="LIVEKIT_API_KEY and LIVEKIT_API_SECRET are required for research mode",
                )

            session_id = str(uuid4())
            room_name = f"wm-{session_id}"
            worker_token = mint_access_token(
                api_key=api_key,
                api_secret=api_secret,
                identity=f"lucid-worker-{session_id}",
                room_name=room_name,
            )
            client_token = mint_access_token(
                api_key=api_key,
                api_secret=api_secret,
                identity=f"lucid-client-{session_id}",
                room_name=room_name,
            )
            record = SessionRecord(
                session_id=session_id,
                room_name=room_name,
                state=SessionState.STARTING,
            )
            assignment = Assignment(
                session_id=session_id,
                room_name=room_name,
                worker_access_token=worker_token,
                control_topic=DEFAULT_CONTROL_TOPIC,
            )
            task = asyncio.create_task(
                self._run_session(record, assignment),
                name=f"research-session:{session_id}",
            )
            self._active = _SessionTask(record=record, task=task)
            return SessionResponse(
                session=record.model_copy(),
                client_access_token=client_token,
                capabilities=capabilities(
                    control_topic=DEFAULT_CONTROL_TOPIC,
                    status_topic=DEFAULT_STATUS_TOPIC,
                ),
            )

    async def get_session(self, session_id: str) -> SessionResponse:
        async with self._lock:
            if self._active is None or self._active.record.session_id != session_id:
                raise HTTPException(status_code=404, detail="session not found")
            return SessionResponse(
                session=self._active.record.model_copy(),
                client_access_token=None,
                capabilities=capabilities(
                    control_topic=DEFAULT_CONTROL_TOPIC,
                    status_topic=DEFAULT_STATUS_TOPIC,
                ),
            )

    async def end_session(self, session_id: str) -> None:
        async with self._lock:
            if self._active is None or self._active.record.session_id != session_id:
                raise HTTPException(status_code=404, detail="session not found")
            if self._active.record.state in {SessionState.ENDED, SessionState.FAILED}:
                return
            self._active.record.state = SessionState.CANCELING
            self._active.client_requested_end = True
            self._runner.stop()

    async def _run_session(self, record: SessionRecord, assignment: Assignment) -> None:
        try:
            result = await self._runner.run_session(assignment)
            if result.error_code:
                record.state = SessionState.FAILED
                record.error_code = result.error_code
                record.end_reason = "WORKER_REPORTED_ERROR"
            else:
                record.state = SessionState.ENDED
                if self._active is not None and self._active.client_requested_end:
                    record.end_reason = "CLIENT_REQUESTED"
                elif result.ended_by_control:
                    record.end_reason = "CONTROL_REQUESTED"
                else:
                    record.end_reason = "NORMAL_COMPLETION"
        except Exception as exc:
            self._logger.exception("research session failed: %s", exc)
            record.state = SessionState.FAILED
            record.error_code = "MODEL_RUNTIME_ERROR"
            record.end_reason = "WORKER_REPORTED_ERROR"

    async def _mark_ready(self, session_id: str) -> None:
        async with self._lock:
            if self._active is None or self._active.record.session_id != session_id:
                return
            if self._active.record.state != SessionState.STARTING:
                return
            self._active.record.state = SessionState.READY

    async def _mark_running(self, session_id: str) -> None:
        async with self._lock:
            if self._active is None or self._active.record.session_id != session_id:
                return
            if self._active.record.state not in {SessionState.STARTING, SessionState.READY}:
                return
            self._active.record.state = SessionState.RUNNING


def create_app(service: ResearchSessionService) -> FastAPI:
    app = FastAPI()

    @app.on_event("startup")
    async def _startup() -> None:
        await service.startup()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await service.shutdown()

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/sessions", response_model=SessionResponse)
    async def create_session() -> SessionResponse:
        return await service.create_session()

    @app.get("/sessions/{session_id}", response_model=SessionResponse)
    async def get_session(session_id: str) -> SessionResponse:
        return await service.get_session(session_id)

    @app.post("/sessions/{session_id}:end")
    async def end_session(session_id: str) -> dict[str, str]:
        await service.end_session(session_id)
        return {"status": "ok"}

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Lucid research session server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        ensure_model_module_loaded()
        host_config = RuntimeConfig.from_env()
    except (ConfigError, RuntimeError) as exc:
        logging.getLogger("lucid.research").error("invalid configuration: %s", exc)
        return 2

    service = ResearchSessionService(host_config, logging.getLogger("lucid.research"))
    app = create_app(service)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
    return 0


if __name__ == "__main__":
    sys.exit(main())
