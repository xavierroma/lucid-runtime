from __future__ import annotations

import logging
from dataclasses import dataclass

from wm_worker.action_buffer import ActionBuffer
from wm_worker.models import ActionSnapshot, ControlMessageType
from wm_worker.protocol import ProtocolError, parse_control_message


@dataclass(slots=True)
class ControlOutcome:
    stop_requested: bool = False
    pong_payload: dict[str, object] | None = None


class SessionControlReducer:
    def __init__(self, initial_prompt: str, logger: logging.Logger) -> None:
        self._actions = ActionBuffer(initial_prompt)
        self._logger = logger

    async def snapshot(self) -> ActionSnapshot:
        return await self._actions.snapshot()

    async def reduce(self, raw: bytes | str, *, session_id: str) -> ControlOutcome:
        try:
            envelope = parse_control_message(raw)
        except ProtocolError as exc:
            self._logger.warning("invalid control payload: %s", exc)
            return ControlOutcome()

        if envelope.session_id and envelope.session_id != session_id:
            return ControlOutcome()
        if envelope.type == ControlMessageType.END:
            return ControlOutcome(stop_requested=True)
        if envelope.type == ControlMessageType.PING:
            return ControlOutcome(
                pong_payload={"client_ts_ms": envelope.payload.get("client_ts_ms")}
            )

        await self._actions.apply_control(envelope)
        return ControlOutcome()
