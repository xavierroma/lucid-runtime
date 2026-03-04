from __future__ import annotations

import asyncio
import time

from wm_worker.models import (
    ActionPayload,
    ActionSnapshot,
    ControlEnvelope,
    ControlMessageType,
)


def _now_ms() -> int:
    return int(time.time() * 1000)


class ActionBuffer:
    """Latest-wins action storage with prompt state."""

    def __init__(self, initial_prompt: str) -> None:
        self._prompt = initial_prompt
        self._action = ActionPayload()
        self._last_seq = 0
        self._updated_at_ms = _now_ms()
        self._lock = asyncio.Lock()

    async def apply_control(self, envelope: ControlEnvelope) -> None:
        async with self._lock:
            if envelope.seq < self._last_seq:
                return
            self._last_seq = envelope.seq
            self._updated_at_ms = envelope.ts_ms
            if envelope.type == ControlMessageType.SET_PROMPT:
                prompt = str(envelope.payload.get("prompt", "")).strip()
                if prompt:
                    self._prompt = prompt
                return
            if envelope.type != ControlMessageType.ACTION:
                return
            payload = envelope.payload
            self._action = ActionPayload(
                keys=[str(key) for key in payload.get("keys", [])][:8],
                mouse_dx=float(payload.get("mouse_dx", 0.0)),
                mouse_dy=float(payload.get("mouse_dy", 0.0)),
                actual_distance=float(payload.get("actual_distance", 1.0)),
                angular_change_rate=float(payload.get("angular_change_rate", 1.0)),
                view_rotation_speed=float(payload.get("view_rotation_speed", 1.0)),
            )

    async def snapshot(self) -> ActionSnapshot:
        async with self._lock:
            return ActionSnapshot(
                prompt=self._prompt,
                action=self._action,
                last_seq=self._last_seq,
                updated_at_ms=self._updated_at_ms,
            )
