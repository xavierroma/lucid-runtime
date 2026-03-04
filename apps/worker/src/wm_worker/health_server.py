from __future__ import annotations

from dataclasses import dataclass

from aiohttp import web


@dataclass(slots=True)
class HealthState:
    alive: bool = True
    ready: bool = False


class HealthServer:
    def __init__(self, port: int, state: HealthState) -> None:
        self._port = port
        self._state = state
        self._runner: web.AppRunner | None = None

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/healthz", self._healthz)
        app.router.add_get("/readyz", self._readyz)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, host="0.0.0.0", port=self._port)
        await site.start()

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def _healthz(self, _request: web.Request) -> web.Response:
        if not self._state.alive:
            return web.json_response({"status": "unhealthy"}, status=503)
        return web.json_response({"status": "ok"}, status=200)

    async def _readyz(self, _request: web.Request) -> web.Response:
        if not self._state.ready:
            return web.json_response({"status": "not_ready"}, status=503)
        return web.json_response({"status": "ready"}, status=200)
