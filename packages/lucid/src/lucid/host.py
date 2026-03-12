from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .config import RuntimeConfig, SessionConfig
from .coordinator import CoordinatorClient
from .discovery import ModelTarget
from .livekit import (
    FakeLiveKitAdapter,
    LiveKitAdapter,
    OutputRouter,
    RealLiveKitAdapter,
    SessionControlReducer,
    SessionStatusPublisher,
)
from .runtime import LucidError, LucidRuntime
from .types import Assignment, SessionResult


@dataclass(frozen=True, slots=True)
class SessionLifecycleHooks:
    on_ready: Callable[[str], Any] | None = None
    on_running: Callable[[str], Any] | None = None


class SessionRunner:
    HEARTBEAT_INTERVAL_SECS = 2.0
    INITIAL_INPUT_GRACE_SECS = 0.5

    def __init__(
        self,
        host_config: RuntimeConfig,
        session_config: SessionConfig | None,
        logger: logging.Logger,
        *,
        model: ModelTarget,
        model_config: Any | None = None,
        runtime_config: Any | None = None,
        coordinator: CoordinatorClient | None = None,
        livekit_factory: Callable[[], LiveKitAdapter] | None = None,
        lifecycle_hooks: SessionLifecycleHooks | None = None,
        runtime: LucidRuntime | None = None,
    ) -> None:
        self._model = model
        self._model_config = model_config
        self._host_config = host_config
        self._session_config = session_config
        self._logger = logger
        self._coordinator = coordinator
        if self._coordinator is None and session_config is not None:
            self._coordinator = CoordinatorClient(
                base_url=session_config.coordinator_base_url,
                worker_internal_token=session_config.worker_internal_token,
            )
        self._runtime_config = runtime_config or host_config
        self._runtime = runtime or LucidRuntime.load_model(
            runtime_config=self._runtime_config,
            logger=logger,
            model=model,
            config=model_config,
        )
        self._owns_runtime = runtime is None
        self._livekit_factory = livekit_factory
        self._lifecycle_hooks = lifecycle_hooks or SessionLifecycleHooks()
        self._session_stop_event = asyncio.Event()
        self._active_session_ctx = None
        self._loaded = False

    @property
    def manifest(self) -> dict[str, object]:
        return self._runtime.manifest()

    @property
    def output_bindings(self) -> list[dict[str, object]]:
        return self._runtime.output_bindings()

    async def load(self) -> None:
        if self._loaded:
            return
        start = perf_counter()
        try:
            await self._runtime.load()
        except Exception as exc:
            self._logger.error(
                "session_runner.load failed duration_ms=%.1f model=%s error_type=%s",
                (perf_counter() - start) * 1000.0,
                self._runtime.definition.name,
                exc.__class__.__name__,
            )
            raise
        self._loaded = True
        self._logger.info(
            "session_runner.load complete duration_ms=%.1f model=%s",
            (perf_counter() - start) * 1000.0,
            self._runtime.definition.name,
        )

    async def close(self) -> None:
        if self._coordinator is not None:
            await self._coordinator.close()
        if self._owns_runtime:
            await self._runtime.unload()

    def stop(self) -> None:
        if self._active_session_ctx is not None:
            self._active_session_ctx.running = False
        self._session_stop_event.set()

    async def run_session(self, assignment: Assignment) -> SessionResult:
        if not self._loaded:
            await self.load()

        self._session_stop_event = asyncio.Event()
        livekit = self._build_livekit_adapter()
        output_router = OutputRouter(
            outputs=self._runtime.outputs,
            livekit=livekit,
            target_fps=self._host_config.target_fps,
            max_queue_frames=self._host_config.max_queue_frames,
            frame_width=self._host_config.frame_width,
            frame_height=self._host_config.frame_height,
        )
        session_ctx = self._runtime.create_session_context(
            session_id=assignment.session_id,
            room_name=assignment.room_name,
            publish_fn=output_router.publish,
            metrics_fn=output_router.snapshot,
        )
        self._active_session_ctx = session_ctx
        control = SessionControlReducer(self._runtime, session_ctx, self._logger)
        status = SessionStatusPublisher(
            livekit=livekit,
            session_id=assignment.session_id,
            logger=self._logger,
        )
        result = SessionResult(error_code=None, ended_by_control=False)

        self._logger.info(
            "starting session session_id=%s room_name=%s model=%s",
            assignment.session_id,
            assignment.room_name,
            self._runtime.definition.name,
        )
        start = perf_counter()
        try:
            await livekit.connect(assignment, self._runtime.outputs)
            self._logger.info(
                "session_runner.run_session livekit_connected elapsed_ms=%.1f session_id=%s",
                (perf_counter() - start) * 1000.0,
                assignment.session_id,
            )
            await status.started(
                self._session_config.worker_id
                if self._session_config is not None
                else "lucid-research"
            )
            self._logger.info(
                "session_runner.run_session status_published elapsed_ms=%.1f session_id=%s",
                (perf_counter() - start) * 1000.0,
                assignment.session_id,
            )
            if self._coordinator is not None:
                await self._coordinator.mark_ready(assignment.session_id)
                self._logger.info(
                    "session_runner.run_session coordinator_marked_ready elapsed_ms=%.1f session_id=%s",
                    (perf_counter() - start) * 1000.0,
                    assignment.session_id,
                )
            await self._emit_lifecycle_hook(
                self._lifecycle_hooks.on_ready,
                assignment.session_id,
            )
            self._logger.info(
                "session_runner.run_session ready_for_start elapsed_ms=%.1f session_id=%s",
                (perf_counter() - start) * 1000.0,
                assignment.session_id,
            )

            tasks = [
                asyncio.create_task(
                    self._control_loop(
                        control=control,
                        session_id=assignment.session_id,
                        status=status,
                        livekit=livekit,
                        result=result,
                        session_ctx=session_ctx,
                    ),
                    name="control_loop",
                ),
                asyncio.create_task(
                    self._model_loop(session_ctx=session_ctx),
                    name="model_loop",
                ),
                *output_router.start(self._session_stop_event),
                asyncio.create_task(
                    self._metrics_loop(
                        output_router=output_router,
                        status=status,
                        session_ctx=session_ctx,
                    ),
                    name="metrics_loop",
                ),
            ]
            if self._coordinator is not None:
                tasks.append(
                    asyncio.create_task(
                        self._heartbeat_loop(session_id=assignment.session_id),
                        name="heartbeat_loop",
                    )
                )
            self._logger.info(
                "session_runner.run_session loops_started elapsed_ms=%.1f session_id=%s task_count=%s",
                (perf_counter() - start) * 1000.0,
                assignment.session_id,
                len(tasks),
            )
            await self._watch_session_tasks(tasks, result, session_ctx)
        except LucidError as exc:
            result.error_code = "MODEL_RUNTIME_ERROR"
            self._logger.error(
                "session_runner.run_session failed duration_ms=%.1f session_id=%s error_code=%s error_type=%s",
                (perf_counter() - start) * 1000.0,
                assignment.session_id,
                result.error_code,
                exc.__class__.__name__,
            )
            self._logger.exception("lucid runtime error: %s", exc)
        except Exception as exc:  # pragma: no cover - integration boundary
            if self._is_modal_input_cancellation(exc):
                self._logger.info("session cancelled by modal input cancellation")
            else:
                result.error_code = "LIVEKIT_DISCONNECT"
            self._logger.error(
                "session_runner.run_session failed duration_ms=%.1f session_id=%s error_code=%s error_type=%s",
                (perf_counter() - start) * 1000.0,
                assignment.session_id,
                result.error_code,
                exc.__class__.__name__,
            )
            self._logger.exception("session failed: %s", exc)
        finally:
            session_ctx.running = False
            self._active_session_ctx = None
            await self._runtime.close_session(session_ctx)
            self._logger.info(
                "session_runner.run_session model_session_ended elapsed_ms=%.1f session_id=%s error_code=%s",
                (perf_counter() - start) * 1000.0,
                assignment.session_id,
                result.error_code,
            )
            await self._emit_terminal_status(status, result, output_router, session_ctx)
            self._logger.info(
                "session_runner.run_session terminal_status_emitted elapsed_ms=%.1f session_id=%s",
                (perf_counter() - start) * 1000.0,
                assignment.session_id,
            )
            await livekit.disconnect()
            self._logger.info(
                "session_runner.run_session livekit_disconnected elapsed_ms=%.1f session_id=%s",
                (perf_counter() - start) * 1000.0,
                assignment.session_id,
            )
            if self._coordinator is not None:
                try:
                    end_reason = None
                    if result.error_code:
                        end_reason = "WORKER_REPORTED_ERROR"
                    elif result.ended_by_control:
                        end_reason = "CONTROL_REQUESTED"
                    await self._coordinator.mark_ended(
                        assignment.session_id,
                        result.error_code,
                        end_reason,
                    )
                except Exception as exc:
                    self._logger.warning("failed to mark session ended: %s", exc)
            self._logger.info(
                "session finished session_id=%s error_code=%s",
                assignment.session_id,
                result.error_code,
            )
            self._logger.info(
                "session_runner.run_session complete duration_ms=%.1f session_id=%s error_code=%s ended_by_control=%s",
                (perf_counter() - start) * 1000.0,
                assignment.session_id,
                result.error_code,
                result.ended_by_control,
            )

        return result

    def _build_livekit_adapter(self) -> LiveKitAdapter:
        if self._livekit_factory is not None:
            return self._livekit_factory()
        if self._host_config.livekit_mode == "real":
            return RealLiveKitAdapter(
                livekit_url=self._host_config.livekit_url,
                frame_width=self._host_config.frame_width,
                frame_height=self._host_config.frame_height,
                status_topic=self._host_config.status_topic,
                logger=self._logger,
            )
        return FakeLiveKitAdapter(self._logger)

    async def _watch_session_tasks(
        self,
        tasks: list[asyncio.Task[None]],
        result: SessionResult,
        session_ctx,
    ) -> None:
        stop_task = asyncio.create_task(self._session_stop_event.wait(), name="session_stop")
        try:
            done, _ = await asyncio.wait(
                [*tasks, stop_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                if task is stop_task:
                    continue
                self._consume_task_result(task, result)
        finally:
            session_ctx.running = False
            self._session_stop_event.set()
            stop_task.cancel()
            for task in tasks:
                task.cancel()
            await asyncio.gather(stop_task, *tasks, return_exceptions=True)

    @staticmethod
    def _is_modal_input_cancellation(exc: BaseException) -> bool:
        return (
            exc.__class__.__module__ == "modal.exception"
            and exc.__class__.__name__ == "InputCancellation"
        )

    async def _control_loop(
        self,
        *,
        control: SessionControlReducer,
        session_id: str,
        status: SessionStatusPublisher,
        livekit: LiveKitAdapter,
        result: SessionResult,
        session_ctx,
    ) -> None:
        while not self._session_stop_event.is_set():
            raw = await livekit.recv_control(timeout_s=0.5)
            if raw is None:
                continue
            outcome = await control.reduce(raw, session_id=session_id)
            if outcome.stop_requested:
                result.ended_by_control = True
                session_ctx.running = False
                self._session_stop_event.set()
                continue
            if outcome.pong_payload is not None:
                await status.pong(outcome.pong_payload)

    async def _model_loop(self, *, session_ctx) -> None:
        initial_inputs_received = await self._wait_for_initial_input_barrier(session_ctx)
        if not session_ctx.running or self._session_stop_event.is_set():
            return
        if self._requires_initial_input_barrier():
            self._logger.info(
                "session_runner._model_loop initial_input_barrier elapsed outcome=%s session_id=%s",
                "input_received" if initial_inputs_received else "timeout",
                session_ctx.session_id,
            )
        if self._coordinator is not None:
            await self._coordinator.mark_running(session_ctx.session_id)
        await self._emit_lifecycle_hook(
            self._lifecycle_hooks.on_running,
            session_ctx.session_id,
        )
        await self._runtime.run_session(session_ctx)
        self._session_stop_event.set()

    async def _wait_for_initial_input_barrier(self, session_ctx) -> bool:
        if not self._requires_initial_input_barrier():
            return False

        wait_task = asyncio.create_task(
            session_ctx.wait_for_initial_input(self.INITIAL_INPUT_GRACE_SECS),
            name="initial_input_barrier",
        )
        stop_task = asyncio.create_task(
            self._session_stop_event.wait(),
            name="initial_input_barrier_stop",
        )
        try:
            done, _ = await asyncio.wait(
                [wait_task, stop_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stop_task in done:
                return False
            return bool(wait_task.result())
        finally:
            wait_task.cancel()
            stop_task.cancel()
            await asyncio.gather(wait_task, stop_task, return_exceptions=True)

    def _requires_initial_input_barrier(self) -> bool:
        return any(definition.binding is None for definition in self._runtime.definition.inputs)

    @staticmethod
    async def _emit_lifecycle_hook(
        hook: Callable[[str], Any] | None,
        session_id: str,
    ) -> None:
        if hook is None:
            return
        result = hook(session_id)
        if inspect.isawaitable(result):
            await result

    async def _metrics_loop(
        self,
        *,
        output_router: OutputRouter,
        status: SessionStatusPublisher,
        session_ctx,
    ) -> None:
        while not self._session_stop_event.is_set():
            await status.frame_metrics(
                output_router.metrics(inference_ms_p50=session_ctx.inference_ms_p50())
            )
            await asyncio.sleep(1.0)

    async def _heartbeat_loop(self, *, session_id: str) -> None:
        while not self._session_stop_event.is_set():
            if self._coordinator is None:
                return
            try:
                await self._coordinator.mark_heartbeat(session_id)
            except Exception as exc:
                self._logger.warning("failed sending coordinator heartbeat: %s", exc)
            try:
                await asyncio.wait_for(
                    self._session_stop_event.wait(),
                    timeout=self.HEARTBEAT_INTERVAL_SECS,
                )
            except TimeoutError:
                continue

    async def _emit_terminal_status(
        self,
        status: SessionStatusPublisher,
        result: SessionResult,
        output_router: OutputRouter,
        session_ctx,
    ) -> None:
        metrics = output_router.metrics(inference_ms_p50=session_ctx.inference_ms_p50())
        if result.error_code:
            await status.error(
                result.error_code,
                publish_dropped_frames=metrics.publish_dropped_frames,
            )
        await status.ended(
            ended_by_control=result.ended_by_control,
            publish_dropped_frames=metrics.publish_dropped_frames,
        )

    def _consume_task_result(
        self,
        task: asyncio.Task[None],
        result: SessionResult,
    ) -> None:
        task_name = task.get_name()
        if task.cancelled():
            result.error_code = result.error_code or "WORKER_TASK_ERROR"
            self._logger.error("session task cancelled unexpectedly task=%s", task_name)
            return

        exc = task.exception()
        if exc is None:
            return
        result.error_code = result.error_code or "WORKER_TASK_ERROR"
        self._logger.exception("session task failed task=%s error=%s", task_name, exc)
