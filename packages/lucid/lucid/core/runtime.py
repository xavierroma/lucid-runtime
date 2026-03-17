from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from time import perf_counter
from typing import Any, TypeAlias, TypeVar

from pydantic import BaseModel, ValidationError

from .model import LoadContext, LucidError, LucidModel, LucidSession, ManifestDict, MetricsFn, PublishFn, SessionContext
from .input_file import InputFile
from .spec import InputDefinition, ModelTarget, OutputBinding, OutputSpec, build_model_definition, resolve_model_class


class ActionDispatchError(LucidError):
    pass


ModelConfigInput: TypeAlias = BaseModel | Mapping[str, object] | None
InputFileResolver: TypeAlias = Callable[[str], InputFile | None]
T = TypeVar("T")


class RuntimeSession:
    def __init__(
        self,
        *,
        runtime: "LucidRuntime",
        session: LucidSession[Any],
        ctx: SessionContext,
        input_file_resolver: InputFileResolver | None = None,
    ) -> None:
        self._runtime = runtime
        self.session = session
        self.ctx = ctx
        self._input_file_resolver = input_file_resolver

    async def run(self) -> None:
        await _maybe_await(self.session.run())

    async def close(self) -> None:
        await _maybe_await(self.session.close())

    async def dispatch_input(self, name: str, args: dict[str, object]) -> None:
        definition = self._require_input(name)
        try:
            validated = definition.arg_model.model_validate(args)
        except ValidationError as exc:
            raise ActionDispatchError(f"invalid input args for {name}: {exc}") from exc
        resolved_args = validated.model_dump()
        for field_name, upload in definition.upload_fields.items():
            upload_id = resolved_args.get(field_name)
            if upload_id is None and upload.optional:
                continue
            if not isinstance(upload_id, str) or not upload_id.strip():
                raise ActionDispatchError(f"invalid input file id for {name}.{field_name}")
            if self._input_file_resolver is None:
                raise ActionDispatchError(f"input file uploads are not available for {name}.{field_name}")
            input_file = self._input_file_resolver(upload_id)
            if input_file is None:
                raise ActionDispatchError(f"unknown input file for {name}.{field_name}: {upload_id}")
            if input_file.mime_type.lower() not in upload.mime_types:
                raise ActionDispatchError(
                    f"invalid MIME type for {name}.{field_name}: {input_file.mime_type}"
                )
            if input_file.size_bytes > upload.max_bytes:
                raise ActionDispatchError(
                    f"input file too large for {name}.{field_name}: {input_file.size_bytes} > {upload.max_bytes}"
                )
            resolved_args[field_name] = input_file
        await _maybe_await(
            getattr(self.session, definition.handler_name)(**resolved_args)
        )

    def allows_input_while_paused(self, name: str) -> bool:
        definition = self._runtime._inputs.get(name)
        return bool(definition and definition.paused)

    def _require_input(self, name: str) -> InputDefinition:
        definition = self._runtime._inputs.get(name)
        if definition is None:
            raise ActionDispatchError(f"unknown input: {name}")
        return definition


class LucidRuntime:
    def __init__(
        self,
        *,
        model_target: ModelTarget,
        model: LucidModel[Any],
        logger: logging.Logger,
    ) -> None:
        self.definition = build_model_definition(type(model))
        self.model_target = model_target
        self.model = model
        self.logger = logger
        self._inputs = {item.name: item for item in self.definition.inputs}
        self._loaded = False
        self._unloaded = False

    @classmethod
    def load_model(
        cls,
        *,
        runtime_config: object,
        logger: logging.Logger,
        model: ModelTarget,
        config: ModelConfigInput = None,
    ) -> "LucidRuntime":
        model_cls = resolve_model_class(model)
        definition = build_model_definition(model_cls)
        instance = definition.cls(_coerce_model_config(definition.config_cls, config))
        instance.bind_runtime(runtime_config, logger)
        return cls(model_target=model, model=instance, logger=logger)

    async def load(self) -> None:
        if self._loaded:
            return
        start = perf_counter()
        try:
            await self.model.load(LoadContext(config=self.model.config, logger=self.logger))
        except Exception as exc:
            self.logger.error(
                "lucid.runtime.load failed duration_ms=%.1f model=%s error_type=%s",
                (perf_counter() - start) * 1000.0,
                self.definition.name,
                exc.__class__.__name__,
            )
            raise
        self._loaded = True
        self.logger.info(
            "lucid.runtime.load complete duration_ms=%.1f model=%s",
            (perf_counter() - start) * 1000.0,
            self.definition.name,
        )

    async def unload(self) -> None:
        if not self._loaded or self._unloaded:
            return
        await _maybe_await(self.model.unload())
        self._unloaded = True

    def manifest(self) -> ManifestDict:
        return self.definition.to_manifest()

    @property
    def outputs(self) -> tuple[OutputSpec, ...]:
        return self.definition.outputs

    def output_bindings(self) -> list[OutputBinding]:
        return self.definition.output_bindings()

    def open_session(
        self,
        *,
        session_id: str,
        room_name: str,
        publish_fn: PublishFn,
        metrics_fn: MetricsFn | None = None,
        input_file_resolver: InputFileResolver | None = None,
    ) -> RuntimeSession:
        ctx = SessionContext(
            session_id=session_id,
            room_name=room_name,
            outputs=self.outputs,
            publish_fn=publish_fn,
            metrics_fn=metrics_fn,
            logger=self.logger,
        )
        session = self.model.create_session(ctx)
        if inspect.isawaitable(session):
            raise LucidError("create_session() must be synchronous")
        if not isinstance(session, self.definition.session_cls):
            raise LucidError(
                f"create_session() returned {session.__class__.__name__}, expected {self.definition.session_cls.__name__}"
            )
        return RuntimeSession(
            runtime=self,
            session=session,
            ctx=ctx,
            input_file_resolver=input_file_resolver,
        )


def _coerce_model_config(
    config_cls: type[BaseModel],
    config: ModelConfigInput,
) -> BaseModel:
    if config is None:
        return config_cls()
    if isinstance(config, config_cls):
        return config
    if isinstance(config, BaseModel):
        raw = config.model_dump(mode="python")
    elif isinstance(config, Mapping):
        raw = config
    else:
        raise LucidError(
            f"lucid model config must be a BaseModel or dict, got {type(config).__name__}"
        )
    try:
        return config_cls.model_validate(raw)
    except ValidationError as exc:
        raise LucidError(f"invalid lucid model config: {exc}") from exc


async def _maybe_await(value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value
