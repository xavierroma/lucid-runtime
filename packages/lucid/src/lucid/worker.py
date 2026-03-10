"""CLI entrypoint for request-based session execution."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .config import ConfigError, RuntimeConfig, SessionConfig
from .discovery import ensure_model_module_loaded
from .host import SessionRunner
from .types import Assignment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lucid worker single-session runtime")
    parser.add_argument("--worker-id", default="wm-worker-1")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--room-name", required=True)
    parser.add_argument("--worker-access-token", required=True)
    parser.add_argument("--control-topic", default="wm.control")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


async def _async_main(args: argparse.Namespace) -> int:
    logger = logging.getLogger("lucid.worker")
    try:
        ensure_model_module_loaded()
        host_config = RuntimeConfig.from_env()
        session_config = SessionConfig.from_env(worker_id_override=args.worker_id)
    except (ConfigError, RuntimeError) as exc:
        logger.error("invalid configuration: %s", exc)
        return 2

    assignment = Assignment(
        session_id=args.session_id,
        room_name=args.room_name,
        worker_access_token=args.worker_access_token,
        control_topic=args.control_topic,
    )
    runner = SessionRunner(host_config, session_config, logger)
    try:
        await runner.run_session(assignment)
    finally:
        await runner.close()
    return 0


def main() -> int:
    args = build_parser().parse_args()
    _configure_logging(args.log_level)
    try:
        return asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # pragma: no cover - defensive process-level catch
        logging.getLogger("lucid.worker").exception("worker crashed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
