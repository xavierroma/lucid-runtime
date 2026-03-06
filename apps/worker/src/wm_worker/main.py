"""Entrypoint for request-based session execution."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from wm_worker.config import ConfigError, RuntimeConfig, SessionConfig
from wm_worker.models import Assignment
from wm_worker.session_runner import SessionRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lucid worker single-session runtime")
    parser.add_argument("--worker-id", default="wm-worker-1")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--room-name", required=True)
    parser.add_argument("--worker-access-token", required=True)
    parser.add_argument("--video-track-name", default="main_video")
    parser.add_argument("--control-topic", default="wm.control.v1")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


async def _async_main(args: argparse.Namespace) -> int:
    logger = logging.getLogger("wm_worker")
    try:
        runtime_config = RuntimeConfig.from_env()
        session_config = SessionConfig.from_env(worker_id_override=args.worker_id)
    except ConfigError as exc:
        logger.error("invalid configuration: %s", exc)
        return 2

    assignment = Assignment(
        session_id=args.session_id,
        room_name=args.room_name,
        worker_access_token=args.worker_access_token,
        video_track_name=args.video_track_name,
        control_topic=args.control_topic,
    )

    runner = SessionRunner(runtime_config, session_config, logger)
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
        logging.getLogger("wm_worker").exception("worker crashed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
