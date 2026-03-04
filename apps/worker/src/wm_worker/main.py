"""Entrypoint for the production worker runtime."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from wm_worker.config import ConfigError, WorkerConfig
from wm_worker.runner import WorkerRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lucid worker runtime")
    parser.add_argument("--worker-id", default="wm-worker-1")
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
        config = WorkerConfig.from_env(worker_id_override=args.worker_id)
    except ConfigError as exc:
        logger.error("invalid configuration: %s", exc)
        return 2

    runner = WorkerRunner(config, logger)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, runner.request_shutdown)
    await runner.run()
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
