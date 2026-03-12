from __future__ import annotations

import argparse
import logging
import sys

from lucid import resolve_model_class

from .config import (
    ConfigError,
    load_livekit_api_credentials_from_env,
    load_model_config_from_path,
    load_runtime_config_from_env,
)
from .research_server import ResearchSessionService, create_app as create_research_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lucid dev server")
    parser.add_argument("model", help="Lucid model spec, for example pkg.module:ClassName")
    parser.add_argument("--config", help="Model config file (YAML or JSON)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--log-level", default="info")
    return parser


def _configure_logging(level_name: str) -> None:
    numeric_level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> int:
    args = build_parser().parse_args()
    _configure_logging(args.log_level)
    try:
        model_cls = resolve_model_class(args.model)
        host_config = load_runtime_config_from_env()
        api_key, api_secret = load_livekit_api_credentials_from_env()
        model_config = load_model_config_from_path(model_cls.config_cls, args.config)
    except (ConfigError, RuntimeError) as exc:
        logging.getLogger("lucid_dev").error("invalid configuration: %s", exc)
        return 2

    service = ResearchSessionService(
        host_config,
        logging.getLogger("lucid_dev"),
        model=args.model,
        model_config=model_config,
    )
    app = create_research_app(service, api_key=api_key, api_secret=api_secret)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
    return 0


if __name__ == "__main__":
    sys.exit(main())
