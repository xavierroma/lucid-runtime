from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lucid Modal helper CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    deploy = subparsers.add_parser("deploy")
    _add_modal_args(deploy)

    serve = subparsers.add_parser("serve")
    _add_modal_args(serve)

    logs = subparsers.add_parser("logs")
    _add_modal_args(logs)

    stop = subparsers.add_parser("stop")
    _add_modal_args(stop)

    volumes = subparsers.add_parser("create-volumes")
    _add_modal_args(volumes)

    download = subparsers.add_parser("download-model")
    _add_modal_args(download)

    return parser


def _add_modal_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env-file")
    parser.add_argument("--project", dest="project_path")
    parser.add_argument("--src", dest="project_src")
    parser.add_argument("--entrypoint", dest="app_entrypoint")
    parser.add_argument("--app-name", dest="app_name")
    parser.add_argument("modal_args", nargs=argparse.REMAINDER)


def _load_env_file(path: str | None) -> dict[str, str]:
    if path is None:
        return {}
    env_path = Path(path).expanduser().resolve()
    if not env_path.exists():
        raise RuntimeError(f"env file not found: {env_path}")
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def _resolved_value(args: argparse.Namespace, env: dict[str, str], flag: str, env_name: str) -> str:
    value = getattr(args, flag)
    if value:
        return value
    return env.get(env_name, os.getenv(env_name, "")).strip()


def _run_modal(args: argparse.Namespace) -> int:
    env_file_values = _load_env_file(args.env_file)
    full_env = os.environ.copy()
    full_env.update(env_file_values)

    project_path = _resolved_value(args, env_file_values, "project_path", "MODAL_PROJECT_PATH")
    project_src = _resolved_value(args, env_file_values, "project_src", "MODAL_PROJECT_SRC")
    app_entrypoint = _resolved_value(args, env_file_values, "app_entrypoint", "MODAL_APP_ENTRYPOINT")
    app_name = _resolved_value(args, env_file_values, "app_name", "MODAL_APP_NAME")

    if args.command in {"deploy", "serve", "download-model"} and not app_entrypoint:
        raise RuntimeError("MODAL_APP_ENTRYPOINT is required")
    if args.command in {"logs", "stop"} and not app_name:
        raise RuntimeError("MODAL_APP_NAME is required")
    if not project_path:
        raise RuntimeError("MODAL_PROJECT_PATH is required")

    repo_root = Path(__file__).resolve().parents[4]
    full_env["PYTHONPATH"] = ":".join(
        part for part in [str(repo_root / project_src) if project_src else "", full_env.get("PYTHONPATH", "")] if part
    )

    if args.command == "create-volumes":
        return _create_volumes(full_env, project_path)
    if args.command == "download-model":
        return _exec_modal(project_path, ["run", f"{app_entrypoint}::download_model", *args.modal_args], full_env)
    if args.command == "deploy":
        return _exec_modal(project_path, ["deploy", app_entrypoint, *args.modal_args], full_env)
    if args.command == "serve":
        return _exec_modal(project_path, ["serve", app_entrypoint, *args.modal_args], full_env)
    if args.command == "logs":
        return _exec_modal(project_path, ["app", "logs", *args.modal_args, app_name], full_env)
    if args.command == "stop":
        return _exec_modal(project_path, ["app", "stop", *args.modal_args, app_name], full_env)
    raise RuntimeError(f"unsupported command: {args.command}")


def _create_volumes(env: dict[str, str], project_path: str) -> int:
    model_volume = env.get("MODAL_MODEL_VOLUME", "lucid-models").strip()
    hf_volume = env.get("MODAL_HF_CACHE_VOLUME", "lucid-hf-cache").strip()
    for volume_name in (model_volume, hf_volume):
        result = _exec_modal(project_path, ["volume", "create", volume_name], env, check=False)
        if result not in {0, 1}:
            return result
    return 0


def _exec_modal(project_path: str, modal_args: list[str], env: dict[str, str], *, check: bool = True) -> int:
    command = ["uv", "run", "--project", project_path, "modal", *modal_args]
    completed = subprocess.run(command, env=env, check=False)
    if check and completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed.returncode


def main() -> int:
    args = build_parser().parse_args()
    try:
        return _run_modal(args)
    except KeyboardInterrupt:
        return 130
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
