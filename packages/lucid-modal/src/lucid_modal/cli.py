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


def _resolve_project_root(args: argparse.Namespace, env: dict[str, str]) -> Path:
    project_path = _resolved_value(args, env, "project_path", "MODAL_PROJECT_PATH") or "."
    return Path(project_path).expanduser().resolve()


def _resolve_project_src(args: argparse.Namespace, env: dict[str, str], project_root: Path) -> str:
    configured = _resolved_value(args, env, "project_src", "MODAL_PROJECT_SRC")
    if configured:
        return configured
    if (project_root / "src").exists():
        return "src"
    return ""


def _build_modal_target(
    app_entrypoint: str,
    *,
    project_root: Path,
    project_src: str,
    function_name: str | None = None,
) -> list[str]:
    entrypoint = app_entrypoint.strip()
    if not entrypoint:
        raise RuntimeError("MODAL_APP_ENTRYPOINT is required")

    if entrypoint.endswith(".py") or "/" in entrypoint or os.sep in entrypoint:
        entry_path = Path(entrypoint)
        if not entry_path.is_absolute():
            entry_path = (project_root / entry_path).resolve()

        if project_src:
            src_root = (project_root / project_src).resolve()
            if entry_path.is_relative_to(src_root):
                module_ref = ".".join(entry_path.relative_to(src_root).with_suffix("").parts)
                if function_name is not None:
                    module_ref = f"{module_ref}::{function_name}"
                return ["-m", module_ref]

        path_ref = str(entry_path)
        if function_name is not None:
            path_ref = f"{path_ref}::{function_name}"
        return [path_ref]

    module_ref = entrypoint
    if function_name is not None:
        module_ref = f"{module_ref}::{function_name}"
    return ["-m", module_ref]


def _run_modal(args: argparse.Namespace) -> int:
    env_file_values = _load_env_file(args.env_file)
    full_env = os.environ.copy()
    full_env.update(env_file_values)

    project_root = _resolve_project_root(args, env_file_values)
    project_src = _resolve_project_src(args, env_file_values, project_root)
    app_entrypoint = _resolved_value(args, env_file_values, "app_entrypoint", "MODAL_APP_ENTRYPOINT")
    app_name = _resolved_value(args, env_file_values, "app_name", "MODAL_APP_NAME")

    if args.command in {"deploy", "serve", "download-model"} and not app_entrypoint:
        raise RuntimeError("MODAL_APP_ENTRYPOINT is required")
    if args.command in {"logs", "stop"} and not app_name:
        raise RuntimeError("MODAL_APP_NAME is required")

    project_src_root = (project_root / project_src).resolve() if project_src else None
    full_env["PYTHONPATH"] = ":".join(
        part
        for part in [
            str(project_src_root) if project_src_root is not None else "",
            full_env.get("PYTHONPATH", ""),
        ]
        if part
    )

    if args.command == "create-volumes":
        return _create_volumes(full_env, project_root)
    if args.command == "download-model":
        modal_target = _build_modal_target(
            app_entrypoint,
            project_root=project_root,
            project_src=project_src,
            function_name="download_model",
        )
        return _exec_modal(project_root, ["run", *modal_target, *args.modal_args], full_env)
    if args.command == "deploy":
        modal_target = _build_modal_target(
            app_entrypoint,
            project_root=project_root,
            project_src=project_src,
        )
        return _exec_modal(project_root, ["deploy", *modal_target, *args.modal_args], full_env)
    if args.command == "serve":
        modal_target = _build_modal_target(
            app_entrypoint,
            project_root=project_root,
            project_src=project_src,
        )
        return _exec_modal(project_root, ["serve", *modal_target, *args.modal_args], full_env)
    if args.command == "logs":
        return _exec_modal(project_root, ["app", "logs", *args.modal_args, app_name], full_env)
    if args.command == "stop":
        return _exec_modal(project_root, ["app", "stop", *args.modal_args, app_name], full_env)
    raise RuntimeError(f"unsupported command: {args.command}")


def _create_volumes(env: dict[str, str], project_root: Path) -> int:
    model_volume = env.get("MODAL_MODEL_VOLUME", "lucid-models").strip()
    hf_volume = env.get("MODAL_HF_CACHE_VOLUME", "lucid-hf-cache").strip()
    for volume_name in (model_volume, hf_volume):
        result = _exec_modal(project_root, ["volume", "create", volume_name], env, check=False)
        if result not in {0, 1}:
            return result
    return 0


def _exec_modal(project_root: Path, modal_args: list[str], env: dict[str, str], *, check: bool = True) -> int:
    command = [sys.executable, "-m", "modal", *modal_args]
    completed = subprocess.run(command, cwd=project_root, env=env, check=False)
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
