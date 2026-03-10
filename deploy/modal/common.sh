#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_ENV_FILE="${SCRIPT_DIR}/.env"
MODAL_APP_ENTRYPOINT="examples/yume_modal/src/yume_modal_example/modal_app.py"
DEFAULT_MODAL_APP_NAME="lucid-runtime-worker"
MODAL_SCRIPT_ARGS=()

print_env_help() {
  cat <<USAGE
Usage: $0 [--env-file PATH] [-- additional modal args]

Environment file resolution order:
1. --env-file PATH
2. MODAL_ENV_FILE
3. ${DEFAULT_ENV_FILE}
USAGE
}

parse_modal_script_args() {
  MODAL_ENV_FILE="${MODAL_ENV_FILE:-${DEFAULT_ENV_FILE}}"

  if [[ ${1:-} == "--env-file" ]]; then
    if [[ $# -lt 2 ]]; then
      echo "missing value for --env-file" >&2
      exit 1
    fi
    MODAL_ENV_FILE="$2"
    shift 2
  fi

  if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
    print_env_help
    exit 0
  fi

  if [[ ${1:-} == "--" ]]; then
    shift
  fi

  MODAL_SCRIPT_ARGS=("$@")
}

load_modal_env() {
  if [[ ! -f "${MODAL_ENV_FILE}" ]]; then
    echo "modal env file not found: ${MODAL_ENV_FILE}" >&2
    echo "copy deploy/modal/.env.example to deploy/modal/.env or pass --env-file" >&2
    exit 1
  fi

  set -a
  # shellcheck disable=SC1090
  source "${MODAL_ENV_FILE}"
  set +a
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "required command not found: $1" >&2
    exit 1
  fi
}

require_env_vars() {
  local missing=()
  local var_name

  for var_name in "$@"; do
    if [[ -z "${!var_name:-}" ]]; then
      missing+=("${var_name}")
    fi
  done

  if (( ${#missing[@]} > 0 )); then
    echo "missing required env vars: ${missing[*]}" >&2
    exit 1
  fi
}

validate_runtime_env() {
  require_env_vars MODAL_DISPATCH_TOKEN LIVEKIT_URL WM_ENGINE WM_LIVEKIT_MODE WM_MODEL_MODULE

  if [[ "${WM_ENGINE}" == "yume" ]]; then
    require_env_vars YUME_MODEL_DIR
  fi
}

run_modal() {
  require_command uv
  (
    cd "${REPO_ROOT}"
    export PYTHONPATH="${REPO_ROOT}/examples/yume_modal/src:${PYTHONPATH:-}"
    uv run --project examples/yume_modal modal "$@"
  )
}
