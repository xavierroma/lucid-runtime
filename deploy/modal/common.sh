#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_ENV_FILE="${SCRIPT_DIR}/.env"
DEFAULT_MODAL_MODEL="yume"
MODAL_SCRIPT_ARGS=()
MODAL_MODEL_CLI=""
MODAL_PRESET_FORCE=0

print_env_help() {
  cat <<USAGE
Usage: $0 [--env-file PATH] [--model NAME] [-- additional modal args]

Environment file resolution order:
1. --env-file PATH
2. MODAL_ENV_FILE
3. ${DEFAULT_ENV_FILE}

Model preset resolution order:
1. --model NAME
2. MODAL_MODEL
3. ${DEFAULT_MODAL_MODEL}

Supported model presets:
- yume
- waypoint
USAGE
}

parse_modal_script_args() {
  MODAL_ENV_FILE="${MODAL_ENV_FILE:-${DEFAULT_ENV_FILE}}"
  MODAL_MODEL_CLI=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --env-file)
        if [[ $# -lt 2 ]]; then
          echo "missing value for --env-file" >&2
          exit 1
        fi
        MODAL_ENV_FILE="$2"
        shift 2
        ;;
      --model)
        if [[ $# -lt 2 ]]; then
          echo "missing value for --model" >&2
          exit 1
        fi
        MODAL_MODEL_CLI="$2"
        shift 2
        ;;
      --help|-h)
        print_env_help
        exit 0
        ;;
      --)
        shift
        MODAL_SCRIPT_ARGS=("$@")
        return
        ;;
      *)
        MODAL_SCRIPT_ARGS=("$@")
        return
        ;;
    esac
  done

  MODAL_SCRIPT_ARGS=()
}

set_default_env() {
  local var_name="$1"
  local value="$2"

  if [[ ${MODAL_PRESET_FORCE} -eq 0 && -n "${!var_name:-}" ]]; then
    return
  fi

  printf -v "$var_name" "%s" "$value"
  export "$var_name"
}

apply_model_preset() {
  case "${MODAL_MODEL}" in
    yume)
      set_default_env MODAL_PROJECT_PATH "examples/yume_modal"
      set_default_env MODAL_PROJECT_SRC "examples/yume_modal/src"
      set_default_env MODAL_APP_ENTRYPOINT "examples/yume_modal/src/yume_modal_example/modal_app.py"
      set_default_env MODAL_APP_NAME "lucid-runtime-worker"
      set_default_env MODAL_MODEL_VOLUME "lucid-yume-models"
      set_default_env MODAL_HF_CACHE_VOLUME "lucid-hf-cache"
      set_default_env MODAL_GPU_SMOKE_FUNCTION "flash_attn_smoke"
      set_default_env WM_MODEL_NAME "yume"
      set_default_env WM_MODEL_MODULE "yume_modal_example.model"
      set_default_env WM_ENGINE "yume"
      ;;
    waypoint)
      set_default_env MODAL_PROJECT_PATH "examples/waypoint_modal"
      set_default_env MODAL_PROJECT_SRC "examples/waypoint_modal/src"
      set_default_env MODAL_APP_ENTRYPOINT "examples/waypoint_modal/src/waypoint_modal_example/modal_app.py"
      set_default_env MODAL_APP_NAME "lucid-waypoint-worker"
      set_default_env MODAL_MODEL_VOLUME "lucid-waypoint-models"
      set_default_env MODAL_HF_CACHE_VOLUME "lucid-hf-cache"
      set_default_env MODAL_GPU_SMOKE_FUNCTION ""
      set_default_env WM_MODEL_NAME "waypoint"
      set_default_env WM_MODEL_MODULE "waypoint_modal_example.model"
      set_default_env WM_ENGINE "waypoint"
      ;;
    *)
      echo "unsupported modal model preset: ${MODAL_MODEL}" >&2
      echo "supported presets: yume, waypoint" >&2
      exit 1
      ;;
  esac
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

  local raw_modal_model="${MODAL_MODEL:-}"
  MODAL_MODEL="${MODAL_MODEL_CLI:-${raw_modal_model:-${DEFAULT_MODAL_MODEL}}}"
  if [[ -n "${MODAL_MODEL_CLI}" || -n "${raw_modal_model}" ]]; then
    MODAL_PRESET_FORCE=1
  else
    MODAL_PRESET_FORCE=0
  fi
  export MODAL_MODEL
  apply_model_preset
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
}

run_modal() {
  require_command uv
  (
    cd "${REPO_ROOT}"
    export PYTHONPATH="${REPO_ROOT}/${MODAL_PROJECT_SRC}:${PYTHONPATH:-}"
    uv run --project "${MODAL_PROJECT_PATH}" modal "$@"
  )
}
