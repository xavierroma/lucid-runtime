#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

parse_modal_script_args "$@"
load_modal_env
validate_runtime_env

if [[ -z "${MODAL_GPU_SMOKE_FUNCTION:-}" ]]; then
  echo "no flash-attn smoke function is configured for MODAL_MODEL=${MODAL_MODEL}" >&2
  echo "supported preset for this script: yume" >&2
  exit 1
fi

if [[ ${#MODAL_SCRIPT_ARGS[@]} -gt 0 ]]; then
  run_modal run "${MODAL_APP_ENTRYPOINT}::${MODAL_GPU_SMOKE_FUNCTION}" "${MODAL_SCRIPT_ARGS[@]}"
else
  run_modal run "${MODAL_APP_ENTRYPOINT}::${MODAL_GPU_SMOKE_FUNCTION}"
fi
