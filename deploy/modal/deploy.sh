#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

parse_modal_script_args "$@"
load_modal_env
validate_runtime_env

if [[ ${#MODAL_SCRIPT_ARGS[@]} -gt 0 ]]; then
  run_modal deploy "${MODAL_APP_ENTRYPOINT}" "${MODAL_SCRIPT_ARGS[@]}"
else
  run_modal deploy "${MODAL_APP_ENTRYPOINT}"
fi
