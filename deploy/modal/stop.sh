#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

parse_modal_script_args "$@"
load_modal_env

if [[ ${#MODAL_SCRIPT_ARGS[@]} -gt 0 ]]; then
  run_modal app stop "${MODAL_SCRIPT_ARGS[@]}" "${MODAL_APP_NAME:-${DEFAULT_MODAL_APP_NAME}}"
else
  run_modal app stop "${MODAL_APP_NAME:-${DEFAULT_MODAL_APP_NAME}}"
fi
