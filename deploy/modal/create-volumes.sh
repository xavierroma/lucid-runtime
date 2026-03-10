#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

parse_modal_script_args "$@"
load_modal_env

MODEL_VOLUME="${MODAL_MODEL_VOLUME:-lucid-yume-models}"
HF_VOLUME="${MODAL_HF_CACHE_VOLUME:-lucid-hf-cache}"

ensure_volume() {
  local volume_name="$1"
  local output=""

  if output="$(run_modal volume create "${volume_name}" 2>&1)"; then
    printf '%s\n' "${output}"
    return 0
  fi

  if [[ "${output}" == *"already exists"* ]]; then
    printf '%s\n' "${output}"
    return 0
  fi

  printf '%s\n' "${output}" >&2
  return 1
}

ensure_volume "${MODEL_VOLUME}"
ensure_volume "${HF_VOLUME}"
