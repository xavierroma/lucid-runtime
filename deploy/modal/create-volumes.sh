#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

parse_modal_script_args "$@"
load_modal_env

MODEL_VOLUME="${MODAL_MODEL_VOLUME:-lucid-yume-models}"
HF_VOLUME="${MODAL_HF_CACHE_VOLUME:-lucid-hf-cache}"

run_modal volume create "${MODEL_VOLUME}"
run_modal volume create "${HF_VOLUME}"
