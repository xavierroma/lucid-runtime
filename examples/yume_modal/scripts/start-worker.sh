#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/examples/yume_modal/src:${PYTHONPATH:-}"

if [[ -z "${WM_MODEL_MODULE:-}" ]]; then
  echo "WM_MODEL_MODULE must be set" >&2
  exit 1
fi

if [[ -z "${SESSION_ID:-}" || -z "${ROOM_NAME:-}" || -z "${WORKER_ACCESS_TOKEN:-}" ]]; then
  echo "SESSION_ID, ROOM_NAME, and WORKER_ACCESS_TOKEN must be set for request-based worker execution" >&2
  exit 1
fi

exec lucid-worker \
  --worker-id "${WORKER_ID:-wm-worker-1}" \
  --session-id "${SESSION_ID}" \
  --room-name "${ROOM_NAME}" \
  --worker-access-token "${WORKER_ACCESS_TOKEN}" \
  --control-topic "${CONTROL_TOPIC:-wm.control}"
