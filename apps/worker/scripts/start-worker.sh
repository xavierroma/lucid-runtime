#!/usr/bin/env bash
set -euo pipefail

if [[ "${WM_ENGINE:-fake}" == "yume" ]]; then
  if [[ -z "${YUME_MODEL_DIR:-}" ]]; then
    echo "YUME_MODEL_DIR is required when WM_ENGINE=yume" >&2
    exit 1
  fi

  required_files=(
    "diffusion_pytorch_model.safetensors"
    "config.yaml"
    "Wan2.2_VAE.pth"
  )
  for file_name in "${required_files[@]}"; do
    if [[ ! -f "${YUME_MODEL_DIR}/${file_name}" ]]; then
      echo "Missing ${YUME_MODEL_DIR}/${file_name}" >&2
      exit 1
    fi
  done
fi

if [[ -z "${SESSION_ID:-}" || -z "${ROOM_NAME:-}" || -z "${WORKER_ACCESS_TOKEN:-}" ]]; then
  echo "SESSION_ID, ROOM_NAME, and WORKER_ACCESS_TOKEN must be set for request-based worker execution" >&2
  exit 1
fi

exec wm-worker \
  --worker-id "${WORKER_ID:-wm-worker-1}" \
  --session-id "${SESSION_ID}" \
  --room-name "${ROOM_NAME}" \
  --worker-access-token "${WORKER_ACCESS_TOKEN}" \
  --video-track-name "${VIDEO_TRACK_NAME:-main_video}" \
  --control-topic "${CONTROL_TOPIC:-wm.control.v1}"
