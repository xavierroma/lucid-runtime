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

exec wm-worker --worker-id "${WORKER_ID:-wm-worker-1}"
