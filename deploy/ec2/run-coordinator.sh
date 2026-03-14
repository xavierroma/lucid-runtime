#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-deploy/ec2/coordinator.env}"
MODELS_FILE="${2:-deploy/ec2/coordinator.models.json}"
CONTAINER_NAME="${CONTAINER_NAME:-lucid-coordinator}"
IMAGE="${COORDINATOR_IMAGE:?Set COORDINATOR_IMAGE to a registry image, e.g. ghcr.io/acme/lucid-coordinator:sha}"
HOST_PORT="${HOST_PORT:-8080}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing env file: ${ENV_FILE}" >&2
  exit 1
fi

if [[ ! -f "${MODELS_FILE}" ]]; then
  echo "Missing models file: ${MODELS_FILE}" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required on the target host" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

if [[ -z "${COORDINATOR_MODELS_FILE:-}" ]]; then
  echo "COORDINATOR_MODELS_FILE must be set in ${ENV_FILE}" >&2
  exit 1
fi

docker pull "${IMAGE}"

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  docker rm -f "${CONTAINER_NAME}"
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  --env-file "${ENV_FILE}" \
  -v "${MODELS_FILE}:${COORDINATOR_MODELS_FILE}:ro" \
  -p "${HOST_PORT}:8080" \
  "${IMAGE}"

echo "Coordinator container started: ${CONTAINER_NAME}"
echo "Health check: curl http://127.0.0.1:${HOST_PORT}/healthz"
