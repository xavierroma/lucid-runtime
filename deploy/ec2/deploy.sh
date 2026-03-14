#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

parse_ec2_script_args "$@"
load_ec2_env
validate_coordinator_runtime_env

IMAGE="$(resolve_coordinator_image)"
REGISTRY_HOST="$(registry_host_from_image "${IMAGE}")"
IS_ECR=0
if is_ecr_registry_host "${REGISTRY_HOST}"; then
  IS_ECR=1
fi

EC2_DEPLOY_TRANSPORT="${EC2_DEPLOY_TRANSPORT:-auto}"
EC2_INSTANCE_ID="${EC2_INSTANCE_ID:-}"
EC2_USER="${EC2_USER:-ec2-user}"
EC2_SSH_PORT="${EC2_SSH_PORT:-22}"
CONTAINER_NAME="${CONTAINER_NAME:-${DEFAULT_CONTAINER_NAME}}"
HOST_PORT="${HOST_PORT:-${DEFAULT_HOST_PORT}}"
DOCKER_BUILD_PLATFORM="${DOCKER_BUILD_PLATFORM:-${DEFAULT_BUILD_PLATFORM}}"
REMOTE_LOGIN_COMMAND="${EC2_DOCKER_LOGIN_COMMAND:-}"
REMOTE_NEEDS_AWS_CLI=0
TRANSPORT=""
RUN_SCRIPT_B64="$(base64 < "${SCRIPT_DIR}/run-coordinator.sh" | tr -d '\n')"
ENV_FILE_B64="$(base64 < "${EC2_ENV_FILE}" | tr -d '\n')"
LOCAL_MODELS_FILE="${EC2_MODELS_FILE:-${DEFAULT_MODELS_FILE}}"

if [[ ! -f "${LOCAL_MODELS_FILE}" ]]; then
  echo "models file not found: ${LOCAL_MODELS_FILE}" >&2
  echo "copy deploy/ec2/coordinator.models.example.json to deploy/ec2/coordinator.models.json or set EC2_MODELS_FILE" >&2
  exit 1
fi

MODELS_FILE_B64="$(base64 < "${LOCAL_MODELS_FILE}" | tr -d '\n')"

if [[ ${IS_ECR} -eq 1 && -z "${REMOTE_LOGIN_COMMAND}" ]]; then
  REMOTE_NEEDS_AWS_CLI=1
fi

if [[ ${EC2_SKIP_BUILD} -eq 0 ]]; then
  require_command docker
  if [[ ${IS_ECR} -eq 1 ]]; then
    require_command aws
    ecr_login "${REGISTRY_HOST}" "${AWS_PROFILE:-}"
  fi
  if ! docker buildx version >/dev/null 2>&1; then
    echo "docker buildx is required for coordinator image builds" >&2
    exit 1
  fi

  (
    cd "${REPO_ROOT}"
    docker buildx build \
      --platform "${DOCKER_BUILD_PLATFORM}" \
      -f apps/coordinator/Dockerfile \
      -t "${IMAGE}" \
      --push \
      .
  )
fi

if [[ ${EC2_SKIP_REMOTE} -eq 1 ]]; then
  echo "Coordinator image published: ${IMAGE}"
  exit 0
fi

require_env_vars EC2_HOST

case "${EC2_DEPLOY_TRANSPORT}" in
  auto)
    if [[ -z "${EC2_INSTANCE_ID}" ]]; then
      EC2_INSTANCE_ID="$(resolve_ec2_instance_id "${EC2_HOST}" || true)"
    fi
    if [[ -n "${EC2_INSTANCE_ID}" ]] && ssm_instance_is_online "${EC2_INSTANCE_ID}"; then
      TRANSPORT="ssm"
    else
      TRANSPORT="ssh"
    fi
    ;;
  ssm|ssh)
    TRANSPORT="${EC2_DEPLOY_TRANSPORT}"
    ;;
  *)
    echo "unsupported EC2_DEPLOY_TRANSPORT: ${EC2_DEPLOY_TRANSPORT}" >&2
    exit 1
    ;;
esac

if [[ -n "${EC2_REMOTE_DIR:-}" ]]; then
  REMOTE_DIR="${EC2_REMOTE_DIR}"
elif [[ "${TRANSPORT}" == "ssh" ]]; then
  require_command ssh
  SSH_ARGS=(-p "${EC2_SSH_PORT}")
  if [[ -n "${EC2_SSH_KEY_PATH:-}" ]]; then
    SSH_ARGS+=(-i "${EC2_SSH_KEY_PATH}")
  fi
  REMOTE_TARGET="${EC2_USER}@${EC2_HOST}"
  REMOTE_HOME="$(
    ssh "${SSH_ARGS[@]}" "${REMOTE_TARGET}" 'printf %s "$HOME"'
  )"
  REMOTE_DIR="${REMOTE_HOME}/lucid-runtime/deploy/ec2"
else
  REMOTE_DIR="/home/${EC2_USER}/lucid-runtime/deploy/ec2"
fi

if [[ "${REMOTE_DIR}" == "~/"* ]]; then
  REMOTE_DIR="/home/${EC2_USER}/${REMOTE_DIR#~/}"
fi

REMOTE_ENV_FILE="${EC2_REMOTE_ENV_FILE:-${REMOTE_DIR}/coordinator.env}"
if [[ "${REMOTE_ENV_FILE}" == "~/"* ]]; then
  REMOTE_ENV_FILE="/home/${EC2_USER}/${REMOTE_ENV_FILE#~/}"
fi

REMOTE_MODELS_FILE="${EC2_REMOTE_MODELS_FILE:-${REMOTE_DIR}/coordinator.models.json}"
if [[ "${REMOTE_MODELS_FILE}" == "~/"* ]]; then
  REMOTE_MODELS_FILE="/home/${EC2_USER}/${REMOTE_MODELS_FILE#~/}"
fi

REMOTE_RUN_SCRIPT="${REMOTE_DIR}/run-coordinator.sh"

REMOTE_LOGIN_SNIPPET=""
if [[ ${IS_ECR} -eq 1 && -z "${REMOTE_LOGIN_COMMAND}" ]]; then
  REMOTE_REGION="$(ecr_region_from_registry_host "${REGISTRY_HOST}")"
  if [[ -n "${EC2_AWS_PROFILE:-}" ]]; then
    REMOTE_LOGIN_SNIPPET="$(
      cat <<EOF
if ! command -v aws >/dev/null 2>&1; then
  echo aws CLI is required on the EC2 host for automatic ECR login >&2
  exit 1
fi
remote_password="\$(AWS_PROFILE=$(printf '%q' "${EC2_AWS_PROFILE}") aws ecr get-login-password --region $(printf '%q' "${REMOTE_REGION}"))" || {
  echo "failed to get ECR login password on the EC2 host using AWS profile $(printf '%q' "${EC2_AWS_PROFILE}")" >&2
  exit 1
}
printf '%s' "\${remote_password}" | docker login --username AWS --password-stdin $(printf '%q' "${REGISTRY_HOST}")
EOF
    )"
  else
    REMOTE_LOGIN_SNIPPET="$(
      cat <<EOF
if ! command -v aws >/dev/null 2>&1; then
  echo aws CLI is required on the EC2 host for automatic ECR login >&2
  exit 1
fi
remote_password="\$(aws ecr get-login-password --region $(printf '%q' "${REMOTE_REGION}"))" || {
  echo "failed to get ECR login password on the EC2 host" >&2
  exit 1
}
printf '%s' "\${remote_password}" | docker login --username AWS --password-stdin $(printf '%q' "${REGISTRY_HOST}")
EOF
    )"
  fi
elif [[ -n "${REMOTE_LOGIN_COMMAND}" ]]; then
  REMOTE_LOGIN_SNIPPET="bash -o pipefail -lc $(printf '%q' "${REMOTE_LOGIN_COMMAND}")"
fi

REMOTE_COMMAND="$(
  cat <<EOF
set -euo pipefail
mkdir -p $(printf '%q' "${REMOTE_DIR}")
printf '%s' $(printf '%q' "${RUN_SCRIPT_B64}") | base64 -d > $(printf '%q' "${REMOTE_RUN_SCRIPT}")
chmod +x $(printf '%q' "${REMOTE_RUN_SCRIPT}")
printf '%s' $(printf '%q' "${ENV_FILE_B64}") | base64 -d > $(printf '%q' "${REMOTE_ENV_FILE}")
chmod 600 $(printf '%q' "${REMOTE_ENV_FILE}")
printf '%s' $(printf '%q' "${MODELS_FILE_B64}") | base64 -d > $(printf '%q' "${REMOTE_MODELS_FILE}")
chmod 600 $(printf '%q' "${REMOTE_MODELS_FILE}")
${REMOTE_LOGIN_SNIPPET}
COORDINATOR_IMAGE=$(printf '%q' "${IMAGE}") HOST_PORT=$(printf '%q' "${HOST_PORT}") CONTAINER_NAME=$(printf '%q' "${CONTAINER_NAME}") $(printf '%q' "${REMOTE_RUN_SCRIPT}") $(printf '%q' "${REMOTE_ENV_FILE}") $(printf '%q' "${REMOTE_MODELS_FILE}")
EOF
)"

if [[ "${TRANSPORT}" == "ssh" ]]; then
  require_command ssh
  SSH_ARGS=(-p "${EC2_SSH_PORT}")
  if [[ -n "${EC2_SSH_KEY_PATH:-}" ]]; then
    SSH_ARGS+=(-i "${EC2_SSH_KEY_PATH}")
  fi
  REMOTE_TARGET="${EC2_USER}@${EC2_HOST}"
  ssh "${SSH_ARGS[@]}" "${REMOTE_TARGET}" "bash -lc $(printf '%q' "${REMOTE_COMMAND}")"
  echo "Coordinator deployed to ${REMOTE_TARGET}"
else
  require_command aws
  require_command python3
  if [[ -z "${EC2_INSTANCE_ID}" ]]; then
    EC2_INSTANCE_ID="$(resolve_ec2_instance_id "${EC2_HOST}" || true)"
  fi
  if [[ -z "${EC2_INSTANCE_ID}" ]]; then
    echo "EC2 instance id is required for SSM transport; set EC2_INSTANCE_ID or make EC2_HOST resolvable via EC2 APIs" >&2
    exit 1
  fi
  if ! ssm_instance_is_online "${EC2_INSTANCE_ID}"; then
    echo "EC2 instance ${EC2_INSTANCE_ID} is not online in SSM" >&2
    exit 1
  fi

  SSM_COMMAND_ID="$(
    python3 -c 'import json, sys; print(json.dumps({"commands": [sys.argv[1]]}))' "bash -lc $(printf '%q' "${REMOTE_COMMAND}")" \
      | aws ssm send-command \
          --instance-ids "${EC2_INSTANCE_ID}" \
          --document-name AWS-RunShellScript \
          --parameters file:///dev/stdin \
          --query 'Command.CommandId' \
          --output text
  )"

  if ! aws ssm wait command-executed --command-id "${SSM_COMMAND_ID}" --instance-id "${EC2_INSTANCE_ID}"; then
    :
  fi

  SSM_STATUS="$(aws ssm get-command-invocation --command-id "${SSM_COMMAND_ID}" --instance-id "${EC2_INSTANCE_ID}" --query 'StatusDetails' --output text)"
  SSM_STDOUT="$(aws ssm get-command-invocation --command-id "${SSM_COMMAND_ID}" --instance-id "${EC2_INSTANCE_ID}" --query 'StandardOutputContent' --output text)"
  SSM_STDERR="$(aws ssm get-command-invocation --command-id "${SSM_COMMAND_ID}" --instance-id "${EC2_INSTANCE_ID}" --query 'StandardErrorContent' --output text)"

  if [[ -n "${SSM_STDOUT}" && "${SSM_STDOUT}" != "None" ]]; then
    printf '%s\n' "${SSM_STDOUT}"
  fi

  if [[ -n "${SSM_STDERR}" && "${SSM_STDERR}" != "None" ]]; then
    printf '%s\n' "${SSM_STDERR}" >&2
  fi

  if [[ "${SSM_STATUS}" != "Success" ]]; then
    echo "SSM deployment command failed with status: ${SSM_STATUS}" >&2
    exit 1
  fi

  echo "Coordinator deployed to ${EC2_INSTANCE_ID} via SSM"
fi

echo "Image: ${IMAGE}"
echo "Health check: curl http://${EC2_HOST}:${HOST_PORT}/healthz"
