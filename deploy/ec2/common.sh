#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_ENV_FILE="${SCRIPT_DIR}/coordinator.env"
DEFAULT_MODELS_FILE="${SCRIPT_DIR}/coordinator.models.json"
DEFAULT_BUILD_PLATFORM="linux/amd64"
DEFAULT_CONTAINER_NAME="lucid-coordinator"
DEFAULT_HOST_PORT="8080"

EC2_ENV_FILE=""
EC2_IMAGE_TAG_CLI=""
EC2_SKIP_BUILD=0
EC2_SKIP_REMOTE=0

print_ec2_usage() {
  cat <<USAGE
Usage: $0 [--env-file PATH] [--image-tag TAG] [--skip-build] [--skip-remote]

Environment file resolution order:
1. --env-file PATH
2. EC2_ENV_FILE
3. ${DEFAULT_ENV_FILE}

Image resolution order:
1. COORDINATOR_IMAGE
2. COORDINATOR_IMAGE_REPOSITORY + (--image-tag | COORDINATOR_IMAGE_TAG | auto tag)

Remote rollout requires:
- EC2_HOST

Optional deployment settings:
- EC2_DEPLOY_TRANSPORT (default: auto; supported: auto, ssm, ssh)
- EC2_INSTANCE_ID (optional; required for SSM if EC2_HOST cannot be resolved via EC2 APIs)
- EC2_USER (default: ec2-user)
- EC2_SSH_PORT (default: 22)
- EC2_SSH_KEY_PATH
- EC2_REMOTE_DIR (default: \$HOME/lucid-runtime/deploy/ec2 over SSH, /home/\$EC2_USER/lucid-runtime/deploy/ec2 over SSM)
- EC2_MODELS_FILE (default: ${DEFAULT_MODELS_FILE})
- EC2_DOCKER_LOGIN_COMMAND
- AWS_PROFILE (optional local AWS profile for automatic ECR login)
- EC2_AWS_PROFILE (optional AWS profile on the EC2 host for automatic ECR login)
- DOCKER_BUILD_PLATFORM (default: ${DEFAULT_BUILD_PLATFORM})
- CONTAINER_NAME (default: ${DEFAULT_CONTAINER_NAME})
- HOST_PORT (default: ${DEFAULT_HOST_PORT})
USAGE
}

parse_ec2_script_args() {
  EC2_ENV_FILE="${EC2_ENV_FILE:-${DEFAULT_ENV_FILE}}"
  EC2_IMAGE_TAG_CLI=""
  EC2_SKIP_BUILD=0
  EC2_SKIP_REMOTE=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --env-file)
        if [[ $# -lt 2 ]]; then
          echo "missing value for --env-file" >&2
          exit 1
        fi
        EC2_ENV_FILE="$2"
        shift 2
        ;;
      --image-tag)
        if [[ $# -lt 2 ]]; then
          echo "missing value for --image-tag" >&2
          exit 1
        fi
        EC2_IMAGE_TAG_CLI="$2"
        shift 2
        ;;
      --skip-build)
        EC2_SKIP_BUILD=1
        shift
        ;;
      --skip-remote)
        EC2_SKIP_REMOTE=1
        shift
        ;;
      --help|-h)
        print_ec2_usage
        exit 0
        ;;
      *)
        echo "unsupported argument: $1" >&2
        print_ec2_usage >&2
        exit 1
        ;;
    esac
  done
}

load_ec2_env() {
  if [[ ! -f "${EC2_ENV_FILE}" ]]; then
    echo "ec2 env file not found: ${EC2_ENV_FILE}" >&2
    echo "copy deploy/ec2/coordinator.env.example to deploy/ec2/coordinator.env or pass --env-file" >&2
    exit 1
  fi

  set -a
  # shellcheck disable=SC1090
  source "${EC2_ENV_FILE}"
  set +a
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "required command not found: $1" >&2
    exit 1
  fi
}

require_env_vars() {
  local missing=()
  local var_name

  for var_name in "$@"; do
    if [[ -z "${!var_name:-}" ]]; then
      missing+=("${var_name}")
    fi
  done

  if (( ${#missing[@]} > 0 )); then
    echo "missing required env vars: ${missing[*]}" >&2
    exit 1
  fi
}

default_coordinator_image_tag() {
  local git_sha

  if command -v git >/dev/null 2>&1 && git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git_sha="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
    if [[ -n "$(git -C "${REPO_ROOT}" status --short --untracked-files=normal)" ]]; then
      echo "${git_sha}-dirty"
    else
      echo "${git_sha}"
    fi
    return
  fi

  date -u +%Y%m%d%H%M%S
}

resolve_coordinator_image() {
  local image_tag=""

  if [[ -n "${EC2_IMAGE_TAG_CLI}" || -n "${COORDINATOR_IMAGE_TAG:-}" ]]; then
    require_env_vars COORDINATOR_IMAGE_REPOSITORY
    image_tag="${EC2_IMAGE_TAG_CLI:-${COORDINATOR_IMAGE_TAG:-}}"
    echo "${COORDINATOR_IMAGE_REPOSITORY}:${image_tag}"
    return
  fi

  if [[ -n "${COORDINATOR_IMAGE:-}" ]]; then
    echo "${COORDINATOR_IMAGE}"
    return
  fi

  require_env_vars COORDINATOR_IMAGE_REPOSITORY
  image_tag="$(default_coordinator_image_tag)"
  echo "${COORDINATOR_IMAGE_REPOSITORY}:${image_tag}"
}

registry_host_from_image() {
  local image_ref="$1"
  echo "${image_ref%%/*}"
}

is_ecr_registry_host() {
  local registry_host="$1"
  [[ "${registry_host}" =~ ^[0-9]{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com$ ]]
}

ecr_region_from_registry_host() {
  local registry_host="$1"

  if [[ "${registry_host}" =~ ^[0-9]{12}\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws\.com$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return
  fi

  echo "unsupported ecr registry host: ${registry_host}" >&2
  exit 1
}

build_ecr_login_command() {
  local registry_host="$1"
  local aws_profile="${2:-}"
  local region

  region="$(ecr_region_from_registry_host "${registry_host}")"

  if [[ -n "${aws_profile}" ]]; then
    printf 'AWS_PROFILE=%q aws ecr get-login-password --region %q | docker login --username AWS --password-stdin %q\n' \
      "${aws_profile}" \
      "${region}" \
      "${registry_host}"
    return
  fi

  printf 'aws ecr get-login-password --region %q | docker login --username AWS --password-stdin %q\n' \
    "${region}" \
    "${registry_host}"
}

ecr_login() {
  local registry_host="$1"
  local aws_profile="${2:-}"
  local region
  local password
  local aws_cmd=(aws)

  region="$(ecr_region_from_registry_host "${registry_host}")"

  if [[ -n "${aws_profile}" ]]; then
    aws_cmd+=(--profile "${aws_profile}")
  fi

  if ! password="$("${aws_cmd[@]}" ecr get-login-password --region "${region}")"; then
    if [[ -n "${aws_profile}" ]]; then
      echo "failed to get ECR login password using AWS profile ${aws_profile}" >&2
      echo "refresh that profile first, for example: aws login --profile ${aws_profile}" >&2
    else
      echo "failed to get ECR login password from local AWS credentials" >&2
    fi
    exit 1
  fi

  printf '%s' "${password}" | docker login --username AWS --password-stdin "${registry_host}"
}

resolve_ec2_instance_id() {
  local host="$1"
  local filter_name="dns-name"
  local instance_id=""

  if [[ "${host}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    filter_name="ip-address"
  fi

  instance_id="$(
    aws ec2 describe-instances \
      --filters "Name=${filter_name},Values=${host}" \
      --query 'Reservations[].Instances[?State.Name!=`terminated`].InstanceId | [0]' \
      --output text 2>/dev/null || true
  )"

  if [[ -z "${instance_id}" || "${instance_id}" == "None" ]]; then
    return 1
  fi

  echo "${instance_id}"
}

ssm_instance_is_online() {
  local instance_id="$1"
  local ping_status=""

  ping_status="$(
    aws ssm describe-instance-information \
      --filters "Key=InstanceIds,Values=${instance_id}" \
      --query 'InstanceInformationList[0].PingStatus' \
      --output text 2>/dev/null || true
  )"

  [[ "${ping_status}" == "Online" ]]
}

validate_coordinator_runtime_env() {
  require_env_vars \
    API_KEY \
    WORKER_INTERNAL_TOKEN \
    LIVEKIT_API_KEY \
    LIVEKIT_API_SECRET \
    COORDINATOR_MODELS_FILE \
    COORDINATOR_CALLBACK_BASE_URL
}
