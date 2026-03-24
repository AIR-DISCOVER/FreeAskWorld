#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROS_SETUP="/opt/ros/humble/setup.bash"
PROJECT_ROS_SETUP="/home/wyabz/Project/FreeAskClaw/runtime/ros2/install/setup.bash"
ROS2_VENV_ACTIVATE="${REPO_ROOT}/.ros2_venv/bin/activate"
DEFAULT_ROS_LOG_DIR="/tmp/roslog"

require_file() {
  local path="$1"
  local message="$2"
  if [[ -f "${path}" ]]; then
    return
  fi
  echo "Error: missing required setup: ${path}" >&2
  echo "${message}" >&2
  exit 1
}

require_file "${ROS_SETUP}" "Install/source ROS2 Humble before running live smoke tests."
require_file "${PROJECT_ROS_SETUP}" "Build or install the FreeAskClaw ROS2 runtime before running live smoke tests."
require_file "${ROS2_VENV_ACTIVATE}" "Create or move a repo-local .ros2_venv into FreeAskWorld before running live smoke tests."

if [[ -z "${ROS_LOG_DIR:-}" ]]; then
  export ROS_LOG_DIR="${DEFAULT_ROS_LOG_DIR}"
fi
if ! mkdir -p "${ROS_LOG_DIR}" 2>/dev/null; then
  echo "Error: failed to create ROS log directory: ${ROS_LOG_DIR}" >&2
  echo "Set ROS_LOG_DIR to a writable path and re-run." >&2
  exit 1
fi
if [[ ! -w "${ROS_LOG_DIR}" ]]; then
  echo "Error: ROS log directory is not writable: ${ROS_LOG_DIR}" >&2
  echo "Set ROS_LOG_DIR to a writable path and re-run." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${ROS2_VENV_ACTIVATE}"
set +u
# shellcheck disable=SC1091
source "${ROS_SETUP}"
# shellcheck disable=SC1091
source "${PROJECT_ROS_SETUP}"
set -u

cd "${REPO_ROOT}"
exec python -m integrations.agent_ros2.live_command_smoke "$@"
