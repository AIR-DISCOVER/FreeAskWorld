#!/usr/bin/env bash

set -euo pipefail

# Wrapper for the ROS2-first agent CLI (Codex/Claude/custom/OpenClaw).
# Live mode depends on the ROS Humble environment; unsourced shells can load
# the wrong Python ABI and break rclpy imports.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROS_SETUP="/opt/ros/humble/setup.bash"
PROJECT_ROS_SETUP="/home/wyabz/Project/FreeAskClaw/runtime/ros2/install/setup.bash"
ROS2_VENV_ACTIVATE="${REPO_ROOT}/.ros2_venv/bin/activate"

print_usage() {
  cat <<'EOF'
Usage: scripts/agent_ros2_cli.sh [agent-cli-args]

Recommended wrapper for live ROS2 commands. It sources the ROS Humble and
project ROS2 setup files before running. If a repo-local `.ros2_venv` exists,
it activates that first:

  python3 -m integrations.agent_ros2.cli "$@"

Example:
  scripts/agent_ros2_cli.sh --ros2-live status --output-json
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_usage
  exit 0
fi

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

require_file "${ROS_SETUP}" \
  "This wrapper is intended to prevent unsourced-shell ROS2/rclpy ABI issues."
require_file "${PROJECT_ROS_SETUP}" \
  "Build or install the FreeAskClaw ROS2 runtime before using live agent commands."

if [[ -f "${ROS2_VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1091
  source "${ROS2_VENV_ACTIVATE}"
fi

# ROS setup scripts may read unset environment variables internally.
set +u
# shellcheck disable=SC1091
source "${ROS_SETUP}"
# shellcheck disable=SC1091
source "${PROJECT_ROS_SETUP}"
set -u

cd "${REPO_ROOT}"
exec python3 -m integrations.agent_ros2.cli "$@"
