#!/usr/bin/env bash

set -euo pipefail

# Wrapper for the ROS2-first OpenClaw CLI.
# Live mode depends on the ROS Humble environment; launching from an unsourced
# conda/base shell can load the wrong Python ABI and break rclpy imports.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROS_SETUP="/opt/ros/humble/setup.bash"
PROJECT_ROS_SETUP="/home/wyabz/Project/FreeAskClaw/runtime/ros2/install/setup.bash"
ROS2_VENV_ACTIVATE="${REPO_ROOT}/.ros2_venv/bin/activate"

print_usage() {
  cat <<'EOF'
Usage: scripts/openclaw_ros2_cli.sh [openclaw-cli-args]

Recommended wrapper for live ROS2 commands. It sources the ROS Humble and
project ROS2 setup files before running. If a repo-local `.ros2_venv` exists,
it activates that first:

  python3 -m integrations.openclaw_ros2.cli "$@"

Example:
  scripts/openclaw_ros2_cli.sh --ros2-live status --output-json
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_usage
  exit 0
fi

if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "Error: missing ROS Humble setup: ${ROS_SETUP}" >&2
  echo "This wrapper is intended to prevent unsourced-shell ROS2/rclpy ABI issues." >&2
  exit 1
fi

if [[ ! -f "${PROJECT_ROS_SETUP}" ]]; then
  echo "Error: missing project ROS2 setup: ${PROJECT_ROS_SETUP}" >&2
  echo "Build or install the FreeAskClaw ROS2 runtime before using live OpenClaw commands." >&2
  exit 1
fi

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
exec python3 -m integrations.openclaw_ros2.cli "$@"
