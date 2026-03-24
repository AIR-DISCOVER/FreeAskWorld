#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROS2_WORKSPACE="${FREEASKWORLD_ROS2_WORKSPACE:-${REPO_ROOT}/runtime/ros2}"
ROS2_STOP_SCRIPT="${FREEASKWORLD_ROS2_STOP_SCRIPT:-${ROS2_WORKSPACE}/stop_ros.bash}"

if [[ ! -f "$ROS2_STOP_SCRIPT" ]]; then
  echo "FreeAskWorld ROS2 runtime stop script not found: $ROS2_STOP_SCRIPT" >&2
  echo "Set FREEASKWORLD_ROS2_WORKSPACE or FREEASKWORLD_ROS2_STOP_SCRIPT to the ROS2 backend stop entrypoint." >&2
  exit 1
fi

cd "$(dirname "$ROS2_STOP_SCRIPT")"
exec bash "$(basename "$ROS2_STOP_SCRIPT")"
