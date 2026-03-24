#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROS2_WORKSPACE="${FREEASKWORLD_ROS2_WORKSPACE:-${REPO_ROOT}/runtime/ros2}"
ROS2_START_SCRIPT="${FREEASKWORLD_ROS2_START_SCRIPT:-${ROS2_WORKSPACE}/ros2server.bash}"

if [[ ! -f "$ROS2_START_SCRIPT" ]]; then
  echo "FreeAskWorld ROS2 runtime start script not found: $ROS2_START_SCRIPT" >&2
  echo "Set FREEASKWORLD_ROS2_WORKSPACE or FREEASKWORLD_ROS2_START_SCRIPT to the ROS2 backend entrypoint." >&2
  exit 1
fi

cd "$(dirname "$ROS2_START_SCRIPT")"
exec bash "$(basename "$ROS2_START_SCRIPT")"
