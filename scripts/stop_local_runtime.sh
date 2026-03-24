#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BRIDGE_PID_FILE="$REPO_ROOT/runtime/.freeaskworld_bridge.pid"

if [[ -f "$BRIDGE_PID_FILE" ]]; then
  BRIDGE_PID="$(cat "$BRIDGE_PID_FILE" || true)"
  if [[ -n "$BRIDGE_PID" ]] && kill -0 "$BRIDGE_PID" 2>/dev/null; then
    echo "Stopping FreeAskWorld bridge (pid=$BRIDGE_PID)..."
    kill "$BRIDGE_PID" 2>/dev/null || true
    sleep 1
  fi
  rm -f "$BRIDGE_PID_FILE"
fi

pkill -f 'python -m freeaskclaw.cli serve --transport ros2' 2>/dev/null || true

echo "Stopping repo-owned ROS2 backend..."
bash "$REPO_ROOT/scripts/stop_ros2_backend.sh" || true

echo "Stopped."
