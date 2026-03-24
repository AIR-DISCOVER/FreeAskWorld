#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HOST="${FREEASKWORLD_HOST:-127.0.0.1}"
PORT="${FREEASKWORLD_PORT:-8787}"
BRIDGE_PID_FILE="$REPO_ROOT/runtime/.freeaskworld_bridge.pid"
BACKEND_PID_FILE="$REPO_ROOT/runtime/ros2/.ros_tcp_endpoint.pid"

if [[ -f "$BRIDGE_PID_FILE" ]]; then
  BRIDGE_PID="$(cat "$BRIDGE_PID_FILE" || true)"
  echo "bridge_pid=$BRIDGE_PID"
else
  echo "bridge_pid=none"
fi

if [[ -f "$BACKEND_PID_FILE" ]]; then
  BACKEND_PID="$(cat "$BACKEND_PID_FILE" || true)"
  echo "backend_pid=$BACKEND_PID"
else
  echo "backend_pid=none"
fi

if curl -fsS "http://$HOST:$PORT/healthz" 2>/dev/null; then
  echo
else
  echo "health=down"
fi
