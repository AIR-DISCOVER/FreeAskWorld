#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HOST="${FREEASKWORLD_HOST:-127.0.0.1}"
PORT="${FREEASKWORLD_PORT:-8787}"
BACKEND_LOG="${FREEASKWORLD_BACKEND_LOG:-$REPO_ROOT/runtime/ros2/backend.log}"
BRIDGE_LOG="${FREEASKWORLD_BRIDGE_LOG:-$REPO_ROOT/runtime/bridge.log}"
BACKEND_PID_FILE="$REPO_ROOT/runtime/ros2/.ros_tcp_endpoint.pid"
BRIDGE_PID_FILE="$REPO_ROOT/runtime/.freeaskworld_bridge.pid"
VENV_ACTIVATE="${FREEASKWORLD_VENV_ACTIVATE:-}"
ROS_HOME_DIR="${FREEASKWORLD_ROS_HOME:-$REPO_ROOT/runtime/ros2/ros_home}"
ROS_LOG_DIR="${FREEASKWORLD_ROS_LOG_DIR:-$ROS_HOME_DIR/log}"

mkdir -p "$REPO_ROOT/runtime/ros2" "$REPO_ROOT/runtime" "$ROS_LOG_DIR"

if [[ -z "$VENV_ACTIVATE" ]]; then
  if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    VENV_ACTIVATE="$REPO_ROOT/.venv/bin/activate"
  elif [[ -f "$REPO_ROOT/.venv-openclaw-smoke/bin/activate" ]]; then
    VENV_ACTIVATE="$REPO_ROOT/.venv-openclaw-smoke/bin/activate"
  else
    echo "Error: no repo-local Python venv found (.venv or .venv-openclaw-smoke)." >&2
    echo "Set FREEASKWORLD_VENV_ACTIVATE to an activate script path and re-run." >&2
    exit 1
  fi
fi

if [[ -f "$BRIDGE_PID_FILE" ]]; then
  BRIDGE_PID="$(cat "$BRIDGE_PID_FILE" || true)"
  if [[ -n "$BRIDGE_PID" ]] && kill -0 "$BRIDGE_PID" 2>/dev/null; then
    echo "FreeAskWorld bridge already running (pid=$BRIDGE_PID)"
    echo "health: http://$HOST:$PORT/healthz"
    exit 0
  fi
  rm -f "$BRIDGE_PID_FILE"
fi

echo "Stopping any stale FreeAskWorld ROS2 backend before restart..."
bash "$REPO_ROOT/scripts/stop_ros2_backend.sh" >/dev/null 2>&1 || true

echo "Starting repo-owned ROS2 backend..."
nohup env ROS_HOME="$ROS_HOME_DIR" ROS_LOG_DIR="$ROS_LOG_DIR" bash "$REPO_ROOT/scripts/start_ros2_backend.sh" >"$BACKEND_LOG" 2>&1 &
BACKEND_LAUNCH_PID=$!
echo "Backend launcher pid=$BACKEND_LAUNCH_PID log=$BACKEND_LOG"

for _ in {1..20}; do
  if [[ -f "$BACKEND_PID_FILE" ]]; then
    break
  fi
  sleep 1
done

echo "Starting FreeAskWorld bridge on http://$HOST:$PORT ..."
nohup env ROS_HOME="$ROS_HOME_DIR" ROS_LOG_DIR="$ROS_LOG_DIR" bash -lc "cd '$REPO_ROOT' && source '$VENV_ACTIVATE' && python -m pip install -e . >/dev/null 2>&1 || true && set +u && source /opt/ros/humble/setup.bash && source '$REPO_ROOT/runtime/ros2/install/setup.bash' && set -u && exec python -m freeaskclaw.cli serve --transport ros2 --host '$HOST' --port '$PORT'" >"$BRIDGE_LOG" 2>&1 &
BRIDGE_PID=$!
echo "$BRIDGE_PID" > "$BRIDGE_PID_FILE"
echo "Bridge pid=$BRIDGE_PID log=$BRIDGE_LOG"

sleep 3
if curl -fsS "http://$HOST:$PORT/healthz" >/dev/null 2>&1; then
  echo "FreeAskWorld local runtime is up: http://$HOST:$PORT"
  echo "Try: curl http://$HOST:$PORT/v1/observation"
else
  echo "Bridge health check failed. Inspect: $BRIDGE_LOG" >&2
  exit 1
fi
