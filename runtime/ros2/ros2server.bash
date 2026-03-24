#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$SCRIPT_DIR"
INSTALL_SETUP="$WORKSPACE_ROOT/install/setup.bash"
PID_FILE="$WORKSPACE_ROOT/.ros_tcp_endpoint.pid"
TCP_PORT="${FREEASKCLAW_ROS2_TCP_PORT:-10000}"
ROS_LOG_DIR_DEFAULT="$WORKSPACE_ROOT/log/ros"

if [[ ! -f "/opt/ros/humble/setup.bash" ]]; then
  echo "ROS2 Humble not found at /opt/ros/humble/setup.bash" >&2
  exit 1
fi

set +u
source /opt/ros/humble/setup.bash
set -u

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-<Disc><DefaultMulticastAddress>0.0.0.0</></>}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-$ROS_LOG_DIR_DEFAULT}"

mkdir -p "$ROS_LOG_DIR"

if [[ ! -f "$INSTALL_SETUP" ]]; then
  colcon build --symlink-install --packages-select simulator_messages ros_tcp_endpoint
fi

set +u
source "$INSTALL_SETUP"
set -u

echo "$$" > "$PID_FILE"
exec ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_TCP_PORT:="$TCP_PORT"
