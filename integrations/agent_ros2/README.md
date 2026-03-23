# OpenClaw ROS2 Scaffold

This directory defines the ROS2-first OpenClaw integration surface for the current local FreeAskWorld Unity setup.

## Scope

- Preserves the simulator's existing ROS2 communication contract
- Targets local Unity ROS2 mode on `127.0.0.1:10000`
- Keeps the `closed_loop` websocket bridge unchanged and available
- Provides a stable Python and CLI surface even when no live ROS2 transport is attached
- Adds an optional `rclpy` transport path for live ROS2 publish/subscribe when the external ROS2 runtime is available

## Files

- `bridge.py`: OpenClaw-facing bridge API with transport abstraction
- `transport_rclpy.py`: optional live ROS2 transport backed by `rclpy`
- `messages.py`: constants, action schema, command envelope, status/observation dataclasses
- `cli.py`: lightweight CLI scaffold
- `tests/test_bridge.py`: scaffold and live-transport fallback tests

## Example commands

```bash
python -m integrations.agent_ros2.cli status --output-json
python -m integrations.agent_ros2.cli observe --output-json
python -m integrations.agent_ros2.cli move-forward --distance-m 1.0 --output-json
python -m integrations.agent_ros2.cli turn-left --degrees 30 --output-json
python -m integrations.agent_ros2.cli turn-around --output-json
python -m integrations.agent_ros2.cli stop --output-json
python -m integrations.agent_ros2.cli ask-human "Where is the target?" --output-json
python -m integrations.agent_ros2.cli action --json '{"action":"move_forward","parameters":{"distance_m":1.0}}' --output-json

# Recommended for live ROS2 commands from shells that have not already sourced ROS.
scripts/agent_ros2_cli.sh --ros2-live status --output-json
scripts/agent_ros2_cli.sh --ros2-live observe --wait-seconds 3 --output-json
scripts/agent_ros2_cli.sh --ros2-live move-forward --distance-m 1.0 --output-json
```

## ROS2 Python environment

For `--ros2-live`, prefer the repo wrapper script:

```bash
scripts/agent_ros2_cli.sh --ros2-live status --output-json
```

The wrapper now activates repo-local `.ros2_venv` first if it exists, then sources:

- `/opt/ros/humble/setup.bash`
- `/home/wyabz/Project/FreeAskClaw/runtime/ros2/install/setup.bash`

Why this may be needed:

- ROS Humble `rclpy` commonly expects the system Python ABI
- launching from conda `(base)` or another mismatched environment can break `rclpy._rclpy_pybind11`

Minimal setup:

```bash
cd ~/research/FreeAskWorld
python3.10 -m venv .ros2_venv
source .ros2_venv/bin/activate
source /opt/ros/humble/setup.bash
source /home/wyabz/Project/FreeAskClaw/runtime/ros2/install/setup.bash
python -c "import rclpy, std_msgs.msg, sensor_msgs.msg, nav_msgs.msg"
```

If `.ros2_venv` is not present, the wrapper does not fail early. It continues and lets the live ROS2 path report the actual import/init failure if the active Python environment is incompatible.

Validate the wrapper path:

```bash
scripts/agent_ros2_cli.sh --ros2-live status --output-json
```

## Runtime note

If no transport is attached, the CLI and Python bridge return explicit scaffold responses instead of claiming live simulator delivery.

If `--ros2-live` is used and `rclpy` plus the standard ROS2 message packages are importable, the bridge will:

- publish JSON `std_msgs/msg/String` envelopes to `/simulator_msg/simulator_command`
- publish JSON `std_msgs/msg/String` envelopes to `/simulator_msg/task`
- subscribe to `/simulator_msg/simulator_command/untiy` as `std_msgs/msg/String`
- subscribe to camera topics as `sensor_msgs/msg/Image`
- subscribe to odometry as `nav_msgs/msg/Odometry`

External prerequisites still apply:

- a sourced ROS2 environment with `rclpy`
- a ROS-compatible Python environment such as repo-local `.ros2_venv` when the current shell Python is incompatible
- standard message packages (`std_msgs`, `sensor_msgs`, `nav_msgs`)
- the simulator-side ROS2 graph or ROS TCP endpoint already running for the local Unity configuration on `127.0.0.1:10000`
- topic/message compatibility on the Unity side for the JSON `String` command/task payloads used here

For short-lived live CLI processes, `observe` can optionally pause before reading subscriptions:

```bash
scripts/agent_ros2_cli.sh --ros2-live observe --wait-seconds 3 --output-json
```

Attach a transport that publishes to:

- `/simulator_msg/simulator_command`
- `/simulator_msg/task`

and subscribes to:

- `/simulator_msg/simulator_command/untiy`
- `/simulator_msg/camera/color/image_raw`
- `/simulator_msg/camera/depth/image_raw`
- `/simulator_msg/odom`
