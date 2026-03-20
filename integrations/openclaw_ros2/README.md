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
python -m integrations.openclaw_ros2.cli status --output-json
python -m integrations.openclaw_ros2.cli --ros2-live status --output-json
python -m integrations.openclaw_ros2.cli observe --output-json
python -m integrations.openclaw_ros2.cli --ros2-live observe --wait-seconds 3 --output-json
python -m integrations.openclaw_ros2.cli --ros2-live observe --output-json
python -m integrations.openclaw_ros2.cli move-forward --distance-m 1.0 --output-json
python -m integrations.openclaw_ros2.cli --ros2-live move-forward --distance-m 1.0 --output-json
python -m integrations.openclaw_ros2.cli turn-left --degrees 30 --output-json
python -m integrations.openclaw_ros2.cli turn-around --output-json
python -m integrations.openclaw_ros2.cli stop --output-json
python -m integrations.openclaw_ros2.cli ask-human "Where is the target?" --output-json
python -m integrations.openclaw_ros2.cli action --json '{"action":"move_forward","parameters":{"distance_m":1.0}}' --output-json
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
- standard message packages (`std_msgs`, `sensor_msgs`, `nav_msgs`)
- the simulator-side ROS2 graph or ROS TCP endpoint already running for the local Unity configuration on `127.0.0.1:10000`
- topic/message compatibility on the Unity side for the JSON `String` command/task payloads used here

For short-lived live CLI processes, `observe` can optionally pause before reading subscriptions:

```bash
python -m integrations.openclaw_ros2.cli --ros2-live observe --wait-seconds 3 --output-json
```

Attach a transport that publishes to:

- `/simulator_msg/simulator_command`
- `/simulator_msg/task`

and subscribes to:

- `/simulator_msg/simulator_command/untiy`
- `/simulator_msg/camera/color/image_raw`
- `/simulator_msg/camera/depth/image_raw`
- `/simulator_msg/odom`
