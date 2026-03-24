# Agent ROS2 Integration (OpenClaw / Claude Code / Codex / custom)

## Current local runtime path

The local Unity FreeAskWorld simulator is currently configured to run in ROS2 mode.

- ROS 2 IP: `127.0.0.1`
- ROS 2 Port: `10000`

For this local configuration, the recommended agent integration path is:

`Agent (OpenClaw / Codex / Claude / custom) -> ROS2-compatible bridge -> ROS TCP Endpoint / ROS2 backend -> Unity FreeAskWorld simulator`

The previously added `closed_loop` websocket Agent bridge remains in the repo, but it is not the active runtime path for the current Unity ROS2 configuration. Treat that websocket path as additive, experimental, and future-facing for this setup.

## Agent compatibility

The bridge interface is agent-agnostic. You can keep one simulator-facing contract and swap upstream agents:

- OpenClaw
- Claude Code
- Codex
- Any custom agent runtime

Only the upstream adapter changes (prompting/tool wrapper/request formatting). The bridge action schema and ROS2 topic mapping stay the same.

## Recommended architecture

The practical repo-side design is:

1. The upstream agent issues a small set of high-level navigation and interaction actions.
2. A ROS2-oriented bridge translates those actions into the simulator's ROS2-facing command/task channels.
3. Unity receives commands through its existing ROS2 communication path without changing the simulator-side ROS2 contract.

Recommended minimum actions:

- `move_forward(distance_m)`
- `turn_left(degrees=30)`
- `turn_right(degrees=30)`
- `turn_around()`
- `stop()`
- `ask_human(prompt)`

Optional utility action:

- `wait(seconds)`

## Recommended observation/state surface

The Agent-facing bridge should expose a narrow status and observation surface:

- `pose`
- RGB availability
- depth availability
- `last_task`
- `last_ack`
- timestamps for command, task, ack, and observation updates

This repo currently provides that surface in two modes:

- scaffold mode: stable CLI/Python contract without live ROS2 delivery
- live transport mode: an optional `rclpy` transport that publishes/subscribes on the simulator-facing ROS2 topics when the repo-owned local runtime or another compatible ROS2 runtime is available

The repository now includes a minimal local runtime path for standalone FreeAskWorld testing, centered around `scripts/start_local_runtime.sh`, `runtime/ros2/`, and `src/freeaskclaw/`.

## ROS2-facing channels

Structure bridge work around these simulator-facing channels:

- `/simulator_msg/simulator_command`
- `/simulator_msg/task`
- `/simulator_msg/simulator_command/untiy`
- `/simulator_msg/camera/color/image_raw`
- `/simulator_msg/camera/depth/image_raw`
- `/simulator_msg/odom`

Notes:

- `/simulator_msg/simulator_command` is the primary command publication target for motion-control actions.
- `/simulator_msg/task` is the recommended publication target for `ask_human` or other explicit task/instruction handoff.
- `/simulator_msg/simulator_command/untiy` is treated here as the simulator-side acknowledgement/status channel name as currently validated externally, even though the segment spelling appears unusual.
- Camera and odometry topics are observation inputs for bridge status and downstream agent logic.

## Action mapping

| Agent action | ROS2 target | Bridge method | Command payload intent |
|---|---|---|---|
| `move_forward(distance_m)` | `/simulator_msg/simulator_command` | `move_forward()` | forward translation request |
| `turn_left(degrees)` | `/simulator_msg/simulator_command` | `turn_left()` | left rotation request |
| `turn_right(degrees)` | `/simulator_msg/simulator_command` | `turn_right()` | right rotation request |
| `turn_around()` | `/simulator_msg/simulator_command` | `turn_around()` | 180 degree rotation request |
| `stop()` | `/simulator_msg/simulator_command` | `stop()` | stop/halt request |
| `ask_human(prompt)` | `/simulator_msg/task` | `ask_human()` | task or prompt handoff |
| `wait(seconds)` | bridge-local or `/simulator_msg/task` | `wait()` | pacing/no-op utility |

## What is implemented in-repo

For environment setup that matches the currently working local ROS2 runtime path, see [`docs/ros2_setup.md`](ros2_setup.md). For the standalone local runtime entrypoint, start with `scripts/start_local_runtime.sh` from the repo root.

The additive ROS2 path under [`integrations/agent_ros2`](../integrations/agent_ros2) provides:

- a documented action and topic contract
- a stable Python bridge interface
- a lightweight CLI
- a transport abstraction that still supports the in-memory scaffold transport for tests
- an optional `RclpyRos2Transport` that publishes action/task envelopes and tracks ack/image/odometry observations through `rclpy`
- smoke tests for action schema, scaffold behavior, and live command validation via `scripts/run_live_smoke.sh`

## Live transport mode

When `rclpy` is available, the live transport currently does the following:

- publishes JSON-serialized `std_msgs/msg/String` command envelopes to `/simulator_msg/simulator_command`
- publishes JSON-serialized `std_msgs/msg/String` task envelopes to `/simulator_msg/task`
- subscribes to `/simulator_msg/simulator_command/untiy` as `std_msgs/msg/String` and exposes that as `last_ack`
- subscribes to `/simulator_msg/camera/color/image_raw` and `/simulator_msg/camera/depth/image_raw` as `sensor_msgs/msg/Image`
- subscribes to `/simulator_msg/odom` as `nav_msgs/msg/Odometry`
- reports `pose`, `rgb_available`, `depth_available`, `last_rgb_at`, `last_depth_at`, `last_odom_at`, and `last_observation_at`

CLI examples:

Short wrapper examples:

```bash
scripts/player_cmd.sh status
scripts/player_cmd.sh observe 1
scripts/player_cmd.sh forward 0.5
scripts/player_cmd.sh left 30
scripts/player_cmd.sh right 30
scripts/player_cmd.sh around
scripts/player_cmd.sh stop
scripts/player_cmd.sh ask "Where is the target?"
```

Low-level wrapper examples:

```bash
python -m integrations.agent_ros2.cli status --output-json
scripts/agent_ros2_cli.sh --ros2-live status --output-json
scripts/agent_ros2_cli.sh --ros2-live observe --wait-seconds 3 --output-json
scripts/agent_ros2_cli.sh --ros2-live move-forward --distance-m 1.0 --output-json
scripts/agent_ros2_cli.sh --ros2-live ask-human "Where is the target?" --output-json
```

For live mode, the wrapper script is the recommended entrypoint because it activates repo-local `.ros2_venv` first if present, then sources:

- `/opt/ros/humble/setup.bash`
- `runtime/ros2/install/setup.bash`

That avoids the common failure mode where `python3 -m integrations.agent_ros2.cli --ros2-live ...` is launched from `(base)` or another mismatched Python environment and `rclpy` fails due to Python ABI mismatch.

## Python environment setup for ROS2 live mode

ROS Humble Python bindings are often built against the system Python ABI. If you launch live mode from conda `base` or another mismatched environment, `rclpy` can fail even when the ROS setup files are sourced correctly.

Recommended pattern:

- create a repo-local `.ros2_venv`, or use another ROS-compatible Python environment
- let `scripts/agent_ros2_cli.sh` auto-activate `.ros2_venv` when it exists
- if `.ros2_venv` is absent, the wrapper still continues, but live ROS2 mode may later fail with a clear `rclpy` import/init error

Example setup:

```bash
cd ~/research/FreeAskWorld
python3.10 -m venv .ros2_venv
source .ros2_venv/bin/activate
source /opt/ros/humble/setup.bash
source runtime/ros2/install/setup.bash
python -c "import rclpy, std_msgs.msg, sensor_msgs.msg, nav_msgs.msg"
```

Validate the wrapper path:

```bash
bash scripts/agent_ros2_cli.sh --ros2-live status --output-json
```

Common failure mode:

- running from `(base)` or another incompatible Python environment can produce `rclpy._rclpy_pybind11` import failures
- if that happens, deactivate the mismatched environment, activate `.ros2_venv` or another ROS-compatible environment, then retry through the wrapper

The `--wait-seconds` option is intended for short-lived live processes where ROS2 subscription callbacks need a moment to populate RGB, depth, and odometry before printing the observation snapshot.

Status output remains honest about three cases:

- no live transport requested: scaffold-only mode
- live transport requested but ROS2 init failed: attached-but-not-ready with explicit error/detail
- live transport requested and ready: `mode=ros2_live` with live topic publishers/subscriptions reported

## External prerequisites

Live mode still depends on runtime pieces outside this repository:

- ROS2 installed and sourced in the shell that launches the CLI/Python process
- a ROS-compatible Python environment when the current shell Python does not match ROS Humble's ABI
- `rclpy`
- standard ROS2 message packages: `std_msgs`, `sensor_msgs`, `nav_msgs`
- a simulator-side ROS2 graph or ROS TCP endpoint already running for the Unity setup configured at `127.0.0.1:10000`

Important limitation:

- this repo does not include Unity-side custom message definitions or a ROS2 endpoint process
- command/task publishing is implemented with JSON `std_msgs/msg/String` payloads because simulator-specific custom message classes are not defined in this repo
- if the Unity side expects different message types, an external ROS2 adapter or message bridge is still required

## What is intentionally not claimed yet

This path does not claim that FreeAskWorld now ships a fully standalone, simulator-complete ROS2 runtime bridge. In particular:

- no simulator-specific ROS2 message definitions are asserted here
- no Unity-side ROS TCP endpoint process is bundled here
- no websocket path is removed or replaced

That is intentional. The current pass is compatibility-first and aligned with the real local Unity runtime path: ROS2 on `127.0.0.1:10000`.
