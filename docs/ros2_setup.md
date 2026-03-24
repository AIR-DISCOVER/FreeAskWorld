# ROS2 Setup Guide for FreeAskWorld (local tested path)

This guide documents the **currently working local ROS2 environment pattern** used with FreeAskWorld.

It is intended for users who want to run the **ROS2-first live integration path**:

`integrations/agent_ros2` + `scripts/agent_ros2_cli.sh`

If you only want a lightweight repo-level smoke check, stop at the scaffold commands in [`README.md`](../README.md) and skip this file.

---

## What this guide assumes

This guide assumes:

- Ubuntu Linux
- ROS2 **Humble**
- Python **3.10**
- repo path like `~/research/FreeAskWorld`
- repo-local ROS env: `.ros2_venv`
- setup files:
  - `/opt/ros/humble/setup.bash`
  - `runtime/ros2/install/setup.bash`

FreeAskWorld now includes a repo-owned local runtime for ROS2 bridge/backend testing. Live operation still depends on system ROS2 support and a working Unity-side ROS2 connection.

---

## 1. Install ROS2 Humble

Install **ROS2 Humble** on Ubuntu first.

Follow the official ROS2 Humble installation instructions for your Ubuntu version, then verify the base setup works:

```bash
source /opt/ros/humble/setup.bash
ros2 --help
```

If `ros2 --help` fails, stop here and fix ROS2 first.

---

## 2. Start with the repo-owned local runtime

The recommended path is to let FreeAskWorld bring up its own local runtime first:

```bash
scripts/start_local_runtime.sh
scripts/status_local_runtime.sh
```

After the local runtime has been built and started at least once, this setup file should exist:

```bash
runtime/ros2/install/setup.bash
```

If either `/opt/ros/humble/setup.bash` or `runtime/ros2/install/setup.bash` is missing, fix that first. If `runtime/ros2/install/setup.bash` does not exist yet, start by running `scripts/start_local_runtime.sh`.

---

## 3. Create a ROS-compatible Python environment

For live ROS2 mode, use a repo-local virtual environment:

```bash
cd ~/research/FreeAskWorld
python3.10 -m venv .ros2_venv
source .ros2_venv/bin/activate
```

Why this matters:

- `rclpy` is sensitive to the active Python ABI.
- A mismatched conda/base environment can break ROS2 imports.
- The wrapper auto-activates `.ros2_venv` when present.

---

## 4. Source ROS2 and the project runtime

After activating `.ros2_venv`, source both setup files:

```bash
source /opt/ros/humble/setup.bash
source runtime/ros2/install/setup.bash
```

Then do a Python import check:

```bash
python -c "import rclpy, std_msgs.msg, sensor_msgs.msg, nav_msgs.msg"
```

If this import check fails, live ROS2 mode will not work yet.

---

## 5. Verify the FreeAskWorld ROS2 wrapper path

This repo provides two command entrypoints:

1. Low-level wrapper:

```bash
cd ~/research/FreeAskWorld
bash scripts/agent_ros2_cli.sh --help
bash scripts/agent_ros2_cli.sh --ros2-live status --output-json
```

2. Short player wrapper:

```bash
scripts/player_cmd.sh status
scripts/player_cmd.sh observe 1
scripts/player_cmd.sh forward 0.5
scripts/player_cmd.sh left 30
scripts/player_cmd.sh stop
```

Expected outcomes:

- `status` should report `transport_ready: true` after the local runtime is up.
- `observe` should show pose / RGB / depth once Unity-side ROS2 data is flowing.
- Action commands should return `ok: true` and `Published ...` when live ROS2 transport is attached.

---

## 6. Minimal live-mode commands

Once the ROS2 environment is healthy, these are the most useful first checks:

```bash
bash scripts/agent_ros2_cli.sh --ros2-live status --output-json
bash scripts/agent_ros2_cli.sh --ros2-live observe --wait-seconds 3 --output-json
bash scripts/agent_ros2_cli.sh --ros2-live move-forward --distance-m 1.0 --output-json
bash scripts/agent_ros2_cli.sh --ros2-live ask-human "Where is the target?" --output-json
```

For a sequential command smoke test that sends the main agent actions and records a JSON report, use the wrapper:

```bash
scripts/run_live_smoke.sh --step-seconds 2 --observe-seconds 1
```

For shorter manual checks, use:

```bash
scripts/player_cmd.sh forward 0.5
scripts/player_cmd.sh left 30
scripts/player_cmd.sh right 30
scripts/player_cmd.sh around
scripts/player_cmd.sh stop
```

This writes `integration_command_smoke.json` in the repo root and helps verify that the agent command surface can actually publish and observe updates in a live ROS2 environment.

Notes:

- `observe --wait-seconds 3` is useful for short-lived CLI runs because ROS2 subscriptions may need a moment before observations appear.
- If the CLI falls back to scaffold behavior, check whether live transport was actually attached.

---

## 7. Common failure cases

### A. `rclpy` import or ABI mismatch

Typical symptom:
- `rclpy` import fails
- `rclpy._rclpy_pybind11` errors
- live mode works in one shell but not another

What to check:
- Are you accidentally in conda `base`?
- Did you activate `.ros2_venv`?
- Did you source `/opt/ros/humble/setup.bash`?
- Did you source `runtime/ros2/install/setup.bash`?

Recommended fix:
- leave the mismatched environment
- activate `.ros2_venv`
- use `bash scripts/agent_ros2_cli.sh ...` instead of calling the live CLI directly

### B. Wrapper says setup file is missing

Typical symptom:
- wrapper exits with a missing setup file error

What to check:
- Does `/opt/ros/humble/setup.bash` exist?
- Does `runtime/ros2/install/setup.bash` exist?

Recommended fix:
- install ROS2 Humble correctly
- run `scripts/start_local_runtime.sh` first so the repo-owned runtime is built

### C. ROS log directory permission error

Typical symptom:
- ROS2 init fails because it cannot write under `~/.ros/log`

Recommended fix:

```bash
mkdir -p /tmp/roslog
ROS_LOG_DIR=/tmp/roslog bash scripts/agent_ros2_cli.sh --ros2-live status --output-json
```

### D. DDS / UDP / shared-memory transport permission issues

Typical symptom:
- Fast DDS / participant creation errors
- UDP socket creation fails
- shared memory transport fails

This often happens in restricted sandboxes, containers, or locked-down environments.

Recommended fix:
- retry on a less restricted host machine
- verify ROS2/DDS networking is allowed
- verify the simulator-side runtime is really running

### E. Live command path works, but no observation data appears

What to check:
- Is the Unity-side ROS2 backend actually publishing to the expected topics?
- Are these channels active?
  - `/simulator_msg/simulator_command`
  - `/simulator_msg/task`
  - `/simulator_msg/simulator_command/untiy`
  - `/simulator_msg/camera/color/image_raw`
  - `/simulator_msg/camera/depth/image_raw`
  - `/simulator_msg/odom`

---

## 8. Recommended workflow for external users

If you are new to this repo, use this order:

1. Run the repo-level smoke checks from `README.md`
2. Install/configure ROS2 Humble
3. Create `.ros2_venv`
4. Run `scripts/start_local_runtime.sh`
5. Confirm `scripts/status_local_runtime.sh` and `/healthz` work
6. Run the live status command through `scripts/agent_ros2_cli.sh`
7. Then move on to Unity-side live control and observation debugging

---

## 9. Related files

- Main overview: [`README.md`](../README.md)
- ROS2 integration design: [`docs/agent_ros2_integration.md`](./agent_ros2_integration.md)
- ROS2 scaffold package: [`integrations/agent_ros2`](../integrations/agent_ros2)
- ROS2 wrapper: [`scripts/agent_ros2_cli.sh`](../scripts/agent_ros2_cli.sh)
- Closed-loop compatibility path: [`closed_loop/README.md`](../closed_loop/README.md)
