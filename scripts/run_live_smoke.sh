#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROS_SETUP="/opt/ros/humble/setup.bash"
PROJECT_ROS_SETUP="${REPO_ROOT}/runtime/ros2/install/setup.bash"
ROS2_VENV_ACTIVATE="${REPO_ROOT}/.ros2_venv/bin/activate"
DEFAULT_ROS_LOG_DIR="/tmp/roslog"
STEP_SECONDS="${STEP_SECONDS:-3}"
OBSERVE_SECONDS="${OBSERVE_SECONDS:-1}"
FORWARD_DISTANCE="${FORWARD_DISTANCE:-1.5}"
TURN_DEGREES="${TURN_DEGREES:-45}"
REPORT_PATH="${REPORT_PATH:-integration_command_smoke.json}"
ASK_PROMPT="${ASK_PROMPT:-Where is the target?}"

require_file() {
  local path="$1"
  local message="$2"
  if [[ -f "${path}" ]]; then
    return
  fi
  echo "Error: missing required setup: ${path}" >&2
  echo "${message}" >&2
  exit 1
}

require_file "${ROS_SETUP}" "Install/source ROS2 Humble before running live smoke tests."
require_file "${PROJECT_ROS_SETUP}" "Build or start the FreeAskWorld local ROS2 runtime before running live smoke tests."
require_file "${ROS2_VENV_ACTIVATE}" "Run scripts/setup_envs.sh first so .ros2_venv exists."

if [[ -z "${ROS_LOG_DIR:-}" ]]; then
  export ROS_LOG_DIR="${DEFAULT_ROS_LOG_DIR}"
fi
mkdir -p "${ROS_LOG_DIR}"

# shellcheck disable=SC1091
source "${ROS2_VENV_ACTIVATE}"
set +u
# shellcheck disable=SC1091
source "${ROS_SETUP}"
# shellcheck disable=SC1091
source "${PROJECT_ROS_SETUP}"
set -u

cd "${REPO_ROOT}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

run_and_capture() {
  local name="$1"
  shift
  local out_file="${TMP_DIR}/${name}.json"
  echo
  echo "=== ${name} ==="
  "$@" | tee "${out_file}"
}

summarize_step() {
  local name="$1"
  local out_file="${TMP_DIR}/${name}.json"
  python - <<PY
import json, pathlib
path = pathlib.Path(${out_file@Q})
try:
    data = json.loads(path.read_text())
except Exception as exc:
    print(f"[summary] ${name}: failed to parse JSON: {exc}")
    raise SystemExit(0)
success = data.get('ok')
if success is None:
    success = data.get('accepted')
print(f"[summary] ${name}: success={success} detail={data.get('detail')}")
PY
}

capture_observe() {
  local tag="$1"
  local out_file="${TMP_DIR}/observe_${tag}.json"
  echo
  echo "=== observe (${tag}) ==="
  bash scripts/player_cmd.sh observe "${OBSERVE_SECONDS}" | tee "${out_file}"
  python - <<PY
import json, pathlib
path = pathlib.Path(${out_file@Q})
try:
    data = json.loads(path.read_text())
except Exception as exc:
    print(f"[summary] observe (${tag}): failed to parse JSON: {exc}")
    raise SystemExit(0)
pose = data.get('pose') or {}
pos = pose.get('position') or {}
print(f"[summary] observe (${tag}): rgb={data.get('rgb_available')} depth={data.get('depth_available')} pose=({pos.get('x')}, {pos.get('y')}, {pos.get('z')})")
PY
}

pose_change_summary() {
  local from_tag="$1"
  local to_tag="$2"
  python - <<PY
import json
from math import sqrt
from pathlib import Path

def load_pos(tag):
    path = Path(${TMP_DIR@Q}) / f"observe_{tag}.json"
    data = json.loads(path.read_text())
    pos = ((data.get('pose') or {}).get('position') or {})
    return float(pos.get('x', 0.0)), float(pos.get('y', 0.0)), float(pos.get('z', 0.0))

x1, y1, z1 = load_pos(${from_tag@Q})
x2, y2, z2 = load_pos(${to_tag@Q})
d = sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
print(f"[movement] {${from_tag@Q}} -> {${to_tag@Q}}: delta={d:.4f} from=({x1:.4f},{y1:.4f},{z1:.4f}) to=({x2:.4f},{y2:.4f},{z2:.4f})")
PY
}

sleep_step() {
  local seconds="$1"
  echo "[wait] sleeping ${seconds}s so motion is visible in simulation..."
  sleep "$seconds"
}

run_and_capture "status_before" bash scripts/player_cmd.sh status
capture_observe "before"

run_and_capture "forward" bash scripts/player_cmd.sh forward "${FORWARD_DISTANCE}"
summarize_step "forward"
sleep_step "$STEP_SECONDS"
capture_observe "after_forward"
pose_change_summary "before" "after_forward"

run_and_capture "left" bash scripts/player_cmd.sh left "${TURN_DEGREES}"
summarize_step "left"
sleep_step "$STEP_SECONDS"
capture_observe "after_left"
pose_change_summary "after_forward" "after_left"

run_and_capture "right" bash scripts/player_cmd.sh right "${TURN_DEGREES}"
summarize_step "right"
sleep_step "$STEP_SECONDS"
capture_observe "after_right"
pose_change_summary "after_left" "after_right"

run_and_capture "around" bash scripts/player_cmd.sh around
summarize_step "around"
sleep_step "$STEP_SECONDS"
capture_observe "after_around"
pose_change_summary "after_right" "after_around"

run_and_capture "wait" bash scripts/player_cmd.sh wait "$STEP_SECONDS"
summarize_step "wait"
capture_observe "after_wait"

run_and_capture "ask" bash scripts/player_cmd.sh ask "$ASK_PROMPT"
summarize_step "ask"
capture_observe "after_ask"

run_and_capture "stop" bash scripts/player_cmd.sh stop
summarize_step "stop"
sleep_step 1
capture_observe "after_stop"

python - <<PY
import json
from math import sqrt
from pathlib import Path

TMP = Path(${TMP_DIR@Q})
step_names = ["forward", "left", "right", "around", "wait", "ask", "stop"]
steps = []
for name in step_names:
    data = json.loads((TMP / f"{name}.json").read_text())
    success = data.get("ok")
    if success is None:
        success = data.get("accepted", False)
    steps.append({"name": name, "result": data, "success": bool(success)})

obs_tags = ["before", "after_forward", "after_left", "after_right", "after_around", "after_wait", "after_ask", "after_stop"]
observations = {}
for tag in obs_tags:
    observations[tag] = json.loads((TMP / f"observe_{tag}.json").read_text())

def pos(tag):
    p = ((observations[tag].get("pose") or {}).get("position") or {})
    return float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0))

def delta(a, b):
    x1, y1, z1 = pos(a)
    x2, y2, z2 = pos(b)
    return sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

movement_checks = {
    "forward_moved": delta("before", "after_forward"),
    "left_changed": delta("after_forward", "after_left"),
    "right_changed": delta("after_left", "after_right"),
    "around_changed": delta("after_right", "after_around"),
}

movement_pass = movement_checks["forward_moved"] > 0.05 and any(v > 0.001 for v in movement_checks.values())

report = {
    "step_seconds": float(${STEP_SECONDS@Q}),
    "observe_seconds": float(${OBSERVE_SECONDS@Q}),
    "forward_distance": float(${FORWARD_DISTANCE@Q}),
    "turn_degrees": float(${TURN_DEGREES@Q}),
    "steps": steps,
    "observations": observations,
    "movement_checks": movement_checks,
    "summary": {
        "total_steps": len(steps),
        "ok_steps": sum(1 for s in steps if s["success"]),
        "failed_steps": sum(1 for s in steps if not s["success"]),
        "movement_pass": movement_pass,
    },
}
Path(${REPORT_PATH@Q}).write_text(json.dumps(report, indent=2), encoding="utf-8")
print("\n=== final summary ===")
print(json.dumps(report["summary"], indent=2))
print(json.dumps(report["movement_checks"], indent=2))
print(f"Wrote report to ${REPORT_PATH}")
PY
