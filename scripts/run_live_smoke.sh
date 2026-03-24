#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROS_SETUP="/opt/ros/humble/setup.bash"
PROJECT_ROS_SETUP="${REPO_ROOT}/runtime/ros2/install/setup.bash"
ROS2_VENV_ACTIVATE="${REPO_ROOT}/.ros2_venv/bin/activate"
DEFAULT_ROS_LOG_DIR="/tmp/roslog"
STEP_SECONDS="${STEP_SECONDS:-2}"
OBSERVE_SECONDS="${OBSERVE_SECONDS:-1}"
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
print(f"[summary] ${name}: ok={data.get('ok')} detail={data.get('detail')}")
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

sleep_step() {
  local seconds="$1"
  echo "[wait] sleeping ${seconds}s so motion is visible in simulation..."
  sleep "$seconds"
}

run_and_capture "status_before" bash scripts/player_cmd.sh status
capture_observe "before"

run_and_capture "forward" bash scripts/player_cmd.sh forward 1.0
summarize_step "forward"
sleep_step "$STEP_SECONDS"
capture_observe "after_forward"

run_and_capture "left" bash scripts/player_cmd.sh left 30
summarize_step "left"
sleep_step "$STEP_SECONDS"
capture_observe "after_left"

run_and_capture "right" bash scripts/player_cmd.sh right 30
summarize_step "right"
sleep_step "$STEP_SECONDS"
capture_observe "after_right"

run_and_capture "around" bash scripts/player_cmd.sh around
summarize_step "around"
sleep_step "$STEP_SECONDS"
capture_observe "after_around"

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
from pathlib import Path
steps = []
for name in ["forward", "left", "right", "around", "wait", "ask", "stop"]:
    path = Path(${TMP_DIR@Q}) / f"{name}.json"
    data = json.loads(path.read_text())
    steps.append({"name": name, "result": data})
report = {
    "step_seconds": float(${STEP_SECONDS@Q}),
    "observe_seconds": float(${OBSERVE_SECONDS@Q}),
    "steps": steps,
    "summary": {
        "total_steps": len(steps),
        "ok_steps": sum(1 for s in steps if s["result"].get("ok")),
        "failed_steps": sum(1 for s in steps if not s["result"].get("ok")),
    },
}
Path(${REPORT_PATH@Q}).write_text(json.dumps(report, indent=2), encoding="utf-8")
print("\n=== final summary ===")
print(json.dumps(report["summary"], indent=2))
print(f"Wrote report to ${REPORT_PATH}")
PY
