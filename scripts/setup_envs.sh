#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
ROS2_VENV_DIR="${REPO_ROOT}/.ros2_venv"
ROS2_ACTIVATE="${ROS2_VENV_DIR}/bin/activate"
ROS2_SETUP_HINT="${REPO_ROOT}/docs/ros2_setup.md"

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

if ! have_cmd "$PYTHON_BIN"; then
  echo "Error: ${PYTHON_BIN} not found." >&2
  echo "Install Python 3.10 first, then re-run this script." >&2
  exit 1
fi

cd "$REPO_ROOT"

if [[ -f "$ROS2_ACTIVATE" ]]; then
  echo "Found existing ROS2 env: ${ROS2_VENV_DIR}"
  echo "Reusing it."
else
  echo "Creating ROS2 env at ${ROS2_VENV_DIR} using ${PYTHON_BIN} ..."
  "$PYTHON_BIN" -m venv "$ROS2_VENV_DIR"
fi

# shellcheck disable=SC1091
source "$ROS2_ACTIVATE"

python - <<'PY'
import importlib.util, sys
mods = ["fastapi", "uvicorn", "pydantic", "numpy"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print("MISSING=" + ",".join(missing))
PY

MISSING=$(python - <<'PY'
import importlib.util
mods = ["fastapi", "uvicorn", "pydantic", "numpy"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print(" ".join(missing))
PY
)

if [[ -n "$MISSING" ]]; then
  echo "Installing missing Python packages into .ros2_venv: $MISSING"
  python -m pip install --upgrade pip
  python -m pip install fastapi uvicorn 'pydantic>=2.8,<3' numpy
else
  echo ".ros2_venv already has the required Python packages."
fi

if [[ -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Ensuring local runtime package is available from repo source ..."
  PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
  export PYTHONPATH
fi

cat <<EOF

Environment setup complete.

Primary environment:
  ${ROS2_VENV_DIR}

Next steps:
  1) Install ROS2 Humble manually if not installed yet.
     See: ${ROS2_SETUP_HINT}
  2) Activate env:
     source .ros2_venv/bin/activate
  3) Start local runtime:
     scripts/start_local_runtime.sh
  4) Check health:
     scripts/status_local_runtime.sh
  5) Run smoke test:
     scripts/run_live_smoke.sh --step-seconds 2 --observe-seconds 1
EOF
