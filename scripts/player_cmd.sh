#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CLI=(bash "$REPO_ROOT/scripts/agent_ros2_cli.sh" --ros2-live)
BASE_URL="${FREEASKWORLD_BRIDGE_URL:-http://127.0.0.1:8787}"

usage() {
  cat <<'EOF'
Usage:
  scripts/player_cmd.sh status
  scripts/player_cmd.sh observe [wait_seconds]
  scripts/player_cmd.sh forward [distance_m]
  scripts/player_cmd.sh left [degrees]
  scripts/player_cmd.sh right [degrees]
  scripts/player_cmd.sh around
  scripts/player_cmd.sh stop
  scripts/player_cmd.sh wait [seconds]
  scripts/player_cmd.sh ask "your prompt"
  scripts/player_cmd.sh action '{"action":"move_forward","distance_m":1.0}'
EOF
}

require_curl() {
  command -v curl >/dev/null 2>&1 || { echo "curl not found" >&2; exit 1; }
}

http_action() {
  local payload="$1"
  require_curl
  curl -fsS -X POST "${BASE_URL}/v1/openclaw/action" \
    -H 'content-type: application/json' \
    -d "$payload"
}

cmd="${1:-}"
if [[ -z "$cmd" ]]; then
  usage
  exit 1
fi
shift || true

case "$cmd" in
  status)
    exec "${CLI[@]}" status --output-json
    ;;
  observe)
    wait_seconds="${1:-2}"
    exec "${CLI[@]}" observe --wait-seconds "$wait_seconds" --output-json
    ;;
  forward)
    distance="${1:-1.0}"
    exec bash -lc "$(printf 'curl -fsS -X POST %q -H %q -d %q' "${BASE_URL}/v1/openclaw/action" 'content-type: application/json' "{\"action\":\"move_forward\",\"distance_m\":${distance}}")"
    ;;
  left)
    degrees="${1:-30}"
    exec bash -lc "$(printf 'curl -fsS -X POST %q -H %q -d %q' "${BASE_URL}/v1/openclaw/action" 'content-type: application/json' "{\"action\":\"turn_left\",\"degrees\":${degrees}}")"
    ;;
  right)
    degrees="${1:-30}"
    exec bash -lc "$(printf 'curl -fsS -X POST %q -H %q -d %q' "${BASE_URL}/v1/openclaw/action" 'content-type: application/json' "{\"action\":\"turn_right\",\"degrees\":${degrees}}")"
    ;;
  around)
    exec bash -lc "$(printf 'curl -fsS -X POST %q -H %q -d %q' "${BASE_URL}/v1/openclaw/action" 'content-type: application/json' '{"action":"turn_around"}')"
    ;;
  stop)
    exec bash -lc "$(printf 'curl -fsS -X POST %q -H %q -d %q' "${BASE_URL}/v1/openclaw/action" 'content-type: application/json' '{"action":"stop"}')"
    ;;
  wait)
    seconds="${1:-1}"
    exec bash -lc "$(printf 'curl -fsS -X POST %q -H %q -d %q' "${BASE_URL}/v1/openclaw/action" 'content-type: application/json' "{\"action\":\"wait\",\"wait_seconds\":${seconds}}")"
    ;;
  ask)
    prompt="${*:-Where is the target?}"
    exec bash -lc "$(printf 'curl -fsS -X POST %q -H %q -d %q' "${BASE_URL}/v1/openclaw/action" 'content-type: application/json' "{\"action\":\"ask_human\",\"prompt\":\"${prompt}\"}")"
    ;;
  action)
    json_payload="${1:-}"
    if [[ -z "$json_payload" ]]; then
      echo "Error: action requires a JSON payload." >&2
      exit 1
    fi
    exec bash -lc "$(printf 'curl -fsS -X POST %q -H %q -d %q' "${BASE_URL}/v1/openclaw/action" 'content-type: application/json' "$json_payload")"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage
    exit 1
    ;;
esac
