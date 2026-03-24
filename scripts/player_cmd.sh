#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CLI=(bash "$REPO_ROOT/scripts/agent_ros2_cli.sh" --ros2-live)

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
  scripts/player_cmd.sh action '{"action":"move_forward","parameters":{"distance_m":1.0}}'
EOF
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
    exec "${CLI[@]}" move-forward --distance-m "$distance" --output-json
    ;;
  left)
    degrees="${1:-30}"
    exec "${CLI[@]}" turn-left --degrees "$degrees" --output-json
    ;;
  right)
    degrees="${1:-30}"
    exec "${CLI[@]}" turn-right --degrees "$degrees" --output-json
    ;;
  around)
    exec "${CLI[@]}" turn-around --output-json
    ;;
  stop)
    exec "${CLI[@]}" stop --output-json
    ;;
  wait)
    seconds="${1:-1}"
    exec "${CLI[@]}" wait --seconds "$seconds" --output-json
    ;;
  ask)
    prompt="${*:-Where is the target?}"
    exec "${CLI[@]}" ask-human "$prompt" --output-json
    ;;
  action)
    json_payload="${1:-}"
    if [[ -z "$json_payload" ]]; then
      echo "Error: action requires a JSON payload." >&2
      exit 1
    fi
    exec "${CLI[@]}" action --json "$json_payload" --output-json
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
