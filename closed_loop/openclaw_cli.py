import argparse
import json
from typing import Any, Dict

try:
    import uvicorn
except ImportError:  # pragma: no cover - optional dependency
    uvicorn = None

try:
    from .messages import OpenClawAction
    from .openclaw_bridge import bridge
    from .openclaw_server import create_app
except ImportError:  # pragma: no cover - script import fallback
    from messages import OpenClawAction
    from openclaw_bridge import bridge
    from openclaw_server import create_app


def _print_result(result: Dict[str, Any], as_json: bool):
    if as_json:
        print(json.dumps(result, indent=2))
    else:
        print(result)


def build_parser():
    parser = argparse.ArgumentParser(description="OpenClaw-facing CLI for FreeAskWorld closed_loop bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="Run the HTTP bridge server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)

    for command in ("status", "observe"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--json", action="store_true", dest="as_json")

    action_parser = subparsers.add_parser("action", help="Run a generic action")
    action_parser.add_argument("action_name")
    action_parser.add_argument("--params", default="{}", help="JSON object with action parameters")
    action_parser.add_argument("--json", action="store_true", dest="as_json")

    move_parser = subparsers.add_parser("move-forward")
    move_parser.add_argument("--distance-m", type=float, default=1.0)
    move_parser.add_argument("--json", action="store_true", dest="as_json")

    turn_left_parser = subparsers.add_parser("turn-left")
    turn_left_parser.add_argument("--degrees", type=float, default=15.0)
    turn_left_parser.add_argument("--json", action="store_true", dest="as_json")

    turn_right_parser = subparsers.add_parser("turn-right")
    turn_right_parser.add_argument("--degrees", type=float, default=15.0)
    turn_right_parser.add_argument("--json", action="store_true", dest="as_json")

    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("--json", action="store_true", dest="as_json")

    step_parser = subparsers.add_parser("step")
    step_parser.add_argument("--json", action="store_true", dest="as_json")

    ask_parser = subparsers.add_parser("ask-human")
    ask_parser.add_argument("prompt")
    ask_parser.add_argument("--json", action="store_true", dest="as_json")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        if uvicorn is None:
            raise RuntimeError("uvicorn is required for 'serve'. Install closed_loop/requirements.txt first.")
        uvicorn.run(create_app(), host=args.host, port=args.port)
        return 0

    if args.command == "status":
        _print_result(bridge.get_status(), args.as_json)
        return 0

    if args.command == "observe":
        _print_result(bridge.get_observation(), args.as_json)
        return 0

    if args.command == "action":
        result = bridge.perform_action(
            OpenClawAction(action=args.action_name, parameters=json.loads(args.params))
        )
        _print_result(result, args.as_json)
        return 0 if result["ok"] else 1

    if args.command == "move-forward":
        result = bridge.move_forward(distance_m=args.distance_m)
    elif args.command == "turn-left":
        result = bridge.turn_left(degrees=args.degrees)
    elif args.command == "turn-right":
        result = bridge.turn_right(degrees=args.degrees)
    elif args.command == "stop":
        result = bridge.stop()
    elif args.command == "step":
        result = bridge.step()
    elif args.command == "ask-human":
        result = bridge.ask_human(args.prompt)
    else:  # pragma: no cover - argparse prevents this
        parser.error(f"Unsupported command: {args.command}")

    _print_result(result, getattr(args, "as_json", False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
