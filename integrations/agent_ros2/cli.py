from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from .bridge import AgentRos2Bridge
from .messages import DEFAULT_ROS2_HOST, DEFAULT_ROS2_PORT, OpenClawAction
from .transport_rclpy import RclpyRos2Transport


def _emit(result: Dict[str, Any], output_json: bool) -> None:
    if output_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(result)


def _result_exit_code(result: Dict[str, Any]) -> int:
    return 0 if result["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ROS2-first OpenClaw CLI scaffold for FreeAskWorld"
    )
    parser.add_argument(
        "--ros2-live",
        action="store_true",
        help="Attempt to attach a live rclpy-backed ROS2 transport instead of scaffold-only mode",
    )
    parser.add_argument("--ros2-host", default=DEFAULT_ROS2_HOST)
    parser.add_argument("--ros2-port", type=int, default=DEFAULT_ROS2_PORT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--output-json", action="store_true")

    observe_parser = subparsers.add_parser("observe")
    observe_parser.add_argument(
        "--wait-seconds",
        type=float,
        default=0.0,
        help="Optionally wait before reading observation so live ROS2 subscriptions can accumulate data",
    )
    observe_parser.add_argument("--output-json", action="store_true")

    action_parser = subparsers.add_parser("action", help="Run a generic OpenClaw action from a JSON blob")
    action_parser.add_argument(
        "--json",
        required=True,
        dest="action_json",
        help='JSON object such as {"action":"move_forward","parameters":{"distance_m":1.0}}',
    )
    action_parser.add_argument("--output-json", action="store_true")

    move_parser = subparsers.add_parser("move-forward")
    move_parser.add_argument("--distance-m", type=float, default=1.0)
    move_parser.add_argument("--output-json", action="store_true")

    turn_left_parser = subparsers.add_parser("turn-left")
    turn_left_parser.add_argument("--degrees", type=float, default=30.0)
    turn_left_parser.add_argument("--output-json", action="store_true")

    turn_right_parser = subparsers.add_parser("turn-right")
    turn_right_parser.add_argument("--degrees", type=float, default=30.0)
    turn_right_parser.add_argument("--output-json", action="store_true")

    turn_around_parser = subparsers.add_parser("turn-around")
    turn_around_parser.add_argument("--output-json", action="store_true")

    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("--output-json", action="store_true")

    ask_parser = subparsers.add_parser("ask-human")
    ask_parser.add_argument("prompt")
    ask_parser.add_argument("--output-json", action="store_true")

    wait_parser = subparsers.add_parser("wait")
    wait_parser.add_argument("--seconds", type=float, default=1.0)
    wait_parser.add_argument("--output-json", action="store_true")

    return parser


def _build_bridge(args: argparse.Namespace) -> AgentRos2Bridge:
    if not args.ros2_live:
        return AgentRos2Bridge(ros2_host=args.ros2_host, ros2_port=args.ros2_port)

    transport = RclpyRos2Transport(ros2_host=args.ros2_host, ros2_port=args.ros2_port)
    return AgentRos2Bridge(
        transport=transport,
        ros2_host=args.ros2_host,
        ros2_port=args.ros2_port,
    )


def main(argv: Any = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    local_bridge = _build_bridge(args)

    try:
        if args.command == "status":
            _emit(local_bridge.get_status(), args.output_json)
            return 0

        if args.command == "observe":
            if args.wait_seconds > 0:
                local_bridge.run_wait(args.wait_seconds)
            _emit(local_bridge.get_observation(), args.output_json)
            return 0

        if args.command == "action":
            payload = json.loads(args.action_json)
            result = local_bridge.perform_action(OpenClawAction.from_dict(payload))
            _emit(result, args.output_json)
            return _result_exit_code(result)

        command_handlers = {
            "move-forward": lambda: local_bridge.move_forward(args.distance_m),
            "turn-left": lambda: local_bridge.turn_left(args.degrees),
            "turn-right": lambda: local_bridge.turn_right(args.degrees),
            "turn-around": local_bridge.turn_around,
            "stop": local_bridge.stop,
            "ask-human": lambda: local_bridge.ask_human(args.prompt),
            "wait": lambda: local_bridge.wait(args.seconds),
        }
        handler = command_handlers.get(args.command)
        if handler is None:  # pragma: no cover
            parser.error(f"Unsupported command: {args.command}")
        result = handler()

        _emit(result, args.output_json)
        return _result_exit_code(result)
    finally:
        if local_bridge.transport and hasattr(local_bridge.transport, "close"):
            local_bridge.transport.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
