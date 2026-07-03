from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from integrations.agent_ros2.bridge import AgentRos2Bridge
from integrations.agent_ros2.transport_rclpy import RclpyRos2Transport


def _now() -> float:
    return time.time()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a minimal live ROS2 smoke test for FreeAskWorld agent commands"
    )
    parser.add_argument("--ros2-host", default="127.0.0.1")
    parser.add_argument("--ros2-port", type=int, default=10000)
    parser.add_argument(
        "--step-seconds",
        type=float,
        default=2.0,
        help="Seconds to wait after each command before reading observation/status",
    )
    parser.add_argument(
        "--observe-seconds",
        type=float,
        default=1.0,
        help="Extra wait before each observation snapshot",
    )
    parser.add_argument(
        "--ask-prompt",
        default="Where is the target?",
        help="Prompt used for the ask-human step",
    )
    parser.add_argument(
        "--output-json",
        default="integration_command_smoke.json",
        help="Where to write the JSON report",
    )
    return parser


def _record_step(
    bridge: AgentRos2Bridge,
    name: str,
    command_result: Dict[str, Any],
    step_seconds: float,
    observe_seconds: float,
) -> Dict[str, Any]:
    started_at = _now()
    if step_seconds > 0:
        bridge.run_wait(step_seconds)
    if observe_seconds > 0:
        bridge.run_wait(observe_seconds)
    observation = bridge.get_observation()
    status = bridge.get_status()
    finished_at = _now()
    return {
        "name": name,
        "command_result": command_result,
        "duration_seconds": round(finished_at - started_at, 3),
        "observation": observation,
        "status": status,
    }


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    transport = RclpyRos2Transport(ros2_host=args.ros2_host, ros2_port=args.ros2_port)
    bridge = AgentRos2Bridge(
        transport=transport,
        ros2_host=args.ros2_host,
        ros2_port=args.ros2_port,
    )

    report: Dict[str, Any] = {
        "ros2_host": args.ros2_host,
        "ros2_port": args.ros2_port,
        "step_seconds": args.step_seconds,
        "observe_seconds": args.observe_seconds,
        "initial_status": bridge.get_status(),
        "steps": [],
    }

    try:
        actions = [
            ("move_forward", lambda: bridge.move_forward(1.0)),
            ("turn_left", lambda: bridge.turn_left(30.0)),
            ("turn_right", lambda: bridge.turn_right(30.0)),
            ("turn_around", bridge.turn_around),
            ("stop", bridge.stop),
            ("ask_human", lambda: bridge.ask_human(args.ask_prompt)),
            ("wait", lambda: bridge.wait(args.step_seconds)),
        ]

        for name, fn in actions:
            command_result = fn()
            report["steps"].append(
                _record_step(
                    bridge=bridge,
                    name=name,
                    command_result=command_result,
                    step_seconds=args.step_seconds,
                    observe_seconds=args.observe_seconds,
                )
            )

        report["final_status"] = bridge.get_status()
        report["summary"] = {
            "total_steps": len(report["steps"]),
            "ok_steps": sum(1 for step in report["steps"] if step["command_result"].get("ok")),
            "failed_steps": sum(1 for step in report["steps"] if not step["command_result"].get("ok")),
        }
    finally:
        if hasattr(transport, "close"):
            transport.close()

    output_path = Path(args.output_json)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"Wrote report to {output_path}")
    return 0 if report["summary"]["failed_steps"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
