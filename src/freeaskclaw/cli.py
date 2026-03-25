from __future__ import annotations

import argparse
import json

from freeaskclaw.config import BridgeConfig
from freeaskclaw.models import AckUpdate, OpenClawAction, SimulatorCommand, TaskUpdate
from freeaskclaw.service import BridgeService, build_transport


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="freeaskclaw")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--transport", default="memory", choices=["memory", "ros2"])
    serve.add_argument("--ros-topic", default="/simulator_msg/simulator_command")
    serve.add_argument("--ros-task-topic", default="/simulator_msg/task")
    serve.add_argument("--ros-ack-topic", default="/simulator_msg/simulator_command/untiy")
    serve.add_argument("--ros-color-topic", default="/simulator_msg/camera/color/image_raw")
    serve.add_argument("--ros-depth-topic", default="/simulator_msg/camera/depth/image_raw")
    serve.add_argument("--ros-odom-topic", default="/simulator_msg/odom")

    send = sub.add_parser("send")
    send.add_argument("--transport", default="memory", choices=["memory", "ros2"])
    send.add_argument("--method", required=True)
    send.add_argument("--params", default="")
    send.add_argument("--ros-topic", default="/simulator_msg/simulator_command")
    send.add_argument("--source", default="freeaskclaw-cli")

    task = sub.add_parser("task")
    task.add_argument("--transport", default="memory", choices=["memory", "ros2"])
    task.add_argument("--text", required=True)
    task.add_argument("--ros-task-topic", default="/simulator_msg/task")
    task.add_argument("--source", default="freeaskclaw-cli")

    ack = sub.add_parser("ack")
    ack.add_argument("--transport", default="memory", choices=["memory", "ros2"])
    ack.add_argument("--text", required=True)
    ack.add_argument("--ros-ack-topic", default="/simulator_msg/simulator_command/untiy")
    ack.add_argument("--source", default="freeaskclaw-cli")

    action = sub.add_parser("action")
    action.add_argument("--transport", default="memory", choices=["memory", "ros2"])
    action.add_argument("--json", required=True, help="OpenClaw action JSON payload.")
    action.add_argument("--ros-topic", default="/simulator_msg/simulator_command")
    action.add_argument("--ros-task-topic", default="/simulator_msg/task")

    observe = sub.add_parser("observe")
    observe.add_argument("--transport", default="memory", choices=["memory", "ros2"])
    observe.add_argument("--ros-task-topic", default="/simulator_msg/task")
    observe.add_argument("--ros-ack-topic", default="/simulator_msg/simulator_command/untiy")
    observe.add_argument("--ros-color-topic", default="/simulator_msg/camera/color/image_raw")
    observe.add_argument("--ros-depth-topic", default="/simulator_msg/camera/depth/image_raw")
    observe.add_argument("--ros-odom-topic", default="/simulator_msg/odom")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.cmd == "serve":
        import uvicorn

        from freeaskclaw.server import create_app

        config = BridgeConfig(
            host=args.host,
            port=args.port,
            transport=args.transport,
            ros_command_topic=args.ros_topic,
            ros_task_topic=args.ros_task_topic,
            ros_ack_topic=args.ros_ack_topic,
            ros_color_topic=args.ros_color_topic,
            ros_depth_topic=args.ros_depth_topic,
            ros_odom_topic=args.ros_odom_topic,
        )
        uvicorn.run(create_app(config), host=config.host, port=config.port)
        return

    if args.cmd == "send":
        config = BridgeConfig(
            transport=args.transport,
            ros_command_topic=args.ros_topic,
        )
        service = BridgeService(build_transport(config))
        result = service.send_command(
            SimulatorCommand(
                method=args.method,
                method_params=args.params,
                source=args.source,
            )
        )
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
        return

    if args.cmd == "observe":
        config = BridgeConfig(
            transport=args.transport,
            ros_task_topic=args.ros_task_topic,
            ros_ack_topic=args.ros_ack_topic,
            ros_color_topic=args.ros_color_topic,
            ros_depth_topic=args.ros_depth_topic,
            ros_odom_topic=args.ros_odom_topic,
        )
        service = BridgeService(build_transport(config))
        print(json.dumps(service.state().model_dump(), ensure_ascii=False, indent=2))
        return

    if args.cmd == "task":
        config = BridgeConfig(
            transport=args.transport,
            ros_task_topic=args.ros_task_topic,
        )
        service = BridgeService(build_transport(config))
        detail = service.publish_task(TaskUpdate(text=args.text, source=args.source))
        print(json.dumps({"accepted": True, "detail": detail}, ensure_ascii=False, indent=2))
        return

    if args.cmd == "ack":
        config = BridgeConfig(
            transport=args.transport,
            ros_ack_topic=args.ros_ack_topic,
        )
        service = BridgeService(build_transport(config))
        detail = service.publish_ack(AckUpdate(text=args.text, source=args.source))
        print(json.dumps({"accepted": True, "detail": detail}, ensure_ascii=False, indent=2))
        return

    if args.cmd == "action":
        payload = json.loads(args.json)
        config = BridgeConfig(
            transport=args.transport,
            ros_command_topic=args.ros_topic,
            ros_task_topic=args.ros_task_topic,
        )
        service = BridgeService(build_transport(config))
        result = service.send_openclaw_action(OpenClawAction(**payload))
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
