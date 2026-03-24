from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_RUNTIME_WORKSPACE = str(Path(__file__).resolve().parents[2])
DEFAULT_RUNTIME_ROS2_START = "bash scripts/start_ros2_backend.sh"
DEFAULT_RUNTIME_ROS2_STOP = "bash scripts/stop_ros2_backend.sh"


@dataclass(slots=True)
class BridgeConfig:
    host: str = "127.0.0.1"
    port: int = 8787
    transport: str = "memory"
    ros_command_topic: str = "/simulator_msg/simulator_command"
    ros_task_topic: str = "/simulator_msg/task"
    ros_ack_topic: str = "/simulator_msg/simulator_command/untiy"
    ros_color_topic: str = "/simulator_msg/camera/color/image_raw"
    ros_depth_topic: str = "/simulator_msg/camera/depth/image_raw"
    ros_odom_topic: str = "/simulator_msg/odom"
    runtime_workspace: str = DEFAULT_RUNTIME_WORKSPACE
    runtime_ros2_start_command: str = DEFAULT_RUNTIME_ROS2_START
    runtime_ros2_stop_command: str = DEFAULT_RUNTIME_ROS2_STOP

    @classmethod
    def from_env(cls) -> "BridgeConfig":
        return cls(
            host=os.getenv("FREEASKCLAW_HOST", "127.0.0.1"),
            port=int(os.getenv("FREEASKCLAW_PORT", "8787")),
            transport=os.getenv("FREEASKCLAW_TRANSPORT", "memory"),
            ros_command_topic=os.getenv(
                "FREEASKCLAW_ROS_COMMAND_TOPIC",
                "/simulator_msg/simulator_command",
            ),
            ros_task_topic=os.getenv(
                "FREEASKCLAW_ROS_TASK_TOPIC",
                "/simulator_msg/task",
            ),
            ros_ack_topic=os.getenv(
                "FREEASKCLAW_ROS_ACK_TOPIC",
                "/simulator_msg/simulator_command/untiy",
            ),
            ros_color_topic=os.getenv(
                "FREEASKCLAW_ROS_COLOR_TOPIC",
                "/simulator_msg/camera/color/image_raw",
            ),
            ros_depth_topic=os.getenv(
                "FREEASKCLAW_ROS_DEPTH_TOPIC",
                "/simulator_msg/camera/depth/image_raw",
            ),
            ros_odom_topic=os.getenv(
                "FREEASKCLAW_ROS_ODOM_TOPIC",
                "/simulator_msg/odom",
            ),
            runtime_workspace=os.getenv(
                "FREEASKCLAW_RUNTIME_WORKSPACE",
                DEFAULT_RUNTIME_WORKSPACE,
            ),
            runtime_ros2_start_command=os.getenv(
                "FREEASKCLAW_RUNTIME_ROS2_START",
                DEFAULT_RUNTIME_ROS2_START,
            ),
            runtime_ros2_stop_command=os.getenv(
                "FREEASKCLAW_RUNTIME_ROS2_STOP",
                DEFAULT_RUNTIME_ROS2_STOP,
            ),
        )
