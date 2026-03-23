from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

DEFAULT_ROS2_HOST = "127.0.0.1"
DEFAULT_ROS2_PORT = 10000

TOPIC_SIMULATOR_COMMAND = "/simulator_msg/simulator_command"
TOPIC_TASK = "/simulator_msg/task"
TOPIC_ACK = "/simulator_msg/simulator_command/untiy"
TOPIC_CAMERA_COLOR = "/simulator_msg/camera/color/image_raw"
TOPIC_CAMERA_DEPTH = "/simulator_msg/camera/depth/image_raw"
TOPIC_ODOM = "/simulator_msg/odom"

OBSERVATION_TOPICS = (
    TOPIC_CAMERA_COLOR,
    TOPIC_CAMERA_DEPTH,
    TOPIC_ODOM,
)

ACTION_TOPIC_MAP = {
    "move_forward": TOPIC_SIMULATOR_COMMAND,
    "turn_left": TOPIC_SIMULATOR_COMMAND,
    "turn_right": TOPIC_SIMULATOR_COMMAND,
    "turn_around": TOPIC_SIMULATOR_COMMAND,
    "stop": TOPIC_SIMULATOR_COMMAND,
    "ask_human": TOPIC_TASK,
    "wait": TOPIC_TASK,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AgentAction:
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "parameters": self.parameters,
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AgentAction":
        return cls(
            action=payload["action"],
            parameters=payload.get("parameters", {}) or {},
            request_id=payload.get("request_id"),
        )


@dataclass
class CommandEnvelope:
    action: str
    topic: str
    parameters: Dict[str, Any]
    request_id: Optional[str] = None
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "topic": self.topic,
            "parameters": self.parameters,
            "request_id": self.request_id,
            "created_at": self.created_at,
        }


@dataclass
class BridgeStatus:
    mode: str
    ros2_host: str
    ros2_port: int
    command_topic: str
    task_topic: str
    ack_topic: str
    observation_topics: List[str]
    transport_ready: bool
    last_command: Optional[Dict[str, Any]] = None
    last_task: Optional[Dict[str, Any]] = None
    last_ack: Optional[Dict[str, Any]] = None
    last_command_at: Optional[str] = None
    last_task_at: Optional[str] = None
    last_ack_at: Optional[str] = None
    detail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "ros2_host": self.ros2_host,
            "ros2_port": self.ros2_port,
            "command_topic": self.command_topic,
            "task_topic": self.task_topic,
            "ack_topic": self.ack_topic,
            "observation_topics": self.observation_topics,
            "transport_ready": self.transport_ready,
            "last_command": self.last_command,
            "last_task": self.last_task,
            "last_ack": self.last_ack,
            "last_command_at": self.last_command_at,
            "last_task_at": self.last_task_at,
            "last_ack_at": self.last_ack_at,
            "detail": self.detail,
        }


@dataclass
class ObservationSnapshot:
    pose: Optional[Dict[str, Any]]
    rgb_available: bool
    depth_available: bool
    last_task: Optional[Dict[str, Any]]
    last_ack: Optional[Dict[str, Any]]
    last_observation_at: Optional[str] = None
    last_rgb_at: Optional[str] = None
    last_depth_at: Optional[str] = None
    last_odom_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pose": self.pose,
            "rgb_available": self.rgb_available,
            "depth_available": self.depth_available,
            "last_task": self.last_task,
            "last_ack": self.last_ack,
            "last_observation_at": self.last_observation_at,
            "last_rgb_at": self.last_rgb_at,
            "last_depth_at": self.last_depth_at,
            "last_odom_at": self.last_odom_at,
        }


@dataclass
class ActionResult:
    ok: bool
    action: str
    detail: str
    topic: str
    scaffolded: bool
    request_id: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "action": self.action,
            "detail": self.detail,
            "topic": self.topic,
            "scaffolded": self.scaffolded,
            "request_id": self.request_id,
            "payload": self.payload,
        }
