from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from .messages import (
    ACTION_TOPIC_MAP,
    DEFAULT_ROS2_HOST,
    DEFAULT_ROS2_PORT,
    OBSERVATION_TOPICS,
    TOPIC_ACK,
    TOPIC_SIMULATOR_COMMAND,
    TOPIC_TASK,
    ActionResult,
    BridgeStatus,
    CommandEnvelope,
    ObservationSnapshot,
    AgentAction,
    utc_now_iso,
)
from .transport_rclpy import RclpyRos2Transport


class Ros2BridgeTransport(Protocol):
    def is_ready(self) -> bool:
        ...

    def publish(self, topic: str, payload: Dict[str, Any]) -> bool:
        ...

    def get_status(self) -> Dict[str, Any]:
        ...

    def get_observation(self) -> Dict[str, Any]:
        ...


@dataclass
class InMemoryRos2ScaffoldTransport:
    """Minimal local transport for tests and scaffolding."""

    ready: bool = False

    def __post_init__(self) -> None:
        self.last_published_topic: Optional[str] = None
        self.last_published_payload: Optional[Dict[str, Any]] = None
        self.last_ack: Optional[Dict[str, Any]] = None
        self.pose: Optional[Dict[str, Any]] = None
        self.rgb_available = False
        self.depth_available = False
        self.last_rgb_at: Optional[str] = None
        self.last_depth_at: Optional[str] = None
        self.last_odom_at: Optional[str] = None
        self.last_observation_at: Optional[str] = None

    def is_ready(self) -> bool:
        return self.ready

    def publish(self, topic: str, payload: Dict[str, Any]) -> bool:
        self.last_published_topic = topic
        self.last_published_payload = payload
        return self.ready

    def get_status(self) -> Dict[str, Any]:
        return {
            "transport_ready": self.ready,
            "last_ack": self.last_ack,
        }

    def get_observation(self) -> Dict[str, Any]:
        return {
            "pose": self.pose,
            "rgb_available": self.rgb_available,
            "depth_available": self.depth_available,
            "last_ack": self.last_ack,
            "last_observation_at": self.last_observation_at,
            "last_rgb_at": self.last_rgb_at,
            "last_depth_at": self.last_depth_at,
            "last_odom_at": self.last_odom_at,
        }


class AgentRos2Bridge:
    """Compatibility-first agent surface aligned to Unity's ROS2 runtime path."""

    def __init__(
        self,
        transport: Optional[Ros2BridgeTransport] = None,
        ros2_host: str = DEFAULT_ROS2_HOST,
        ros2_port: int = DEFAULT_ROS2_PORT,
    ) -> None:
        self.transport = transport
        self.ros2_host = ros2_host
        self.ros2_port = ros2_port
        self.last_command: Optional[Dict[str, Any]] = None
        self.last_task: Optional[Dict[str, Any]] = None
        self.last_ack: Optional[Dict[str, Any]] = None
        self.last_command_at: Optional[str] = None
        self.last_task_at: Optional[str] = None
        self.last_ack_at: Optional[str] = None

    def get_status(self) -> Dict[str, Any]:
        transport_status = self.transport.get_status() if self.transport else {}
        self._refresh_last_ack(transport_status.get("last_ack"))
        transport_ready = bool(self.transport and self.transport.is_ready())

        status = BridgeStatus(
            mode="ros2_live" if transport_ready else "ros2_scaffold",
            ros2_host=self.ros2_host,
            ros2_port=self.ros2_port,
            command_topic=TOPIC_SIMULATOR_COMMAND,
            task_topic=TOPIC_TASK,
            ack_topic=TOPIC_ACK,
            observation_topics=list(OBSERVATION_TOPICS),
            transport_ready=transport_ready,
            last_command=self.last_command,
            last_task=self.last_task,
            last_ack=self.last_ack,
            last_command_at=self.last_command_at,
            last_task_at=self.last_task_at,
            last_ack_at=self.last_ack_at,
            detail=self._status_detail(transport_status),
        )
        merged_status = status.to_dict()
        merged_status.update({key: value for key, value in transport_status.items() if key != "last_ack"})
        return merged_status

    def get_observation(self) -> Dict[str, Any]:
        transport_observation = self.transport.get_observation() if self.transport else {}
        observation = ObservationSnapshot(
            pose=transport_observation.get("pose"),
            rgb_available=bool(transport_observation.get("rgb_available", False)),
            depth_available=bool(transport_observation.get("depth_available", False)),
            last_task=self.last_task,
            last_ack=transport_observation.get("last_ack", self.last_ack),
            last_observation_at=transport_observation.get("last_observation_at"),
            last_rgb_at=transport_observation.get("last_rgb_at"),
            last_depth_at=transport_observation.get("last_depth_at"),
            last_odom_at=transport_observation.get("last_odom_at"),
        )
        return observation.to_dict()

    def move_forward(self, distance_m: float, request_id: Optional[str] = None) -> Dict[str, Any]:
        return self._dispatch(
            action_name="move_forward",
            parameters={"distance_m": float(distance_m)},
            request_id=request_id,
        )

    def turn_left(self, degrees: float = 30.0, request_id: Optional[str] = None) -> Dict[str, Any]:
        return self._dispatch(
            action_name="turn_left",
            parameters={"degrees": float(degrees)},
            request_id=request_id,
        )

    def turn_right(self, degrees: float = 30.0, request_id: Optional[str] = None) -> Dict[str, Any]:
        return self._dispatch(
            action_name="turn_right",
            parameters={"degrees": float(degrees)},
            request_id=request_id,
        )

    def turn_around(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        return self._dispatch(
            action_name="turn_around",
            parameters={"degrees": 180.0},
            request_id=request_id,
        )

    def stop(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        return self._dispatch(action_name="stop", parameters={}, request_id=request_id)

    def ask_human(self, prompt: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        return self._dispatch(
            action_name="ask_human",
            parameters={"prompt": prompt},
            request_id=request_id,
        )

    def wait(self, seconds: float, request_id: Optional[str] = None) -> Dict[str, Any]:
        return self._dispatch(
            action_name="wait",
            parameters={"seconds": float(seconds)},
            request_id=request_id,
        )

    def perform_action(self, action: AgentAction) -> Dict[str, Any]:
        action_name = action.action
        request_id = action.request_id or str(uuid.uuid4())
        parameters = action.parameters or {}

        if action_name == "move_forward":
            return self.move_forward(parameters.get("distance_m", 1.0), request_id=request_id)
        if action_name == "turn_left":
            return self.turn_left(parameters.get("degrees", 30.0), request_id=request_id)
        if action_name == "turn_right":
            return self.turn_right(parameters.get("degrees", 30.0), request_id=request_id)
        if action_name == "turn_around":
            return self.turn_around(request_id=request_id)
        if action_name == "stop":
            return self.stop(request_id=request_id)
        if action_name == "ask_human":
            return self.ask_human(parameters.get("prompt", ""), request_id=request_id)
        if action_name == "wait":
            return self.wait(parameters.get("seconds", 0.0), request_id=request_id)

        return ActionResult(
            ok=False,
            action=action_name,
            detail=f"Unsupported ROS2 bridge action: {action_name}",
            topic="",
            scaffolded=True,
            request_id=request_id,
            payload=parameters,
        ).to_dict()

    def _dispatch(self, action_name: str, parameters: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
        topic = ACTION_TOPIC_MAP[action_name]
        envelope = CommandEnvelope(
            action=action_name,
            topic=topic,
            parameters=parameters,
            request_id=request_id or str(uuid.uuid4()),
        )
        payload = envelope.to_dict()
        published = bool(self.transport and self.transport.publish(topic, payload))
        published_at = utc_now_iso()

        if topic == TOPIC_TASK:
            self.last_task = payload
            self.last_task_at = published_at
        else:
            self.last_command = payload
            self.last_command_at = published_at

        detail = (
            f"Published to {topic}"
            if published
            else self._publish_failure_detail(topic)
        )
        return ActionResult(
            ok=published,
            action=action_name,
            detail=detail,
            topic=topic,
            scaffolded=not published,
            request_id=envelope.request_id,
            payload=payload,
        ).to_dict()

    def run_wait(self, seconds: float) -> None:
        time.sleep(seconds)

    def _refresh_last_ack(self, last_ack: Optional[Dict[str, Any]]) -> None:
        if last_ack is None:
            return
        self.last_ack = last_ack
        self.last_ack_at = utc_now_iso()

    def _status_detail(self, transport_status: Dict[str, Any]) -> Optional[str]:
        if not self.transport:
            return "ROS2 scaffold only; attach a transport backed by ROS2 or ROS TCP to publish live commands"

        transport_detail = transport_status.get("transport_detail")
        if transport_detail:
            return str(transport_detail)
        if self.transport.is_ready():
            return "Live ROS2 transport attached"
        return "ROS2 transport is attached but not ready"

    def _publish_failure_detail(self, topic: str) -> str:
        if not self.transport:
            return f"ROS2 scaffold only; no live transport attached for topic {topic}"

        transport_detail = self.transport.get_status().get("transport_detail")
        if transport_detail:
            return f"{transport_detail} (topic {topic})"
        return f"ROS2 transport was unable to publish to topic {topic}"


bridge = AgentRos2Bridge()
