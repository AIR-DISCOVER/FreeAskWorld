from __future__ import annotations

import importlib
import json
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .messages import (
    DEFAULT_ROS2_HOST,
    DEFAULT_ROS2_PORT,
    TOPIC_ACK,
    TOPIC_CAMERA_COLOR,
    TOPIC_CAMERA_DEPTH,
    TOPIC_ODOM,
    TOPIC_SIMULATOR_COMMAND,
    TOPIC_TASK,
    utc_now_iso,
)


@dataclass
class RclpyRos2Transport:
    """ROS2 transport backed by rclpy when the local runtime is available."""

    ros2_host: str = DEFAULT_ROS2_HOST
    ros2_port: int = DEFAULT_ROS2_PORT
    node_name: str = "agent_ros2_bridge"
    qos_depth: int = 10
    spin_timeout_sec: float = 0.1
    _modules: Optional[Dict[str, Any]] = None
    _import_error: Optional[str] = field(init=False, default=None)
    _init_error: Optional[str] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._ready = False
        self._runtime_available = False
        self._detail = (
            "ROS2 transport not initialized; install/source ROS2 with rclpy and standard message packages"
        )
        self._warnings: List[str] = []
        self._subscriptions_ready: List[str] = []
        self._publishers_ready: List[str] = []
        self._last_publish_error: Optional[str] = None
        self._last_published_topic: Optional[str] = None
        self._last_published_payload: Optional[Dict[str, Any]] = None
        self.last_ack: Optional[Dict[str, Any]] = None
        self.pose: Optional[Dict[str, Any]] = None
        self.rgb_available = False
        self.depth_available = False
        self.last_rgb_at: Optional[str] = None
        self.last_depth_at: Optional[str] = None
        self.last_odom_at: Optional[str] = None
        self.last_observation_at: Optional[str] = None
        self._executor = None
        self._executor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._node = None
        self._owns_rclpy_context = False
        self._rclpy = None
        self._publishers: Dict[str, Any] = {}

        modules = self._modules or self._load_modules()
        if modules is None:
            return

        self._runtime_available = True
        self._rclpy = modules["rclpy"]
        try:
            self._initialize_runtime(modules)
        except Exception as exc:  # pragma: no cover - depends on ROS2 runtime behavior
            self._init_error = str(exc)
            self._detail = f"ROS2 transport initialization failed: {exc}"
            self.close()

    def _load_modules(self) -> Optional[Dict[str, Any]]:
        try:
            return {
                "rclpy": importlib.import_module("rclpy"),
                "node": importlib.import_module("rclpy.node"),
                "executors": importlib.import_module("rclpy.executors"),
                "std_msgs": importlib.import_module("std_msgs.msg"),
                "sensor_msgs": importlib.import_module("sensor_msgs.msg"),
                "nav_msgs": importlib.import_module("nav_msgs.msg"),
            }
        except Exception as exc:
            self._import_error = str(exc)
            self._detail = (
                "ROS2 transport unavailable: could not import rclpy/std ROS2 messages. "
                f"Import error: {exc}"
            )
            return None

    def _initialize_runtime(self, modules: Dict[str, Any]) -> None:
        rclpy = modules["rclpy"]
        node_module = modules["node"]
        executors_module = modules["executors"]
        std_msgs = modules["std_msgs"]
        sensor_msgs = modules["sensor_msgs"]
        nav_msgs = modules["nav_msgs"]

        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy_context = True

        node_cls = getattr(node_module, "Node")
        executor_cls = getattr(executors_module, "SingleThreadedExecutor")
        string_cls = getattr(std_msgs, "String")
        image_cls = getattr(sensor_msgs, "Image")
        odom_cls = getattr(nav_msgs, "Odometry")

        self._node = node_cls(f"{self.node_name}_{uuid.uuid4().hex[:8]}")
        self._publishers = {
            TOPIC_SIMULATOR_COMMAND: self._node.create_publisher(string_cls, TOPIC_SIMULATOR_COMMAND, self.qos_depth),
            TOPIC_TASK: self._node.create_publisher(string_cls, TOPIC_TASK, self.qos_depth),
        }
        self._publishers_ready = list(self._publishers)

        self._node.create_subscription(string_cls, TOPIC_ACK, self._on_ack, self.qos_depth)
        self._subscriptions_ready.append(TOPIC_ACK)
        self._node.create_subscription(image_cls, TOPIC_CAMERA_COLOR, self._on_rgb, self.qos_depth)
        self._subscriptions_ready.append(TOPIC_CAMERA_COLOR)
        self._node.create_subscription(image_cls, TOPIC_CAMERA_DEPTH, self._on_depth, self.qos_depth)
        self._subscriptions_ready.append(TOPIC_CAMERA_DEPTH)
        self._node.create_subscription(odom_cls, TOPIC_ODOM, self._on_odom, self.qos_depth)
        self._subscriptions_ready.append(TOPIC_ODOM)

        self._string_cls = string_cls
        self._executor = executor_cls()
        self._executor.add_node(self._node)
        self._executor_thread = threading.Thread(target=self._spin_loop, name="agent-ros2-spin", daemon=True)
        self._executor_thread.start()

        self._ready = True
        self._detail = (
            f"Live ROS2 transport attached via rclpy node {self._node.get_name()} "
            f"(Unity ROS2 config remains {self.ros2_host}:{self.ros2_port})"
        )

    def _spin_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._executor.spin_once(timeout_sec=self.spin_timeout_sec)
            except Exception as exc:  # pragma: no cover - runtime dependent
                self._init_error = str(exc)
                self._detail = f"ROS2 executor stopped: {exc}"
                self._ready = False
                return

    def is_ready(self) -> bool:
        return self._ready

    def publish(self, topic: str, payload: Dict[str, Any]) -> bool:
        if not self._ready:
            return False

        publisher = self._publishers.get(topic)
        if publisher is None:
            self._last_publish_error = f"No ROS2 publisher is configured for topic {topic}"
            self._detail = self._last_publish_error
            return False

        try:
            message = self._string_cls()
            message.data = json.dumps(payload, sort_keys=True)
            publisher.publish(message)
            self._last_published_topic = topic
            self._last_published_payload = payload
            self._last_publish_error = None
            return True
        except Exception as exc:  # pragma: no cover - runtime dependent
            self._last_publish_error = str(exc)
            self._detail = f"ROS2 publish failed for topic {topic}: {exc}"
            return False

    def get_status(self) -> Dict[str, Any]:
        return {
            "transport_kind": "rclpy",
            "transport_ready": self._ready,
            "transport_live": self._runtime_available,
            "transport_detail": self._detail,
            "transport_error": self._init_error or self._import_error or self._last_publish_error,
            "ros2_runtime_available": self._runtime_available,
            "publisher_topics": list(self._publishers_ready),
            "subscription_topics": list(self._subscriptions_ready),
            "warnings": list(self._warnings),
            "last_ack": self.last_ack,
            "last_published_topic": self._last_published_topic,
            "last_published_payload": self._last_published_payload,
            "requested_ros2_host": self.ros2_host,
            "requested_ros2_port": self.ros2_port,
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

    def close(self) -> None:
        self._ready = False
        self._stop_event.set()
        if self._executor_thread and self._executor_thread.is_alive():
            self._executor_thread.join(timeout=1.0)
        if self._executor is not None:
            try:
                self._executor.shutdown()
            except Exception:  # pragma: no cover - runtime dependent
                pass
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:  # pragma: no cover - runtime dependent
                pass
        if self._owns_rclpy_context and self._rclpy is not None:
            try:
                self._rclpy.shutdown()
            except Exception:  # pragma: no cover - runtime dependent
                pass

    def _on_ack(self, msg: Any) -> None:
        payload = self._decode_string_message(msg)
        payload.setdefault("received_at", utc_now_iso())
        self.last_ack = payload
        self.last_observation_at = payload["received_at"]

    def _on_rgb(self, msg: Any) -> None:
        self._mark_image_observed(
            channel="rgb",
            warning_text="RGB topic callback received an unexpected message shape",
            msg=msg,
        )

    def _on_depth(self, msg: Any) -> None:
        self._mark_image_observed(
            channel="depth",
            warning_text="Depth topic callback received an unexpected message shape",
            msg=msg,
        )

    def _mark_image_observed(self, channel: str, warning_text: str, msg: Any) -> None:
        observed_at = utc_now_iso()
        if channel == "rgb":
            self.rgb_available = True
            self.last_rgb_at = observed_at
        else:
            self.depth_available = True
            self.last_depth_at = observed_at
        self.last_observation_at = observed_at
        self._warnings = [warning for warning in self._warnings if channel not in warning.lower()]
        if not hasattr(msg, "height") or not hasattr(msg, "width"):
            self._warnings.append(warning_text)

    def _on_odom(self, msg: Any) -> None:
        observed_at = utc_now_iso()
        pose = getattr(getattr(msg, "pose", None), "pose", None)
        position = getattr(pose, "position", None)
        orientation = getattr(pose, "orientation", None)

        self.pose = {
            "position": {
                "x": getattr(position, "x", None),
                "y": getattr(position, "y", None),
                "z": getattr(position, "z", None),
            },
            "orientation": {
                "x": getattr(orientation, "x", None),
                "y": getattr(orientation, "y", None),
                "z": getattr(orientation, "z", None),
                "w": getattr(orientation, "w", None),
            },
        }
        self.last_odom_at = observed_at
        self.last_observation_at = observed_at

    @staticmethod
    def _decode_string_message(msg: Any) -> Dict[str, Any]:
        raw = getattr(msg, "data", "")
        if not isinstance(raw, str):
            return {"raw": raw}
        try:
            decoded = json.loads(raw)
        except Exception:
            return {"raw": raw}
        if isinstance(decoded, dict):
            return decoded
        return {"raw": raw, "decoded": decoded}
