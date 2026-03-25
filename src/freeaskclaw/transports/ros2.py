from __future__ import annotations

import atexit
import base64
import math
import threading
import time
from dataclasses import dataclass

from freeaskclaw.models import (
    AckUpdate,
    ImageFrame,
    ImageObservation,
    ObservationState,
    PoseObservation,
    SimulatorCommand,
    TaskUpdate,
    TransportUpdates,
)
from freeaskclaw.transports.base import CommandTransport


@dataclass(slots=True)
class Ros2TransportConfig:
    command_topic: str = "/simulator_msg/simulator_command"
    task_topic: str = "/simulator_msg/task"
    ack_topic: str = "/simulator_msg/simulator_command/untiy"
    color_topic: str = "/simulator_msg/camera/color/image_raw"
    depth_topic: str = "/simulator_msg/camera/depth/image_raw"
    odom_topic: str = "/simulator_msg/odom"


class Ros2Transport(CommandTransport):
    name = "ros2"

    def __init__(self, config: Ros2TransportConfig) -> None:
        self._config = config
        self._node = None
        self._command_pub = None
        self._task_pub = None
        self._ack_pub = None
        self._header_cls = None
        self._command_cls = None
        self._string_cls = None
        self._clock = None
        self._rclpy = None
        self._lock = threading.Lock()
        self._init_lock = threading.Lock()
        self._pending_tasks: list[TaskUpdate] = []
        self._pending_acks: list[AckUpdate] = []
        self._observation = ObservationState()
        self._image_bytes: dict[str, bytes | None] = {"color": None, "depth": None}
        self._spin_thread: threading.Thread | None = None
        self._stop_spin = threading.Event()
        self._closed = False
        self._owns_rclpy = False
        atexit.register(self.close)

    def _ensure_ros(self) -> None:
        if self._node is not None:
            return
        with self._init_lock:
            if self._node is not None:
                return
            self._init_ros()

    def _init_ros(self) -> None:
        try:
            import rclpy
            from nav_msgs.msg import Odometry
            from rclpy.node import Node
            from sensor_msgs.msg import Image
            from simulator_messages.msg import SimulatorCommand as RosSimulatorCommand
            from std_msgs.msg import Header, String
        except ImportError as exc:
            raise RuntimeError(
                "ROS2 transport requires rclpy, nav_msgs, sensor_msgs, std_msgs, and simulator_messages to be installed."
            ) from exc

        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True

        class BridgeNode(Node):
            def __init__(self, transport: Ros2Transport, config: Ros2TransportConfig) -> None:
                super().__init__("freeaskclaw_bridge")
                self.command_pub = self.create_publisher(
                    RosSimulatorCommand,
                    config.command_topic,
                    10,
                )
                self.task_pub = self.create_publisher(
                    String,
                    config.task_topic,
                    10,
                )
                self.ack_pub = self.create_publisher(
                    String,
                    config.ack_topic,
                    10,
                )
                self.task_sub = self.create_subscription(
                    String,
                    config.task_topic,
                    transport._on_task,
                    10,
                )
                self.ack_sub = self.create_subscription(
                    String,
                    config.ack_topic,
                    transport._on_ack,
                    10,
                )
                self.color_sub = self.create_subscription(
                    Image,
                    config.color_topic,
                    transport._on_color_image,
                    10,
                )
                self.depth_sub = self.create_subscription(
                    Image,
                    config.depth_topic,
                    transport._on_depth_image,
                    10,
                )
                self.odom_sub = self.create_subscription(
                    Odometry,
                    config.odom_topic,
                    transport._on_odom,
                    10,
                )

        self._node = BridgeNode(self, self._config)
        self._command_pub = self._node.command_pub
        self._task_pub = self._node.task_pub
        self._ack_pub = self._node.ack_pub
        self._command_cls = RosSimulatorCommand
        self._header_cls = Header
        self._string_cls = String
        self._clock = self._node.get_clock
        self._rclpy = rclpy
        self._spin_thread = threading.Thread(target=self._spin_loop, name="freeaskclaw-ros2-spin", daemon=True)
        self._spin_thread.start()

    def _spin_loop(self) -> None:
        while not self._stop_spin.is_set() and self._node is not None and self._rclpy is not None and self._rclpy.ok():
            self._rclpy.spin_once(self._node, timeout_sec=0.1)
            time.sleep(0.01)

    def _on_task(self, msg: object) -> None:
        text = getattr(msg, "data", "").strip()
        if not text:
            return
        with self._lock:
            self._pending_tasks.append(TaskUpdate(text=text))

    def _on_ack(self, msg: object) -> None:
        text = getattr(msg, "data", "").strip()
        if not text:
            return
        with self._lock:
            self._pending_acks.append(AckUpdate(text=text))

    def _on_color_image(self, msg: object) -> None:
        self._update_image("color", self._config.color_topic, msg)

    def _on_depth_image(self, msg: object) -> None:
        self._update_image("depth", self._config.depth_topic, msg)

    def _update_image(self, field_name: str, topic: str, msg: object) -> None:
        image = ImageObservation(
            topic=topic,
            width=int(getattr(msg, "width", 0)),
            height=int(getattr(msg, "height", 0)),
            encoding=str(getattr(msg, "encoding", "")),
            step=int(getattr(msg, "step", 0)),
            is_bigendian=int(getattr(msg, "is_bigendian", 0)),
            data_size=len(getattr(msg, "data", b"")),
        )
        payload = bytes(getattr(msg, "data", b""))
        with self._lock:
            setattr(self._observation, field_name, image)
            self._image_bytes[field_name] = payload

    def _on_odom(self, msg: object) -> None:
        pose = getattr(msg, "pose", None)
        pose_value = getattr(pose, "pose", None)
        position = getattr(pose_value, "position", None)
        orientation = getattr(pose_value, "orientation", None)
        if position is None or orientation is None:
            return

        qx = float(getattr(orientation, "x", 0.0))
        qy = float(getattr(orientation, "y", 0.0))
        qz = float(getattr(orientation, "z", 0.0))
        qw = float(getattr(orientation, "w", 1.0))
        yaw_rad = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        pose_obs = PoseObservation(
            x=float(getattr(position, "x", 0.0)),
            y=float(getattr(position, "y", 0.0)),
            z=float(getattr(position, "z", 0.0)),
            yaw_deg=math.degrees(yaw_rad),
        )
        with self._lock:
            self._observation.pose = pose_obs

    def publish_command(self, command: SimulatorCommand) -> str:
        self._ensure_ros()
        msg = self._command_cls()
        msg.header = self._header_cls()
        msg.header.stamp = self._clock().now().to_msg()
        msg.header.frame_id = command.source
        msg.method = command.method
        msg.method_params = command.method_params
        self._command_pub.publish(msg)
        return f"ros2 published {command.method} to {self._config.command_topic}"

    def publish_task(self, task: TaskUpdate) -> str:
        self._ensure_ros()
        msg = self._string_cls()
        msg.data = task.text
        self._task_pub.publish(msg)
        return f"ros2 published task to {self._config.task_topic}"

    def publish_ack(self, ack: AckUpdate) -> str:
        self._ensure_ros()
        msg = self._string_cls()
        msg.data = ack.text
        self._ack_pub.publish(msg)
        return f"ros2 published ack to {self._config.ack_topic}"

    def drain_updates(self) -> TransportUpdates:
        self._ensure_ros()
        with self._lock:
            tasks = list(self._pending_tasks)
            acks = list(self._pending_acks)
            self._pending_tasks.clear()
            self._pending_acks.clear()
            observation = self._observation.model_copy(deep=True)
        if observation.color is None and observation.depth is None and observation.pose is None:
            observation = None
        return TransportUpdates(tasks=tasks, acks=acks, observation=observation)

    def get_observation(self) -> ObservationState | None:
        self._ensure_ros()
        with self._lock:
            observation = self._observation.model_copy(deep=True)
        if observation.color is None and observation.depth is None and observation.pose is None:
            return None
        return observation

    def get_image_frame(self, image_kind: str) -> ImageFrame | None:
        self._ensure_ros()
        if image_kind not in {"color", "depth"}:
            return None

        with self._lock:
            image = getattr(self._observation, image_kind)
            payload = self._image_bytes.get(image_kind)
            if image is None or payload is None:
                return None

            return ImageFrame(
                topic=image.topic,
                width=image.width,
                height=image.height,
                encoding=image.encoding,
                step=image.step,
                is_bigendian=image.is_bigendian,
                data_size=image.data_size,
                data_base64=base64.b64encode(payload).decode("ascii"),
                created_at=image.created_at,
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_spin.set()
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)
            self._spin_thread = None
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None
        if self._rclpy is not None:
            try:
                if self._owns_rclpy and self._rclpy.ok():
                    self._rclpy.shutdown()
            except Exception:
                pass
