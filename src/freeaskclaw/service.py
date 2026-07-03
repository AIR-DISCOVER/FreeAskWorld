from __future__ import annotations

from freeaskclaw.config import BridgeConfig
from freeaskclaw.models import (
    AckUpdate,
    BridgeState,
    Capability,
    CommandResult,
    ObservationState,
    OpenClawAction,
    SimulatorCommand,
    TaskUpdate,
)
from freeaskclaw.transports.base import CommandTransport
from freeaskclaw.transports.memory import MemoryTransport
from freeaskclaw.transports.ros2 import Ros2Transport, Ros2TransportConfig


OPENCLAW_ACTIONS = [
    "move_forward",
    "turn_left",
    "turn_right",
    "turn_around",
    "move_relative",
    "ask_human",
    "wait",
    "stop",
]


def build_transport(config: BridgeConfig) -> CommandTransport:
    transport = config.transport.lower()
    if transport == "memory":
        return MemoryTransport()
    if transport == "ros2":
        return Ros2Transport(
            Ros2TransportConfig(
                command_topic=config.ros_command_topic,
                task_topic=config.ros_task_topic,
                ack_topic=config.ros_ack_topic,
                color_topic=config.ros_color_topic,
                depth_topic=config.ros_depth_topic,
                odom_topic=config.ros_odom_topic,
            )
        )
    raise ValueError(f"unsupported transport: {config.transport}")


class BridgeService:
    def __init__(self, transport: CommandTransport) -> None:
        self.transport = transport
        self.last_task: TaskUpdate | None = None
        self.last_ack: AckUpdate | None = None
        self.last_command: SimulatorCommand | None = None
        self.last_observation: ObservationState | None = None
        self.command_count = 0
        self.task_count = 0
        self.ack_count = 0

    def capabilities(self) -> Capability:
        return Capability(
            transport=self.transport.name,
            methods=sorted(
                [
                    "ask",
                    "forward",
                    "move",
                    "stop",
                    "turnaround",
                    "turnleft",
                    "turnright",
                    "wait",
                ]
            ),
            openclaw_actions=OPENCLAW_ACTIONS,
            reverse_channels=["task", "ack"],
        )

    def state(self) -> BridgeState:
        self._refresh_from_transport()
        return BridgeState(
            transport=self.transport.name,
            last_task=self.last_task,
            last_ack=self.last_ack,
            last_command=self.last_command,
            last_observation=self.last_observation,
            command_count=self.command_count,
            task_count=self.task_count,
            ack_count=self.ack_count,
        )

    def publish_task(self, task: TaskUpdate) -> str:
        self.last_task = task
        self.task_count += 1
        return self.transport.publish_task(task)

    def publish_ack(self, ack: AckUpdate) -> str:
        self.last_ack = ack
        self.ack_count += 1
        return self.transport.publish_ack(ack)

    def send_command(self, command: SimulatorCommand) -> CommandResult:
        detail = self.transport.publish_command(command)
        self.last_command = command
        self.command_count += 1
        return CommandResult(
            transport=self.transport.name,
            command=command,
            detail=detail,
        )

    def send_openclaw_action(self, action: OpenClawAction) -> CommandResult:
        return self.send_command(self._translate_action(action))

    def _refresh_from_transport(self) -> None:
        updates = self.transport.drain_updates()
        for task in updates.tasks:
            self.last_task = task
            self.task_count += 1
        for ack in updates.acks:
            self.last_ack = ack
            self.ack_count += 1
        if updates.observation is not None:
            self.last_observation = updates.observation

    def _translate_action(self, action: OpenClawAction) -> SimulatorCommand:
        name = action.action
        if name in {"move_forward", "forward"}:
            distance = 1.0 if action.distance_m is None else action.distance_m
            return SimulatorCommand(method="forward", method_params=f"{distance}")
        if name in {"turn_left", "left"}:
            degrees = 30.0 if action.degrees is None else action.degrees
            return SimulatorCommand(method="turnleft", method_params=f"{degrees}")
        if name in {"turn_right", "right"}:
            degrees = 30.0 if action.degrees is None else action.degrees
            return SimulatorCommand(method="turnright", method_params=f"{degrees}")
        if name in {"turn_around", "turnaround"}:
            return SimulatorCommand(method="turnaround", method_params="")
        if name in {"move_relative", "move"}:
            x = 0.0 if action.x is None else action.x
            y = 0.0 if action.y is None else action.y
            yaw_deg = 0.0 if action.yaw_deg is None else action.yaw_deg
            return SimulatorCommand(method="move", method_params=f"{x},{y},{yaw_deg}")
        if name in {"ask_human", "ask"}:
            prompt = "" if action.prompt is None else action.prompt
            return SimulatorCommand(method="ask", method_params=prompt)
        if name == "wait":
            wait_seconds = 1.0 if action.wait_seconds is None else action.wait_seconds
            return SimulatorCommand(method="wait", method_params=f"{wait_seconds}")
        if name == "stop":
            return SimulatorCommand(method="stop", method_params="")
        raise ValueError(f"unsupported openclaw action: {action.action}")
