from __future__ import annotations

from freeaskclaw.models import AckUpdate, SimulatorCommand, TaskUpdate
from freeaskclaw.transports.base import CommandTransport


class MemoryTransport(CommandTransport):
    name = "memory"

    def __init__(self) -> None:
        self.commands: list[SimulatorCommand] = []
        self.tasks: list[TaskUpdate] = []
        self.acks: list[AckUpdate] = []

    def publish_command(self, command: SimulatorCommand) -> str:
        self.commands.append(command)
        return f"memory accepted command #{len(self.commands)}"

    def publish_task(self, task: TaskUpdate) -> str:
        self.tasks.append(task)
        return f"memory accepted task #{len(self.tasks)}"

    def publish_ack(self, ack: AckUpdate) -> str:
        self.acks.append(ack)
        return f"memory accepted ack #{len(self.acks)}"
