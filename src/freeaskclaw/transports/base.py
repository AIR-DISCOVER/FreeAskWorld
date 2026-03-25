from __future__ import annotations

from abc import ABC, abstractmethod

from freeaskclaw.models import (
    AckUpdate,
    ImageFrame,
    ObservationState,
    SimulatorCommand,
    TaskUpdate,
    TransportUpdates,
)


class CommandTransport(ABC):
    name: str

    @abstractmethod
    def publish_command(self, command: SimulatorCommand) -> str:
        raise NotImplementedError

    def publish_task(self, task: TaskUpdate) -> str:
        return f"{self.name}: task cached only"

    def publish_ack(self, ack: AckUpdate) -> str:
        return f"{self.name}: ack cached only"

    def drain_updates(self) -> TransportUpdates:
        return TransportUpdates()

    def get_observation(self) -> ObservationState | None:
        return None

    def get_image_frame(self, image_kind: str) -> ImageFrame | None:
        return None

    def close(self) -> None:
        return None
