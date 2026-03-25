from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


UNITY_METHODS = {
    "move",
    "forward",
    "turnleft",
    "turnright",
    "turnaround",
    "ask",
    "wait",
    "stop",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SimulatorCommand(BaseModel):
    method: str = Field(description="Unity BenchmarkPlayer command method.")
    method_params: str = Field(default="", description="Serialized parameters string.")
    source: str = Field(default="freeaskclaw")
    created_at: str = Field(default_factory=utc_now_iso)

    @field_validator("method")
    @classmethod
    def normalize_method(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in UNITY_METHODS:
            raise ValueError(f"unsupported method: {value}")
        return normalized


class TaskUpdate(BaseModel):
    text: str
    source: str = "unity"
    created_at: str = Field(default_factory=utc_now_iso)


class AckUpdate(BaseModel):
    text: str
    source: str = "unity"
    created_at: str = Field(default_factory=utc_now_iso)


class ImageObservation(BaseModel):
    topic: str
    width: int
    height: int
    encoding: str = ""
    step: int = 0
    is_bigendian: int = 0
    data_size: int
    created_at: str = Field(default_factory=utc_now_iso)


class ImageFrame(BaseModel):
    topic: str
    width: int
    height: int
    encoding: str = ""
    step: int = 0
    is_bigendian: int = 0
    data_size: int
    data_base64: str
    created_at: str = Field(default_factory=utc_now_iso)


class PoseObservation(BaseModel):
    x: float
    y: float
    z: float
    yaw_deg: float
    created_at: str = Field(default_factory=utc_now_iso)


class ObservationState(BaseModel):
    color: ImageObservation | None = None
    depth: ImageObservation | None = None
    pose: PoseObservation | None = None


class TransportUpdates(BaseModel):
    tasks: list[TaskUpdate] = Field(default_factory=list)
    acks: list[AckUpdate] = Field(default_factory=list)
    observation: ObservationState | None = None


class BridgeState(BaseModel):
    transport: str
    last_task: TaskUpdate | None = None
    last_ack: AckUpdate | None = None
    last_command: SimulatorCommand | None = None
    last_observation: ObservationState | None = None
    command_count: int = 0
    task_count: int = 0
    ack_count: int = 0


class CommandResult(BaseModel):
    accepted: bool = True
    transport: str
    command: SimulatorCommand
    detail: str


class OpenClawAction(BaseModel):
    action: str = Field(description="Normalized or OpenClaw-facing action name.")
    distance_m: float | None = None
    degrees: float | None = None
    x: float | None = None
    y: float | None = None
    yaw_deg: float | None = None
    prompt: str | None = None
    wait_seconds: float | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @field_validator("action")
    @classmethod
    def normalize_action(cls, value: str) -> str:
        return value.strip().lower()


class Capability(BaseModel):
    transport: str
    methods: list[str]
    openclaw_actions: list[str]
    reverse_channels: list[str]


class HealthStatus(BaseModel):
    status: Literal["ok"] = "ok"
    transport: str


class RuntimeProcessStatus(BaseModel):
    name: str
    command: str
    cwd: str
    running: bool
    pid: int | None = None


class RuntimeStatus(BaseModel):
    processes: list[RuntimeProcessStatus]


class RuntimeActionResult(BaseModel):
    accepted: bool = True
    detail: str
    status: RuntimeStatus
