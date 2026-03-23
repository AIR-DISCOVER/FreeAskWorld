from dataclasses import dataclass, field
import numpy as np
from typing import Any, Dict, Optional, Tuple

@dataclass
class NavigationCommand:
    LocalPositionOffset: np.ndarray  # shape (3,)
    LocalRotationOffset: np.ndarray  # shape (4,)
    IsStopped: False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "LocalPositionOffset": self.LocalPositionOffset.tolist(),
            "LocalRotationOffset": self.LocalRotationOffset.tolist(),
            "IsStopped": self.IsStopped
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "NavigationCommand":
        return NavigationCommand(
            LocalPositionOffset=np.array(data["LocalPositionOffset"]),
            LocalRotationOffset=np.array(data["LocalRotationOffset"]),
            IsStopped=data.get("IsStopped", False)  # 默认 False
        )

@dataclass
class Step:
    IsStep: bool = True

    def to_dict(self):
        return {
            "IsStep": self.IsStep
        }

    @staticmethod
    def from_dict(data):
        return Step(IsStep=data.get("IsStep", True))


@dataclass
class TransformData:
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float, float]
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'TransformData':
        return TransformData(
            position=tuple(data["position"]),
            rotation=tuple(data["rotation"])
        )


@dataclass
class ObservationSnapshot:
    rgb_available: bool
    depth_available: bool
    transform_data: Optional[TransformData]
    instruction: Optional[str]
    last_rgbd_update_at: Optional[str] = None
    last_json_update_at: Optional[str] = None
    last_observation_update_at: Optional[str] = None
    rgb_shape: Optional[Tuple[int, ...]] = None
    depth_shape: Optional[Tuple[int, ...]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rgb_available": self.rgb_available,
            "depth_available": self.depth_available,
            "transform_data": None if self.transform_data is None else {
                "position": list(self.transform_data.position),
                "rotation": list(self.transform_data.rotation),
            },
            "instruction": self.instruction,
            "last_rgbd_update_at": self.last_rgbd_update_at,
            "last_json_update_at": self.last_json_update_at,
            "last_observation_update_at": self.last_observation_update_at,
            "rgb_shape": None if self.rgb_shape is None else list(self.rgb_shape),
            "depth_shape": None if self.depth_shape is None else list(self.depth_shape),
        }


@dataclass
class BridgeStatus:
    connected: bool
    active_client: Optional[str]
    last_connected_at: Optional[str]
    last_disconnected_at: Optional[str]
    last_command_at: Optional[str]
    last_error_at: Optional[str]
    last_error_message: Optional[str]
    last_command_type: Optional[str]
    last_command_payload: Optional[Dict[str, Any]]
    last_json_type: Optional[str]
    last_rgbd_update_at: Optional[str]
    last_json_update_at: Optional[str]
    last_observation_update_at: Optional[str]
    pending_human_prompt: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "connected": self.connected,
            "active_client": self.active_client,
            "last_connected_at": self.last_connected_at,
            "last_disconnected_at": self.last_disconnected_at,
            "last_command_at": self.last_command_at,
            "last_error_at": self.last_error_at,
            "last_error_message": self.last_error_message,
            "last_command_type": self.last_command_type,
            "last_command_payload": self.last_command_payload,
            "last_json_type": self.last_json_type,
            "last_rgbd_update_at": self.last_rgbd_update_at,
            "last_json_update_at": self.last_json_update_at,
            "last_observation_update_at": self.last_observation_update_at,
            "pending_human_prompt": self.pending_human_prompt,
        }


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


@dataclass
class ActionResult:
    ok: bool
    action: str
    detail: str
    sent: bool = False
    connected: bool = False
    payload: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "action": self.action,
            "detail": self.detail,
            "sent": self.sent,
            "connected": self.connected,
            "payload": self.payload,
            "request_id": self.request_id,
        }
