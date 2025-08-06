from dataclasses import dataclass, field
import numpy as np
from typing import Any, Dict, Tuple

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