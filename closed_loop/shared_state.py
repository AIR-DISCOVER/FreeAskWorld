from datetime import datetime, timezone
from typing import Any, Dict, Optional
import numpy as np
try:
    from messages import *
except ImportError:  # pragma: no cover - package import fallback
    from .messages import *

# 可共享的变量（全局缓存）
rgb_array: Optional[np.ndarray] = None
depth_array: Optional[np.ndarray] = None
transform_data: Optional[TransformData] = None
instruction: Optional[str] = None

# 标志位
Init: Optional[bool] = False

# Bridge/runtime state
bridge_connected: bool = False
active_client: Optional[str] = None
last_connected_at: Optional[str] = None
last_disconnected_at: Optional[str] = None
last_rgbd_update_at: Optional[str] = None
last_json_update_at: Optional[str] = None
last_observation_update_at: Optional[str] = None
last_command_at: Optional[str] = None
last_command_type: Optional[str] = None
last_command_payload: Optional[Dict[str, Any]] = None
last_error_at: Optional[str] = None
last_error_message: Optional[str] = None
last_json_type: Optional[str] = None
pending_human_prompt: Optional[str] = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def mark_connected(client: Optional[str] = None):
    global bridge_connected, active_client, last_connected_at
    bridge_connected = True
    active_client = client
    last_connected_at = _utc_now_iso()


def mark_disconnected(client: Optional[str] = None):
    global bridge_connected, active_client, last_disconnected_at
    bridge_connected = False
    active_client = None if client is None or client == active_client else active_client
    last_disconnected_at = _utc_now_iso()


def record_observation_update(source: str):
    global last_rgbd_update_at, last_json_update_at, last_observation_update_at
    timestamp = _utc_now_iso()
    last_observation_update_at = timestamp
    if source == "rgbd":
        last_rgbd_update_at = timestamp
    elif source == "json":
        last_json_update_at = timestamp


def record_command(command_type: str, payload: Optional[Dict[str, Any]] = None):
    global last_command_at, last_command_type, last_command_payload
    last_command_at = _utc_now_iso()
    last_command_type = command_type
    last_command_payload = payload


def record_error(message: str):
    global last_error_at, last_error_message
    last_error_at = _utc_now_iso()
    last_error_message = message


def set_pending_human_prompt(prompt: Optional[str]):
    global pending_human_prompt
    pending_human_prompt = prompt


def get_observation_snapshot() -> ObservationSnapshot:
    return ObservationSnapshot(
        rgb_available=rgb_array is not None,
        depth_available=depth_array is not None,
        transform_data=transform_data,
        instruction=instruction,
        last_rgbd_update_at=last_rgbd_update_at,
        last_json_update_at=last_json_update_at,
        last_observation_update_at=last_observation_update_at,
        rgb_shape=None if rgb_array is None else tuple(rgb_array.shape),
        depth_shape=None if depth_array is None else tuple(depth_array.shape),
    )


def get_status_snapshot() -> BridgeStatus:
    return BridgeStatus(
        connected=bridge_connected,
        active_client=active_client,
        last_connected_at=last_connected_at,
        last_disconnected_at=last_disconnected_at,
        last_command_at=last_command_at,
        last_error_at=last_error_at,
        last_error_message=last_error_message,
        last_command_type=last_command_type,
        last_command_payload=last_command_payload,
        last_json_type=last_json_type,
        last_rgbd_update_at=last_rgbd_update_at,
        last_json_update_at=last_json_update_at,
        last_observation_update_at=last_observation_update_at,
        pending_human_prompt=pending_human_prompt,
    )

def clear_shared_state():
    global rgb_array, depth_array, transform_data, instruction
    rgb_array = None
    depth_array = None
    transform_data = None
    instruction = None
