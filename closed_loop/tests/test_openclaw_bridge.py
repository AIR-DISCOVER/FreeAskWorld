import os
import sys

import numpy as np

TESTS_DIR = os.path.dirname(__file__)
CLOSED_LOOP_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if CLOSED_LOOP_DIR not in sys.path:
    sys.path.insert(0, CLOSED_LOOP_DIR)

import shared_state
from messages import OpenClawAction, TransformData
from openclaw_bridge import OpenClawBridge


def reset_runtime_state():
    shared_state.clear_shared_state()
    shared_state.Init = False
    shared_state.bridge_connected = False
    shared_state.active_client = None
    shared_state.last_connected_at = None
    shared_state.last_disconnected_at = None
    shared_state.last_rgbd_update_at = None
    shared_state.last_json_update_at = None
    shared_state.last_observation_update_at = None
    shared_state.last_command_at = None
    shared_state.last_command_type = None
    shared_state.last_command_payload = None
    shared_state.last_error_at = None
    shared_state.last_error_message = None
    shared_state.last_json_type = None
    shared_state.pending_human_prompt = None


def test_disconnected_status_snapshot():
    reset_runtime_state()
    bridge = OpenClawBridge()

    status = bridge.get_status()

    assert status["connected"] is False
    assert status["active_client"] is None
    assert status["last_command_type"] is None


def test_action_translation_move_forward(monkeypatch):
    reset_runtime_state()
    bridge = OpenClawBridge()
    captured = {}

    def fake_send_navigation_command_via_ws(command, websocket=None, timeout=5.0):
        captured["payload"] = command.to_dict()
        return True

    monkeypatch.setattr("openclaw_bridge.ws_handlers.send_navigation_command_via_ws", fake_send_navigation_command_via_ws)
    shared_state.mark_connected("127.0.0.1:8765")

    result = bridge.perform_action(OpenClawAction(action="move_forward", parameters={"distance_m": 2.5}))

    assert result["ok"] is True
    assert result["action"] == "navigation_command"
    assert captured["payload"]["LocalPositionOffset"] == [0.0, 0.0, 2.5]
    assert captured["payload"]["IsStopped"] is False


def test_move_forward_and_stop_result_structure(monkeypatch):
    reset_runtime_state()
    bridge = OpenClawBridge()

    def fake_send_navigation_command_via_ws(command, websocket=None, timeout=5.0):
        return True

    monkeypatch.setattr("openclaw_bridge.ws_handlers.send_navigation_command_via_ws", fake_send_navigation_command_via_ws)
    shared_state.mark_connected("simulator")

    move_result = bridge.move_forward(distance_m=1.0)
    stop_result = bridge.stop()

    assert move_result["ok"] is True
    assert move_result["payload"]["LocalPositionOffset"] == [0.0, 0.0, 1.0]
    assert stop_result["ok"] is True
    assert stop_result["payload"]["IsStopped"] is True


def test_observation_snapshot_shape():
    reset_runtime_state()
    bridge = OpenClawBridge()
    shared_state.rgb_array = np.zeros((4, 5, 3), dtype=np.uint8)
    shared_state.depth_array = np.zeros((4, 5), dtype=np.uint8)
    shared_state.transform_data = TransformData(position=(1.0, 2.0, 3.0), rotation=(0.0, 0.0, 0.0, 1.0))
    shared_state.instruction = "go forward"
    shared_state.record_observation_update("rgbd")
    shared_state.record_observation_update("json")

    observation = bridge.get_observation()

    assert observation["rgb_available"] is True
    assert observation["depth_available"] is True
    assert observation["rgb_shape"] == [4, 5, 3]
    assert observation["depth_shape"] == [4, 5]
    assert observation["transform_data"]["position"] == [1.0, 2.0, 3.0]
    assert observation["instruction"] == "go forward"
