import uuid
from typing import Any, Dict, Optional

import numpy as np

try:
    from messages import ActionResult, AgentAction, NavigationCommand, Step
    import shared_state
    import ws_handlers
except ImportError:  # pragma: no cover - package import fallback
    from .messages import ActionResult, AgentAction, NavigationCommand, Step
    from . import shared_state, ws_handlers


class AgentBridge:
    """Compatibility-first bridge into the existing Unity websocket protocol."""

    def get_status(self) -> Dict[str, Any]:
        return shared_state.get_status_snapshot().to_dict()

    def get_observation(self) -> Dict[str, Any]:
        return shared_state.get_observation_snapshot().to_dict()

    def send_navigation_command(
        self,
        local_position_offset,
        local_rotation_offset=None,
        is_stopped: bool = False,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        action = "stop" if is_stopped else "navigation_command"
        command = NavigationCommand(
            LocalPositionOffset=np.array(local_position_offset, dtype=float),
            LocalRotationOffset=np.array(
                local_rotation_offset if local_rotation_offset is not None else [0.0, 0.0, 0.0, 1.0],
                dtype=float,
            ),
            IsStopped=is_stopped,
        )
        sent = ws_handlers.send_navigation_command_via_ws(command)
        detail = "Navigation command sent" if sent else "No active Unity websocket connection"
        return ActionResult(
            ok=sent,
            action=action,
            detail=detail,
            sent=sent,
            connected=shared_state.get_status_snapshot().connected,
            payload=command.to_dict(),
            request_id=request_id,
        ).to_dict()

    def move_forward(self, distance_m: float = 1.0, request_id: Optional[str] = None) -> Dict[str, Any]:
        return self.send_navigation_command(
            local_position_offset=[0.0, 0.0, float(distance_m)],
            local_rotation_offset=[0.0, 0.0, 0.0, 1.0],
            request_id=request_id,
        )

    def turn_left(self, degrees: float = 15.0, request_id: Optional[str] = None) -> Dict[str, Any]:
        return self._send_rotation_action("turn_left", -abs(float(degrees)), request_id=request_id)

    def turn_right(self, degrees: float = 15.0, request_id: Optional[str] = None) -> Dict[str, Any]:
        return self._send_rotation_action("turn_right", abs(float(degrees)), request_id=request_id)

    def stop(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        return self.send_navigation_command(
            local_position_offset=[0.0, 0.0, 0.0],
            local_rotation_offset=[0.0, 0.0, 0.0, 1.0],
            is_stopped=True,
            request_id=request_id,
        )

    def step(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        payload = Step(IsStep=True)
        sent = ws_handlers.send_step_via_ws(payload)
        detail = "Step command sent" if sent else "No active Unity websocket connection"
        return ActionResult(
            ok=sent,
            action="step",
            detail=detail,
            sent=sent,
            connected=shared_state.get_status_snapshot().connected,
            payload=payload.to_dict(),
            request_id=request_id,
        ).to_dict()

    def ask_human(self, prompt: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        shared_state.set_pending_human_prompt(prompt)
        shared_state.record_command("ask_human", {"prompt": prompt})
        return ActionResult(
            ok=True,
            action="ask_human",
            detail="Human prompt recorded locally; simulator-side delivery is not implemented yet",
            sent=False,
            connected=shared_state.get_status_snapshot().connected,
            payload={"prompt": prompt},
            request_id=request_id,
        ).to_dict()

    def perform_action(self, action: AgentAction) -> Dict[str, Any]:
        request_id = action.request_id or str(uuid.uuid4())
        parameters = action.parameters or {}
        action_name = action.action

        if action_name == "move_forward":
            return self.move_forward(distance_m=parameters.get("distance_m", 1.0), request_id=request_id)
        if action_name == "turn_left":
            return self.turn_left(degrees=parameters.get("degrees", 15.0), request_id=request_id)
        if action_name == "turn_right":
            return self.turn_right(degrees=parameters.get("degrees", 15.0), request_id=request_id)
        if action_name == "stop":
            return self.stop(request_id=request_id)
        if action_name == "step":
            return self.step(request_id=request_id)
        if action_name == "ask_human":
            return self.ask_human(prompt=parameters.get("prompt", ""), request_id=request_id)
        if action_name == "navigation_command":
            return self.send_navigation_command(
                local_position_offset=parameters.get("local_position_offset", [0.0, 0.0, 0.0]),
                local_rotation_offset=parameters.get("local_rotation_offset", [0.0, 0.0, 0.0, 1.0]),
                is_stopped=parameters.get("is_stopped", False),
                request_id=request_id,
            )

        shared_state.record_error(f"Unsupported agent action: {action_name}")
        return ActionResult(
            ok=False,
            action=action_name,
            detail=f"Unsupported action: {action_name}",
            sent=False,
            connected=shared_state.get_status_snapshot().connected,
            payload=parameters,
            request_id=request_id,
        ).to_dict()

    def _send_rotation_action(self, action: str, yaw_degrees: float, request_id: Optional[str] = None) -> Dict[str, Any]:
        command = NavigationCommand(
            LocalPositionOffset=np.array([0.0, 0.0, 0.0], dtype=float),
            LocalRotationOffset=np.array([0.0, float(yaw_degrees), 0.0, 1.0], dtype=float),
            IsStopped=False,
        )
        sent = ws_handlers.send_navigation_command_via_ws(command)
        detail = (
            "Rotation command sent using LocalRotationOffset compatibility mapping"
            if sent
            else "No active Unity websocket connection"
        )
        return ActionResult(
            ok=sent,
            action=action,
            detail=detail,
            sent=sent,
            connected=shared_state.get_status_snapshot().connected,
            payload=command.to_dict(),
            request_id=request_id,
        ).to_dict()


bridge = AgentBridge()
