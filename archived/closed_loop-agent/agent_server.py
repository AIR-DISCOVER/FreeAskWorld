from typing import Any, Dict, Optional

from pydantic import BaseModel

try:
    from .messages import AgentAction
    from .agent_bridge import bridge
except ImportError:  # pragma: no cover - script import fallback
    from messages import AgentAction
    from agent_bridge import bridge


class ActionRequest(BaseModel):
    action: str
    parameters: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None


def create_app():
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "fastapi is required to run the agent HTTP bridge. Install closed_loop/requirements.txt first."
        ) from exc

    app = FastAPI(title="FreeAskWorld Agent Bridge", version="0.1.0")

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "service": "freeaskworld-agent-bridge"}

    @app.get("/v1/status")
    def get_status():
        return bridge.get_status()

    @app.get("/v1/observation")
    def get_observation():
        return bridge.get_observation()

    @app.post("/v1/action")
    def post_action(action_request: ActionRequest):
        result = bridge.perform_action(
            AgentAction(
                action=action_request.action,
                parameters=action_request.parameters or {},
                request_id=action_request.request_id,
            )
        )
        if not result["ok"] and result["action"] not in {"ask_human", "turn_left", "turn_right"}:
            raise HTTPException(status_code=409, detail=result)
        return result

    @app.post("/v1/navigation-command")
    def post_navigation_command(payload: Dict[str, Any]):
        result = bridge.send_navigation_command(
            local_position_offset=payload.get("local_position_offset", [0.0, 0.0, 0.0]),
            local_rotation_offset=payload.get("local_rotation_offset", [0.0, 0.0, 0.0, 1.0]),
            is_stopped=payload.get("is_stopped", False),
            request_id=payload.get("request_id"),
        )
        if not result["ok"]:
            raise HTTPException(status_code=409, detail=result)
        return result

    @app.post("/v1/step")
    def post_step(payload: Optional[Dict[str, Any]] = None):
        payload = payload or {}
        result = bridge.step(request_id=payload.get("request_id"))
        if not result["ok"]:
            raise HTTPException(status_code=409, detail=result)
        return result

    return app

try:  # pragma: no cover - depends on optional fastapi install
    app = create_app()
except RuntimeError:
    app = None
