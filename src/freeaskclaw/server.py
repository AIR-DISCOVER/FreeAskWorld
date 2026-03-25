from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from freeaskclaw.config import BridgeConfig
from freeaskclaw.models import (
    AckUpdate,
    Capability,
    CommandResult,
    HealthStatus,
    ImageFrame,
    ObservationState,
    OpenClawAction,
    RuntimeActionResult,
    RuntimeStatus,
    SimulatorCommand,
    TaskUpdate,
)
from freeaskclaw.runtime import RuntimeManager
from freeaskclaw.service import BridgeService, build_transport


def create_app(config: BridgeConfig | None = None) -> FastAPI:
    cfg = config or BridgeConfig.from_env()
    service = BridgeService(build_transport(cfg))
    runtime = RuntimeManager(cfg)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            runtime.close()
            service.transport.close()

    app = FastAPI(title="FreeAskClaw", version="0.1.0", lifespan=lifespan)
    app.state.bridge_service = service
    app.state.runtime_manager = runtime

    @app.get("/healthz", response_model=HealthStatus)
    def healthz() -> HealthStatus:
        return HealthStatus(transport=service.transport.name)

    @app.get("/v1/capabilities", response_model=Capability)
    def capabilities() -> Capability:
        return service.capabilities()

    @app.get("/v1/state")
    def state() -> dict:
        return service.state().model_dump()

    @app.get("/v1/observation", response_model=ObservationState | None)
    def observation() -> ObservationState | None:
        return service.state().last_observation

    @app.get("/v1/observation/frame/{image_kind}", response_model=ImageFrame | None)
    def observation_frame(image_kind: str) -> ImageFrame | None:
        normalized = image_kind.strip().lower()
        if normalized not in {"color", "depth"}:
            raise HTTPException(status_code=404, detail=f"unknown image kind: {image_kind}")
        return service.transport.get_image_frame(normalized)

    @app.get("/v1/runtime/status", response_model=RuntimeStatus)
    def runtime_status() -> RuntimeStatus:
        return runtime.status()

    @app.post("/v1/runtime/start", response_model=RuntimeActionResult)
    def runtime_start() -> RuntimeActionResult:
        return runtime.start()

    @app.post("/v1/runtime/stop", response_model=RuntimeActionResult)
    def runtime_stop() -> RuntimeActionResult:
        return runtime.stop()

    @app.post("/v1/task")
    def publish_task(task: TaskUpdate) -> dict:
        detail = service.publish_task(task)
        return {"accepted": True, "detail": detail, "task": task.model_dump()}

    @app.post("/v1/ack")
    def publish_ack(ack: AckUpdate) -> dict:
        detail = service.publish_ack(ack)
        return {"accepted": True, "detail": detail, "ack": ack.model_dump()}

    @app.post("/v1/commands", response_model=CommandResult)
    def publish_command(command: SimulatorCommand) -> CommandResult:
        try:
            return service.send_command(command)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/openclaw/action", response_model=CommandResult)
    def publish_openclaw_action(action: OpenClawAction) -> CommandResult:
        try:
            return service.send_openclaw_action(action)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
