from __future__ import annotations

import shlex
import subprocess
import threading
from dataclasses import dataclass

from freeaskclaw.config import BridgeConfig
from freeaskclaw.models import RuntimeActionResult, RuntimeProcessStatus, RuntimeStatus


@dataclass(frozen=True, slots=True)
class ManagedProcessSpec:
    name: str
    start_command: str
    cwd: str
    stop_command: str | None = None


class RuntimeManager:
    def __init__(self, config: BridgeConfig) -> None:
        workspace = config.runtime_workspace
        self._specs = {
            "ros2_backend": ManagedProcessSpec(
                name="ros2_backend",
                start_command=config.runtime_ros2_start_command,
                stop_command=config.runtime_ros2_stop_command,
                cwd=workspace,
            ),
        }
        self._processes: dict[str, subprocess.Popen[bytes] | None] = {
            name: None for name in self._specs
        }
        self._started_by_manager: set[str] = set()
        self._lock = threading.Lock()

    def status(self) -> RuntimeStatus:
        with self._lock:
            return RuntimeStatus(processes=[self._status_for(name) for name in self._specs])

    def start(self) -> RuntimeActionResult:
        details: list[str] = []
        with self._lock:
            for name, spec in self._specs.items():
                proc = self._refresh_process(name)
                if proc is not None:
                    details.append(f"{name} already running (pid={proc.pid})")
                    continue

                process = subprocess.Popen(
                    shlex.split(spec.start_command),
                    cwd=spec.cwd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                self._processes[name] = process
                self._started_by_manager.add(name)
                details.append(f"started {name} (pid={process.pid})")

            status = RuntimeStatus(processes=[self._status_for(name) for name in self._specs])
        return RuntimeActionResult(detail="; ".join(details), status=status)

    def stop(self) -> RuntimeActionResult:
        details: list[str] = []
        with self._lock:
            for name, spec in self._specs.items():
                proc = self._refresh_process(name)
                managed = name in self._started_by_manager
                if managed and spec.stop_command:
                    completed = subprocess.run(
                        shlex.split(spec.stop_command),
                        cwd=spec.cwd,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        check=False,
                        text=True,
                    )
                    output = completed.stdout.strip()
                    if output:
                        details.append(f"{name} stop command: {output}")
                if proc is None:
                    self._processes[name] = None
                    self._started_by_manager.discard(name)
                    details.append(f"{name} not running")
                    continue
                self._terminate_process(proc)
                self._processes[name] = None
                self._started_by_manager.discard(name)
                details.append(f"stopped {name}")

            status = RuntimeStatus(processes=[self._status_for(name) for name in self._specs])
        return RuntimeActionResult(detail="; ".join(details), status=status)

    def close(self) -> None:
        self.stop()

    def _status_for(self, name: str) -> RuntimeProcessStatus:
        spec = self._specs[name]
        proc = self._refresh_process(name)
        return RuntimeProcessStatus(
            name=name,
            command=spec.start_command,
            cwd=spec.cwd,
            running=proc is not None,
            pid=None if proc is None else proc.pid,
        )

    def _refresh_process(self, name: str) -> subprocess.Popen[bytes] | None:
        proc = self._processes[name]
        if proc is not None and proc.poll() is not None:
            self._processes[name] = None
            return None
        return proc

    def _terminate_process(self, proc: subprocess.Popen[bytes]) -> None:
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)
