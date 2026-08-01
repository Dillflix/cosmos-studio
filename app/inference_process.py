from __future__ import annotations

import multiprocessing
import traceback
from pathlib import Path
from queue import Empty
from typing import Any

from .config import Settings
from .runtimes.base import GenerationResult


class InferenceProcessError(RuntimeError):
    pass


def _inference_main(
    settings: Settings,
    commands: Any,
    events: Any,
) -> None:
    # Runtime imports stay in the child process. The FastAPI process never owns
    # a ROCm context, so a wedged kernel can be killed without taking down HTTP.
    from .runtimes.manager import RuntimeManager

    manager = RuntimeManager(settings)
    try:
        while True:
            command = commands.get()
            if command.get("type") == "shutdown":
                return
            if command.get("type") != "generate":
                continue
            job = command["job"]

            def progress(value: float, message: str) -> None:
                events.put(
                    {
                        "type": "progress",
                        "job_id": job["id"],
                        "value": value,
                        "message": message,
                    }
                )

            try:
                result = manager.generate(
                    job,
                    Path(command["output_dir"]),
                    progress,
                )
                events.put(
                    {
                        "type": "result",
                        "job_id": job["id"],
                        "path": str(result.path),
                        "media_type": result.media_type,
                        "metadata": result.metadata,
                    }
                )
            except Exception as exc:
                events.put(
                    {
                        "type": "error",
                        "job_id": job["id"],
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
                manager.unload()
    finally:
        manager.unload()


class InferenceProcess:
    """Persistent, replaceable process that exclusively owns model runtimes."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.context = multiprocessing.get_context("spawn")
        self.process: Any | None = None
        self.commands: Any | None = None
        self.events: Any | None = None

    @property
    def is_alive(self) -> bool:
        try:
            return bool(self.process and self.process.is_alive())
        except ValueError:
            return False

    @property
    def pid(self) -> int | None:
        try:
            return self.process.pid if self.process else None
        except ValueError:
            return None

    def start(self) -> None:
        if self.is_alive:
            return
        self._discard_handles()
        self.commands = self.context.Queue()
        self.events = self.context.Queue()
        self.process = self.context.Process(
            target=_inference_main,
            name="cosmos-inference",
            args=(self.settings, self.commands, self.events),
        )
        self.process.start()

    def submit(self, job: dict[str, Any], output_dir: Path) -> None:
        self.start()
        assert self.commands is not None
        self.commands.put(
            {
                "type": "generate",
                "job": job,
                "output_dir": str(output_dir),
            }
        )

    def receive(self, timeout: float = 0.25) -> dict[str, Any] | None:
        if self.events is None:
            raise InferenceProcessError("Inference process has not started")
        try:
            return self.events.get(timeout=timeout)
        except Empty:
            if not self.is_alive:
                exit_code = self.process.exitcode if self.process else None
                self._discard_handles()
                raise InferenceProcessError(
                    f"Inference process exited unexpectedly ({exit_code=})"
                )
            return None

    def abort(self) -> None:
        """Force-stop a running native pipeline and discard its ROCm context."""
        process = self.process
        if process and process.is_alive():
            process.terminate()
            process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join(timeout=3)
        self._discard_handles()

    def close(self) -> None:
        process = self.process
        if process and process.is_alive() and self.commands is not None:
            self.commands.put({"type": "shutdown"})
            process.join(timeout=10)
        if process and process.is_alive():
            self.abort()
        else:
            self._discard_handles()

    def _discard_handles(self) -> None:
        for queue in (self.commands, self.events):
            if queue is not None:
                queue.cancel_join_thread()
                queue.close()
        if self.process is not None:
            self.process.close()
        self.process = None
        self.commands = None
        self.events = None

    @staticmethod
    def result_from_event(event: dict[str, Any]) -> GenerationResult:
        return GenerationResult(
            path=Path(event["path"]),
            media_type=event["media_type"],
            metadata=event["metadata"],
        )
