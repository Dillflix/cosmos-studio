from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import Settings
from .base import GenerationResult, ProgressCallback, RuntimeAdapter
from .cosmos import CosmosRuntime
from .mock import MockRuntime
from .vace import VaceRuntime


class RuntimeManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.current: RuntimeAdapter | None = None
        self.current_family: str | None = None

    def generate(
        self,
        job: dict[str, Any],
        output_dir: Path,
        progress: ProgressCallback,
    ) -> GenerationResult:
        requested_family = (
            "mock"
            if self.settings.backend == "mock"
            else ("cosmos" if job["mode"].startswith("cosmos_") else "vace")
        )
        if self.current is None or self.current_family != requested_family:
            self.unload()
            if requested_family == "mock":
                self.current = MockRuntime()
            elif requested_family == "cosmos":
                self.current = CosmosRuntime(self.settings)
            else:
                self.current = VaceRuntime(self.settings)
            self.current_family = requested_family
        return self.current.generate(job, output_dir, progress)

    def unload(self) -> None:
        if self.current is not None:
            self.current.unload()
        self.current = None
        self.current_family = None
