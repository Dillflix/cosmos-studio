from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

ProgressCallback = Callable[[float, str], None]


@dataclass(slots=True)
class GenerationResult:
    path: Path
    media_type: str
    metadata: dict[str, Any]


class RuntimeAdapter(Protocol):
    family: str

    def generate(
        self,
        job: dict[str, Any],
        output_dir: Path,
        progress: ProgressCallback,
    ) -> GenerationResult: ...

    def unload(self) -> None: ...
