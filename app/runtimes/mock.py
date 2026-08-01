from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

from .base import GenerationResult, ProgressCallback

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zl1sAAAAASUVORK5CYII="
)


class MockRuntime:
    family = "mock"

    def generate(
        self,
        job: dict[str, Any],
        output_dir: Path,
        progress: ProgressCallback,
    ) -> GenerationResult:
        progress(0.2, "Mock runtime preparing output")
        time.sleep(0.03)
        if job["mode"] == "cosmos_image":
            path = output_dir / f"{job['id']}.png"
            path.write_bytes(PNG_1X1)
            media_type = "image/png"
        else:
            path = output_dir / f"{job['id']}.mock.json"
            path.write_text(
                json.dumps(
                    {
                        "mock": True,
                        "mode": job["mode"],
                        "seed": job["seed"],
                        "prompt": job["prompt"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            media_type = "application/json"
        progress(0.95, "Mock output written")
        return GenerationResult(
            path=path,
            media_type=media_type,
            metadata={"backend": "mock", "validated_inference": False},
        )

    def unload(self) -> None:
        return None
