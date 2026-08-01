from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .config import Settings


def parse_size(size: str) -> tuple[int, int]:
    try:
        width_text, height_text = size.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except (AttributeError, ValueError) as exc:
        raise ValueError("size must use WIDTHxHEIGHT") from exc
    if not (256 <= width <= 2048 and 256 <= height <= 2048):
        raise ValueError("width and height must be between 256 and 2048")
    if width % 16 or height % 16:
        raise ValueError("width and height must be divisible by 16")
    return width, height


def _locate_output(target: Path) -> Path:
    candidates = [
        target,
        target / "prompt_0.json",
        target.with_suffix(".json"),
        target.with_suffix(".json") / "prompt_0.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(target.parent.glob("**/prompt_0.json"))
    if matches:
        return matches[0]
    raise RuntimeError("Prompt upsampler produced no JSON output")


class PromptEnhancer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.lock = asyncio.Lock()

    @property
    def configured(self) -> bool:
        return bool(self.settings.prompt_endpoint and self.settings.prompt_model)

    async def enhance(
        self,
        prompt: str,
        *,
        size: str,
        mode: str,
        use_cache: bool = True,
        image_path: str | None = None,
    ) -> Any:
        width, height = parse_size(size)
        key = hashlib.sha256(
            f"{self.settings.prompt_model}\0{mode}\0{size}\0{image_path or ''}\0{prompt}".encode()
        ).hexdigest()
        if use_cache and key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        async with self.lock:
            if use_cache and key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            value = await asyncio.to_thread(
                self._enhance_sync, prompt, width, height, mode, image_path
            )
            if use_cache:
                self.cache[key] = value
                self.cache.move_to_end(key)
                while len(self.cache) > self.settings.prompt_cache_max:
                    self.cache.popitem(last=False)
            return value

    def _enhance_sync(
        self,
        prompt: str,
        width: int,
        height: int,
        mode: str,
        image_path: str | None,
    ) -> Any:
        if not self.configured:
            raise RuntimeError(
                "Prompt enhancement is not configured; set "
                "PROMPT_UPSAMPLER_ENDPOINT_URL and PROMPT_UPSAMPLER_MODEL_NAME"
            )
        tier = min((256, 480, 720), key=lambda value: abs(value - min(width, height)))
        from math import gcd

        divisor = gcd(width, height)
        ratio = f"{width // divisor},{height // divisor}"
        supported_mode = {
            "cosmos_image": "text2image",
            "cosmos_text_video": "text2video",
            "cosmos_image_video": "image2video",
            "cosmos_video_continue": "text2video",
            "vace_text_video": "text2video",
            "vace_reference_video": "image2video",
            "vace_control_edit": "text2video",
        }.get(mode, mode)
        if supported_mode == "image2video" and not image_path:
            supported_mode = "text2video"
        with tempfile.TemporaryDirectory(
            prefix="cosmos-studio-prompt-", dir=self.settings.prompt_tmp_dir
        ) as temp_directory:
            temp = Path(temp_directory)
            input_path = temp / "prompt.txt"
            output_target = temp / "upsampled"
            input_path.write_text(prompt.strip() + "\n", encoding="utf-8")
            command = [
                sys.executable,
                "-m",
                "cosmos_framework.inference.prompt_upsampling",
                "--input",
                str(input_path),
                "--output",
                str(output_target),
                "--mode",
                supported_mode,
                "--endpoint-url",
                self.settings.prompt_endpoint,
                "--model",
                self.settings.prompt_model,
                "--api-token",
                self.settings.prompt_token,
                "--resolution",
                str(tier),
                "--aspect-ratio",
                ratio,
            ]
            if supported_mode == "image2video" and image_path:
                command.extend(["--image-url", image_path])
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.settings.prompt_timeout_seconds,
            )
            if completed.returncode:
                raise RuntimeError(
                    f"Prompt upsampler exited {completed.returncode}: "
                    f"{completed.stderr.strip()}"
                )
            return json.loads(_locate_output(output_target).read_text(encoding="utf-8"))


def prompt_for_runtime(structured: Any, family: str) -> str:
    if family == "cosmos":
        return json.dumps(structured, ensure_ascii=False)
    candidate = structured
    if isinstance(candidate, list) and len(candidate) == 1:
        candidate = candidate[0]
    if isinstance(candidate, dict):
        for key in (
            "comprehensive_t2v_caption",
            "comprehensive_t2i_caption",
            "caption",
            "prompt",
        ):
            value = candidate.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return json.dumps(candidate, ensure_ascii=False)
