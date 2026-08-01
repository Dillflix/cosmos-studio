from __future__ import annotations

import asyncio
import json
import logging
import traceback
from typing import Any

from .config import Settings
from .db import JobStore
from .prompting import PromptEnhancer, prompt_for_runtime
from .runtimes import RuntimeManager

logger = logging.getLogger("cosmos-studio.worker")


class GenerationWorker:
    def __init__(
        self,
        settings: Settings,
        store: JobStore,
        enhancer: PromptEnhancer,
    ) -> None:
        self.settings = settings
        self.store = store
        self.enhancer = enhancer
        self.runtimes = RuntimeManager(settings)
        self.wakeup = asyncio.Event()
        self.stopping = asyncio.Event()
        self.task: asyncio.Task[None] | None = None
        self.current_job_id: str | None = None
        self.last_error: str | None = None

    def start(self) -> None:
        self.task = asyncio.create_task(self.run(), name="generation-worker")
        self.wakeup.set()

    def notify(self) -> None:
        self.wakeup.set()

    async def stop(self) -> None:
        self.stopping.set()
        self.wakeup.set()
        if self.task:
            await self.task
        await asyncio.to_thread(self.runtimes.unload)

    async def run(self) -> None:
        while not self.stopping.is_set():
            job = await asyncio.to_thread(self.store.claim_next)
            if job is None:
                self.wakeup.clear()
                try:
                    await asyncio.wait_for(
                        self.wakeup.wait(), timeout=self.settings.worker_poll_seconds
                    )
                except TimeoutError:
                    pass
                continue
            self.current_job_id = job["id"]
            try:
                await self._execute(job)
                self.last_error = None
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                logger.error("Job %s failed:\n%s", job["id"], traceback.format_exc())
                await asyncio.to_thread(self.store.fail, job["id"], self.last_error)
            finally:
                self.current_job_id = None

    async def _execute(self, job: dict[str, Any]) -> None:
        params = job["params"]
        structured = params.get("structured_prompt")
        prompt_mode = "plain"
        if structured is not None:
            params["prepared_prompt"] = prompt_for_runtime(
                structured, "cosmos" if job["mode"].startswith("cosmos_") else "vace"
            )
            prompt_mode = "provided_structured"
        elif params.get("enhance_prompt", False):
            await asyncio.to_thread(
                self.store.progress, job["id"], 0.02, "Enhancing prompt"
            )
            try:
                enhancement_image = None
                if job["inputs"].get("image"):
                    enhancement_image = job["inputs"]["image"]["path"]
                elif job["inputs"].get("reference_images"):
                    enhancement_image = job["inputs"]["reference_images"][0]["path"]
                structured = await self.enhancer.enhance(
                    job["prompt"],
                    size=params.get("size", "1280x720"),
                    mode=job["mode"],
                    image_path=enhancement_image,
                )
                family = "cosmos" if job["mode"].startswith("cosmos_") else "vace"
                params["prepared_prompt"] = prompt_for_runtime(structured, family)
                prompt_mode = "enhanced"
            except Exception:
                if not params.get("enhancement_fallback_to_plain", True):
                    raise
                logger.warning(
                    "Prompt enhancement failed for %s; using plain prompt",
                    job["id"],
                    exc_info=True,
                )
                params["prepared_prompt"] = job["prompt"]
                prompt_mode = "plain_fallback"
        else:
            params["prepared_prompt"] = job["prompt"]
        negative = params.get("negative_prompt")
        if negative is not None and not isinstance(negative, str):
            params["negative_prompt"] = json.dumps(negative, ensure_ascii=False)

        def update_progress(value: float, message: str) -> None:
            self.store.progress(job["id"], value, message)

        result = await asyncio.to_thread(
            self.runtimes.generate,
            job,
            self.settings.output_dir,
            update_progress,
        )
        metadata = {
            **result.metadata,
            "job_id": job["id"],
            "mode": job["mode"],
            "seed": job["seed"],
            "prompt": job["prompt"],
            "prompt_mode": prompt_mode,
            "structured_prompt": structured,
            "parameters": {
                key: value
                for key, value in params.items()
                if key not in {"prepared_prompt", "structured_prompt"}
            },
            "inputs": {
                key: (
                    [
                        {k: v for k, v in item.items() if k != "path"}
                        for item in value
                    ]
                    if isinstance(value, list)
                    else {k: v for k, v in value.items() if k != "path"}
                )
                for key, value in job["inputs"].items()
            },
        }
        sidecar = result.path.with_suffix(result.path.suffix + ".json")
        sidecar.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if await asyncio.to_thread(self.store.mark_cancelled_after_run, job["id"]):
            return
        await asyncio.to_thread(
            self.store.complete,
            job["id"],
            output_path=str(result.path),
            output_url=f"/media/{result.path.name}",
            media_type=result.media_type,
            metadata=metadata,
        )
