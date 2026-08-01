import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any

from app.config import Settings
from app.db import JobStore
from app.prompting import PromptEnhancer
from app.worker import GenerationWorker


class StuckInference:
    def __init__(self) -> None:
        self.aborted = False

    def submit(self, job: dict[str, Any], output_dir: Path) -> None:
        return None

    def receive(self, timeout: float) -> None:
        raise AssertionError("Cancellation should be checked before waiting for inference")

    def abort(self) -> None:
        self.aborted = True


def test_worker_cancellation_aborts_inference_and_releases_job(tmp_path: Path) -> None:
    settings = replace(
        Settings.from_env(),
        backend="mock",
        data_dir=tmp_path,
        db_path=tmp_path / "jobs.sqlite3",
        output_dir=tmp_path / "outputs",
    )
    settings.output_dir.mkdir()
    store = JobStore(settings.db_path)
    store.initialize()
    store.enqueue(
        {
            "id": "stuck",
            "mode": "cosmos_image",
            "seed": 42,
            "prompt": "cancel this job",
            "params": {"size": "512x512", "enhance_prompt": False},
            "inputs": {},
        }
    )
    running_job = store.claim_next()
    assert running_job is not None
    store.request_cancel("stuck")

    worker = GenerationWorker(settings, store, PromptEnhancer(settings))
    inference = StuckInference()
    worker.inference = inference  # type: ignore[assignment]
    asyncio.run(worker._execute(running_job))

    assert inference.aborted
    assert store.get("stuck")["state"] == "cancelled"
