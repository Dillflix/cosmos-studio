from pathlib import Path

from app.db import JobStore


def record(job_id: str, seed: int = 42) -> dict:
    return {
        "id": job_id,
        "mode": "cosmos_image",
        "seed": seed,
        "prompt": "test prompt",
        "params": {"size": "512x512"},
        "inputs": {},
    }


def test_queue_is_fifo_and_seed_is_persisted(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.initialize()
    store.enqueue(record("first", 101))
    store.enqueue(record("second", 202))
    assert store.queue_position("first") == 1
    assert store.queue_position("second") == 2
    claimed = store.claim_next()
    assert claimed is not None
    assert claimed["id"] == "first"
    assert claimed["seed"] == 101


def test_running_jobs_recover_without_changing_seed(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.initialize()
    store.enqueue(record("job", 999))
    assert store.claim_next()["state"] == "running"
    assert store.recover_interrupted() == 1
    recovered = store.get("job")
    assert recovered["state"] == "queued"
    assert recovered["seed"] == 999


def test_queued_job_can_be_cancelled(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.initialize()
    store.enqueue(record("job"))
    assert store.request_cancel("job") == "cancelled"
    assert store.get("job")["state"] == "cancelled"
