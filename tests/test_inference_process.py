from dataclasses import replace
from pathlib import Path
from time import monotonic

from app.config import Settings
from app.inference_process import InferenceProcess


def job(job_id: str) -> dict:
    return {
        "id": job_id,
        "mode": "cosmos_image",
        "seed": 42,
        "prompt": "isolated mock generation",
        "params": {"size": "512x512"},
        "inputs": {},
    }


def wait_for_result(process: InferenceProcess, timeout: float = 10) -> dict:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        event = process.receive(0.25)
        if event and event["type"] == "result":
            return event
    raise TimeoutError("Timed out waiting for inference result")


def test_inference_process_can_be_aborted_and_replaced(tmp_path: Path):
    settings = replace(Settings.from_env(), backend="mock", output_dir=tmp_path)
    process = InferenceProcess(settings)
    try:
        process.submit(job("aborted"), tmp_path)
        first_pid = process.pid
        process.abort()
        assert not process.is_alive

        process.submit(job("replacement"), tmp_path)
        assert process.pid != first_pid
        event = wait_for_result(process)
        result = process.result_from_event(event)
        assert result.path == tmp_path / "replacement.png"
        assert result.path.is_file()
    finally:
        process.close()
