from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .models import JobState

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    state TEXT NOT NULL,
    seed INTEGER NOT NULL,
    prompt TEXT NOT NULL,
    negative_prompt TEXT,
    params_json TEXT NOT NULL,
    inputs_json TEXT NOT NULL,
    output_path TEXT,
    output_url TEXT,
    media_type TEXT,
    metadata_json TEXT,
    error TEXT,
    progress REAL NOT NULL DEFAULT 0,
    progress_message TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    started_at REAL,
    completed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_state_created
ON jobs(state, created_at);
"""


class JobStore:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    def recover_interrupted(self) -> int:
        with self.connect() as connection:
            cancelled = connection.execute(
                """
                UPDATE jobs
                SET state = ?, completed_at = ?,
                    progress_message = 'Cancelled during service restart'
                WHERE state = ? AND cancel_requested = 1
                """,
                (JobState.CANCELLED, time.time(), JobState.RUNNING),
            ).rowcount
            cursor = connection.execute(
                """
                UPDATE jobs
                SET state = ?, started_at = NULL, progress = 0,
                    progress_message = 'Recovered after service restart',
                    error = NULL
                WHERE state = ? AND cancel_requested = 0
                """,
                (JobState.QUEUED, JobState.RUNNING),
            )
            return cursor.rowcount + cancelled

    def enqueue(self, record: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    id, mode, state, seed, prompt, negative_prompt,
                    params_json, inputs_json, created_at,
                    progress_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"],
                    record["mode"],
                    JobState.QUEUED,
                    record["seed"],
                    record["prompt"],
                    record.get("negative_prompt"),
                    json.dumps(record.get("params", {}), ensure_ascii=False),
                    json.dumps(record.get("inputs", {}), ensure_ascii=False),
                    record.get("created_at", time.time()),
                    "Waiting for worker",
                ),
            )

    def claim_next(self) -> dict[str, Any] | None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT rowid AS _rowid, * FROM jobs
                WHERE state = ? AND cancel_requested = 0
                ORDER BY created_at ASC LIMIT 1
                """,
                (JobState.QUEUED,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE jobs SET state = ?, started_at = ?, progress = 0.01,
                    progress_message = 'Claimed by worker'
                WHERE id = ? AND state = ?
                """,
                (JobState.RUNNING, time.time(), row["id"], JobState.QUEUED),
            )
            return self._decode(dict(row) | {"state": JobState.RUNNING})

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            return self._decode(dict(row)) if row else None

    def list(self, *, limit: int = 50, completed_only: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM jobs"
        values: list[Any] = []
        if completed_only:
            query += " WHERE state = ?"
            values.append(JobState.COMPLETED)
        query += " ORDER BY created_at DESC LIMIT ?"
        values.append(limit)
        with self.connect() as connection:
            rows = connection.execute(query, values).fetchall()
            return [self._decode(dict(row)) for row in rows]

    def queue_position(self, job_id: str) -> int | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT rowid AS _rowid, state, created_at FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if row is None or row["state"] != JobState.QUEUED:
                return None
            count = connection.execute(
                """
                SELECT COUNT(*) FROM jobs
                WHERE state = ? AND (
                    created_at < ? OR (created_at = ? AND rowid <= ?)
                )
                """,
                (
                    JobState.QUEUED,
                    row["created_at"],
                    row["created_at"],
                    row["_rowid"],
                ),
            ).fetchone()[0]
            return int(count)

    def counts(self) -> dict[str, int]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT state, COUNT(*) AS count FROM jobs GROUP BY state"
            ).fetchall()
        counts = {state.value: 0 for state in JobState}
        counts.update({row["state"]: row["count"] for row in rows})
        return counts

    def progress(self, job_id: str, value: float, message: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE jobs SET progress = ?, progress_message = ?
                WHERE id = ? AND state = ?
                """,
                (max(0.0, min(value, 0.99)), message, job_id, JobState.RUNNING),
            )

    def complete(
        self,
        job_id: str,
        *,
        output_path: str,
        output_url: str,
        media_type: str,
        metadata: dict[str, Any],
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE jobs SET state = ?, output_path = ?, output_url = ?,
                    media_type = ?, metadata_json = ?, progress = 1,
                    progress_message = 'Completed', completed_at = ?
                WHERE id = ?
                """,
                (
                    JobState.COMPLETED,
                    output_path,
                    output_url,
                    media_type,
                    json.dumps(metadata, ensure_ascii=False),
                    time.time(),
                    job_id,
                ),
            )

    def fail(self, job_id: str, error: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE jobs SET state = ?, error = ?, progress_message = ?,
                    completed_at = ? WHERE id = ?
                """,
                (JobState.FAILED, error, "Generation failed", time.time(), job_id),
            )

    def request_cancel(self, job_id: str) -> str | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT state FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return None
            if row["state"] == JobState.QUEUED:
                connection.execute(
                    """
                    UPDATE jobs SET state = ?, cancel_requested = 1,
                        progress_message = 'Cancelled', completed_at = ?
                    WHERE id = ?
                    """,
                    (JobState.CANCELLED, time.time(), job_id),
                )
                return JobState.CANCELLED
            if row["state"] == JobState.RUNNING:
                connection.execute(
                    """
                    UPDATE jobs SET cancel_requested = 1,
                        progress_message = 'Cancellation requested'
                    WHERE id = ?
                    """,
                    (job_id,),
                )
            return row["state"]

    def mark_cancelled_after_run(self, job_id: str) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if not row or not row["cancel_requested"]:
                return False
            connection.execute(
                """
                UPDATE jobs SET state = ?, progress_message = 'Cancelled',
                    completed_at = ? WHERE id = ?
                """,
                (JobState.CANCELLED, time.time(), job_id),
            )
            return True

    @staticmethod
    def _decode(row: dict[str, Any]) -> dict[str, Any]:
        row.pop("_rowid", None)
        row["params"] = json.loads(row.pop("params_json") or "{}")
        row["inputs"] = json.loads(row.pop("inputs_json") or "{}")
        metadata_json = row.pop("metadata_json", None)
        row["metadata"] = json.loads(metadata_json) if metadata_json else None
        row["cancel_requested"] = bool(row.get("cancel_requested"))
        return row
