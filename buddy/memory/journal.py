"""Durable checkpoints for the cross-database memory workflow."""

import json
import sqlite3
from dataclasses import dataclass
from enum import IntEnum


class MemoryProcessingStage(IntEnum):
    """Ordered checkpoints completed for one conversation."""

    STARTED = 0
    SESSION_PERSISTED = 1
    GLOBAL_APPLIED = 2
    GAME_PROFILES_APPLIED = 3


@dataclass(frozen=True, slots=True)
class MemoryProcessingJob:
    """Reloadable progress and analyzer result for one conversation."""

    conversation_id: int
    stage: MemoryProcessingStage
    analysis: dict[str, object] | None
    last_error: str | None


def init_memory_journal(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_processing_jobs (
            conversation_id INTEGER PRIMARY KEY,
            stage INTEGER NOT NULL DEFAULT 0 CHECK (stage BETWEEN 0 AND 3),
            analysis_json TEXT,
            last_error TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id)
                REFERENCES conversations (id) ON DELETE CASCADE
        )
        """
    )


def get_or_create_memory_job(
    conn: sqlite3.Connection,
    conversation_id: int,
) -> MemoryProcessingJob:
    with conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO memory_processing_jobs (conversation_id)
            VALUES (?)
            """,
            (conversation_id,),
        )
    return _load_job(conn, conversation_id)


def checkpoint_memory_job(
    conn: sqlite3.Connection,
    conversation_id: int,
    stage: MemoryProcessingStage,
    *,
    analysis: dict[str, object] | None = None,
) -> MemoryProcessingJob:
    current = _load_job(conn, conversation_id)
    if stage < current.stage:
        raise ValueError("A memory processing checkpoint cannot move backwards.")
    serialized = (
        json.dumps(analysis, ensure_ascii=False, sort_keys=True)
        if analysis is not None
        else None
    )
    with conn:
        conn.execute(
            """
            UPDATE memory_processing_jobs
            SET stage = ?,
                analysis_json = COALESCE(?, analysis_json),
                last_error = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE conversation_id = ?
            """,
            (int(stage), serialized, conversation_id),
        )
    return _load_job(conn, conversation_id)


def record_memory_job_failure(
    conn: sqlite3.Connection,
    conversation_id: int,
    error: str,
) -> None:
    with conn:
        conn.execute(
            """
            UPDATE memory_processing_jobs
            SET last_error = ?, updated_at = CURRENT_TIMESTAMP
            WHERE conversation_id = ?
            """,
            (error, conversation_id),
        )


def delete_memory_job(conn: sqlite3.Connection, conversation_id: int) -> None:
    with conn:
        conn.execute(
            "DELETE FROM memory_processing_jobs WHERE conversation_id = ?",
            (conversation_id,),
        )


def _load_job(
    conn: sqlite3.Connection,
    conversation_id: int,
) -> MemoryProcessingJob:
    row = conn.execute(
        "SELECT * FROM memory_processing_jobs WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError("Memory processing job could not be loaded.")
    try:
        analysis = json.loads(row["analysis_json"]) if row["analysis_json"] else None
    except json.JSONDecodeError as exc:
        raise RuntimeError("Memory processing journal contains invalid JSON.") from exc
    if analysis is not None and not isinstance(analysis, dict):
        raise RuntimeError("Memory processing journal analysis must be an object.")
    return MemoryProcessingJob(
        conversation_id=row["conversation_id"],
        stage=MemoryProcessingStage(row["stage"]),
        analysis=analysis,
        last_error=row["last_error"],
    )
