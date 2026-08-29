"""SQLite metadata and report storage for locally recorded video segments."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Mapping

from buddy.video_models import (
    VideoReport,
    VideoReportMode,
    VideoSegment,
    VideoSegmentStatus,
)


def init_video_storage(conn: sqlite3.Connection) -> None:
    """Create video segment and analysis report tables."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS video_segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            file_path TEXT NOT NULL UNIQUE,
            started_at TEXT NOT NULL,
            ended_at TEXT NOT NULL,
            duration_seconds REAL NOT NULL CHECK (duration_seconds > 0),
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id)
                REFERENCES conversations (id) ON DELETE SET NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS video_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            segment_id INTEGER NOT NULL,
            mode TEXT NOT NULL CHECK (mode IN ('overview', 'detail')),
            question TEXT,
            provider TEXT NOT NULL,
            model_name TEXT NOT NULL,
            report_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (segment_id)
                REFERENCES video_segments (id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_video_segments_status_created
        ON video_segments (status, created_at)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_video_segments_conversation
        ON video_segments (conversation_id, started_at)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_video_reports_segment_mode
        ON video_reports (segment_id, mode, created_at)
        """
    )


def create_video_segment(
    conn: sqlite3.Connection,
    *,
    file_path: str | Path,
    started_at: datetime,
    ended_at: datetime,
    conversation_id: int | None = None,
) -> VideoSegment:
    """Persist one completed local recording and return its metadata."""
    normalized_path = _normalize_file_path(file_path)
    start = _normalize_datetime(started_at)
    end = _normalize_datetime(ended_at)
    duration_seconds = (end - start).total_seconds()
    if duration_seconds <= 0:
        raise ValueError("A video segment must end after it starts.")

    now = datetime.now(timezone.utc)
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO video_segments (
                conversation_id,
                file_path,
                started_at,
                ended_at,
                duration_seconds,
                status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                normalized_path,
                start.isoformat(),
                end.isoformat(),
                duration_seconds,
                VideoSegmentStatus.PENDING.value,
                now.isoformat(),
                now.isoformat(),
            ),
        )
    segment_id = cursor.lastrowid
    if segment_id is None:
        raise RuntimeError("SQLite did not return an ID for the video segment.")
    segment = get_video_segment(conn, segment_id)
    if segment is None:
        raise RuntimeError("The created video segment could not be reloaded.")
    return segment


def get_video_segment(
    conn: sqlite3.Connection,
    segment_id: int,
) -> VideoSegment | None:
    """Return one video segment by ID."""
    row = conn.execute(
        "SELECT * FROM video_segments WHERE id = ?",
        (segment_id,),
    ).fetchone()
    return _segment_from_row(row) if row is not None else None


def claim_video_segment_for_analysis(
    conn: sqlite3.Connection,
    segment_id: int,
) -> VideoSegment | None:
    """Atomically move one pending or failed segment into processing state."""
    now = datetime.now(timezone.utc).isoformat()
    with conn:
        cursor = conn.execute(
            """
            UPDATE video_segments
            SET status = ?, error = NULL, updated_at = ?
            WHERE id = ? AND status IN (?, ?)
            """,
            (
                VideoSegmentStatus.PROCESSING.value,
                now,
                segment_id,
                VideoSegmentStatus.PENDING.value,
                VideoSegmentStatus.FAILED.value,
            ),
        )
    if cursor.rowcount == 0:
        return None
    claimed = get_video_segment(conn, segment_id)
    if claimed is None:
        raise RuntimeError("The claimed video segment could not be reloaded.")
    return claimed


def update_video_segment_status(
    conn: sqlite3.Connection,
    segment_id: int,
    status: VideoSegmentStatus,
    *,
    error: str | None = None,
) -> VideoSegment:
    """Update overview-analysis state and preserve a useful failure reason."""
    if status is VideoSegmentStatus.FAILED and not (error and error.strip()):
        raise ValueError("A failed video segment requires an error message.")
    stored_error = error.strip() if error and error.strip() else None
    with conn:
        cursor = conn.execute(
            """
            UPDATE video_segments
            SET status = ?, error = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                status.value,
                stored_error,
                datetime.now(timezone.utc).isoformat(),
                segment_id,
            ),
        )
    if cursor.rowcount == 0:
        raise ValueError(f"Unknown video segment: {segment_id}.")
    segment = get_video_segment(conn, segment_id)
    if segment is None:
        raise RuntimeError("The updated video segment could not be reloaded.")
    return segment


def save_video_report(
    conn: sqlite3.Connection,
    *,
    segment_id: int,
    mode: VideoReportMode,
    provider: str,
    model_name: str,
    content: Mapping[str, object],
    question: str | None = None,
) -> VideoReport:
    """Persist a structured overview or question-specific detail report."""
    normalized_question = question.strip() if question and question.strip() else None
    if mode is VideoReportMode.DETAIL and normalized_question is None:
        raise ValueError("A detailed video report requires its question.")
    if mode is VideoReportMode.OVERVIEW and normalized_question is not None:
        raise ValueError("An overview video report cannot have a question.")
    if not provider.strip() or not model_name.strip():
        raise ValueError("A video report requires provider and model names.")

    serialized = json.dumps(dict(content), ensure_ascii=False, sort_keys=True)
    now = datetime.now(timezone.utc).isoformat()
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO video_reports (
                segment_id,
                mode,
                question,
                provider,
                model_name,
                report_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                segment_id,
                mode.value,
                normalized_question,
                provider.strip(),
                model_name.strip(),
                serialized,
                now,
            ),
        )
        if mode is VideoReportMode.OVERVIEW:
            conn.execute(
                """
                UPDATE video_segments
                SET status = ?, error = NULL, updated_at = ?
                WHERE id = ?
                """,
                (VideoSegmentStatus.COMPLETED.value, now, segment_id),
            )
    report_id = cursor.lastrowid
    if report_id is None:
        raise RuntimeError("SQLite did not return an ID for the video report.")
    row = conn.execute(
        "SELECT * FROM video_reports WHERE id = ?",
        (report_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError("The created video report could not be reloaded.")
    return _report_from_row(row)


def get_video_reports(
    conn: sqlite3.Connection,
    segment_id: int,
    *,
    mode: VideoReportMode | None = None,
) -> list[VideoReport]:
    """Return reports for one segment in creation order."""
    if mode is None:
        rows = conn.execute(
            "SELECT * FROM video_reports WHERE segment_id = ? ORDER BY id",
            (segment_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT * FROM video_reports
            WHERE segment_id = ? AND mode = ?
            ORDER BY id
            """,
            (segment_id, mode.value),
        ).fetchall()
    return [_report_from_row(row) for row in rows]


def _normalize_file_path(file_path: str | Path) -> str:
    raw_path = str(file_path).strip()
    if not raw_path:
        raise ValueError("A video segment file path cannot be empty.")
    return str(Path(raw_path))


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Video segment timestamps must include a timezone.")
    return value.astimezone(timezone.utc)


def _parse_datetime(value: str | datetime) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _segment_from_row(row: sqlite3.Row) -> VideoSegment:
    return VideoSegment(
        id=row["id"],
        conversation_id=row["conversation_id"],
        file_path=Path(row["file_path"]),
        started_at=_parse_datetime(row["started_at"]),
        ended_at=_parse_datetime(row["ended_at"]),
        duration_seconds=row["duration_seconds"],
        status=VideoSegmentStatus(row["status"]),
        error=row["error"],
        created_at=_parse_datetime(row["created_at"]),
        updated_at=_parse_datetime(row["updated_at"]),
    )


def _report_from_row(row: sqlite3.Row) -> VideoReport:
    return VideoReport(
        id=row["id"],
        segment_id=row["segment_id"],
        mode=VideoReportMode(row["mode"]),
        question=row["question"],
        provider=row["provider"],
        model_name=row["model_name"],
        content=json.loads(row["report_json"]),
        created_at=_parse_datetime(row["created_at"]),
    )
