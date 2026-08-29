"""SQLite metadata and report storage for locally recorded video segments."""

import json
import sqlite3
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Mapping

from pydantic import ValidationError

from buddy.video.models import (
    VideoContextUpdate,
    VideoDetailAnalysis,
    VideoOverview,
    VideoReport,
    VideoReportMode,
    VideoSegment,
    VideoSegmentStatus,
)


class VideoStorageDataError(RuntimeError):
    """Persisted video data does not match the application schema."""


def init_video_storage(conn: sqlite3.Connection) -> None:
    """Create video segment and analysis report tables."""
    _migrate_legacy_video_schema(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS video_segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            file_path TEXT NOT NULL,
            recording_key TEXT NOT NULL UNIQUE,
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
            request_key TEXT,
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
        CREATE TABLE IF NOT EXISTS video_context_updates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id INTEGER NOT NULL UNIQUE,
            changed INTEGER NOT NULL CHECK (changed IN (0, 1)),
            context TEXT NOT NULL DEFAULT '',
            sources_json TEXT NOT NULL DEFAULT '[]',
            uncertainty TEXT,
            created_at TEXT NOT NULL,
            injected_at TEXT,
            FOREIGN KEY (report_id)
                REFERENCES video_reports (id) ON DELETE CASCADE
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
        CREATE INDEX IF NOT EXISTS idx_video_segments_path_created
        ON video_segments (file_path, created_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_video_reports_segment_mode
        ON video_reports (segment_id, mode, created_at)
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_video_reports_request_key
        ON video_reports (request_key)
        WHERE request_key IS NOT NULL
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_video_context_updates_pending
        ON video_context_updates (changed, injected_at, created_at)
        """
    )


def create_video_segment(
    conn: sqlite3.Connection,
    *,
    file_path: str | Path,
    started_at: datetime,
    ended_at: datetime,
    conversation_id: int | None = None,
    recording_key: str | None = None,
) -> VideoSegment:
    """Persist one completed local recording and return its metadata."""
    normalized_path = _normalize_file_path(file_path)
    start = _normalize_datetime(started_at)
    end = _normalize_datetime(ended_at)
    duration_seconds = (end - start).total_seconds()
    if duration_seconds <= 0:
        raise ValueError("A video segment must end after it starts.")
    normalized_recording_key = recording_key or _metadata_recording_key(
        normalized_path,
        start,
        end,
        conversation_id,
    )

    now = datetime.now(timezone.utc)
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO video_segments (
                conversation_id,
                file_path,
                recording_key,
                started_at,
                ended_at,
                duration_seconds,
                status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                normalized_path,
                normalized_recording_key,
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


def get_video_segment_by_path(
    conn: sqlite3.Connection,
    file_path: str | Path,
) -> VideoSegment | None:
    """Return the newest registered segment for one normalized local path."""
    normalized_path = _normalize_file_path(file_path)
    row = conn.execute(
        """
        SELECT * FROM video_segments
        WHERE file_path = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (normalized_path,),
    ).fetchone()
    return _segment_from_row(row) if row is not None else None


def get_video_segment_by_recording_key(
    conn: sqlite3.Connection,
    recording_key: str,
) -> VideoSegment | None:
    """Return the exact logical recording represented by a stable key."""
    row = conn.execute(
        "SELECT * FROM video_segments WHERE recording_key = ?",
        (recording_key,),
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
    """Apply one allowed overview-analysis state transition."""
    if status is VideoSegmentStatus.FAILED and not (error and error.strip()):
        raise ValueError("A failed video segment requires an error message.")
    current = get_video_segment(conn, segment_id)
    if current is None:
        raise ValueError(f"Unknown video segment: {segment_id}.")
    allowed = {
        VideoSegmentStatus.PENDING: {VideoSegmentStatus.PROCESSING},
        VideoSegmentStatus.PROCESSING: {
            VideoSegmentStatus.PENDING,
            VideoSegmentStatus.COMPLETED,
            VideoSegmentStatus.FAILED,
        },
        VideoSegmentStatus.FAILED: {VideoSegmentStatus.PROCESSING},
        VideoSegmentStatus.COMPLETED: set(),
    }
    if status not in allowed[current.status]:
        raise ValueError(
            f"Invalid video segment transition: {current.status.value} -> "
            f"{status.value}."
        )
    stored_error = error.strip() if error and error.strip() else None
    with conn:
        cursor = conn.execute(
            """
            UPDATE video_segments
            SET status = ?, error = ?, updated_at = ?
            WHERE id = ? AND status = ?
            """,
            (
                status.value,
                stored_error,
                datetime.now(timezone.utc).isoformat(),
                segment_id,
                current.status.value,
            ),
        )
    if cursor.rowcount == 0:
        raise RuntimeError("Video segment state changed concurrently.")
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
    request_key: str | None = None,
) -> VideoReport:
    """Persist a structured overview or question-specific detail report."""
    normalized_question = question.strip() if question and question.strip() else None
    if mode is VideoReportMode.DETAIL and normalized_question is None:
        raise ValueError("A detailed video report requires its question.")
    if mode is VideoReportMode.OVERVIEW and normalized_question is not None:
        raise ValueError("An overview video report cannot have a question.")
    if not provider.strip() or not model_name.strip():
        raise ValueError("A video report requires provider and model names.")
    normalized_request_key = request_key.strip() if request_key else None
    if mode is VideoReportMode.OVERVIEW and normalized_request_key is not None:
        raise ValueError("An overview report cannot have a detail request key.")

    segment = get_video_segment(conn, segment_id)
    if segment is None:
        raise ValueError(f"Unknown video segment: {segment_id}.")
    if mode is VideoReportMode.OVERVIEW:
        if segment.status is not VideoSegmentStatus.PROCESSING:
            raise ValueError("An overview report requires a processing segment.")
        _validate_report_content(mode, content)
    elif segment.status is not VideoSegmentStatus.COMPLETED:
        raise ValueError("A detail report requires a completed segment.")
    else:
        _validate_report_content(mode, content)

    serialized = json.dumps(dict(content), ensure_ascii=False, sort_keys=True)
    now = datetime.now(timezone.utc).isoformat()
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO video_reports (
                segment_id,
                mode,
                question,
                request_key,
                provider,
                model_name,
                report_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                segment_id,
                mode.value,
                normalized_question,
                normalized_request_key,
                provider.strip(),
                model_name.strip(),
                serialized,
                now,
            ),
        )
        if mode is VideoReportMode.OVERVIEW:
            completed = conn.execute(
                """
                UPDATE video_segments
                SET status = ?, error = NULL, updated_at = ?
                WHERE id = ? AND status = ?
                """,
                (
                    VideoSegmentStatus.COMPLETED.value,
                    now,
                    segment_id,
                    VideoSegmentStatus.PROCESSING.value,
                ),
            )
            if completed.rowcount == 0:
                raise RuntimeError("Video segment state changed concurrently.")
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


def get_video_report(
    conn: sqlite3.Connection,
    report_id: int,
) -> VideoReport | None:
    """Return one exact persisted video report."""
    row = conn.execute(
        "SELECT * FROM video_reports WHERE id = ?",
        (report_id,),
    ).fetchone()
    return _report_from_row(row) if row is not None else None


def get_video_report_by_request_key(
    conn: sqlite3.Connection,
    request_key: str,
) -> VideoReport | None:
    """Return one cached detail report by its canonical request key."""
    row = conn.execute(
        "SELECT * FROM video_reports WHERE request_key = ?",
        (request_key,),
    ).fetchone()
    return _report_from_row(row) if row is not None else None


def save_video_context_update(
    conn: sqlite3.Connection,
    *,
    report_id: int,
    changed: bool,
    context: str,
    sources: list[str],
    uncertainty: str | None = None,
) -> VideoContextUpdate:
    """Persist one idempotency marker and optional replacement context state."""
    normalized_context = context.strip()
    normalized_sources = list(dict.fromkeys(source.strip() for source in sources if source.strip()))
    if changed and (not normalized_context or not normalized_sources):
        raise ValueError("A changed video context requires text and sources.")
    if not changed:
        normalized_context = ""
        normalized_sources = []
    normalized_uncertainty = uncertainty.strip() if uncertainty and uncertainty.strip() else None
    now = datetime.now(timezone.utc).isoformat()
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO video_context_updates (
                report_id,
                changed,
                context,
                sources_json,
                uncertainty,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                report_id,
                int(changed),
                normalized_context,
                json.dumps(normalized_sources, ensure_ascii=False),
                normalized_uncertainty,
                now,
            ),
        )
    update_id = cursor.lastrowid
    if update_id is None:
        raise RuntimeError("SQLite did not return an ID for the video context update.")
    row = conn.execute(
        "SELECT * FROM video_context_updates WHERE id = ?",
        (update_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError("The video context update could not be reloaded.")
    return _context_update_from_row(row)


def get_video_context_update_for_report(
    conn: sqlite3.Connection,
    report_id: int,
) -> VideoContextUpdate | None:
    """Return the existing Butler decision for one overview report."""
    row = conn.execute(
        "SELECT * FROM video_context_updates WHERE report_id = ?",
        (report_id,),
    ).fetchone()
    return _context_update_from_row(row) if row is not None else None


def get_latest_video_context(
    conn: sqlite3.Connection,
    *,
    conversation_id: int | None,
) -> VideoContextUpdate | None:
    """Return the latest meaningful replacement state in one conversation."""
    if conversation_id is None:
        return None
    row = conn.execute(
        """
        SELECT updates.*
        FROM video_context_updates AS updates
        JOIN video_reports AS reports ON reports.id = updates.report_id
        JOIN video_segments AS segments ON segments.id = reports.segment_id
        WHERE updates.changed = 1
          AND segments.conversation_id = ?
        ORDER BY segments.ended_at DESC, segments.id DESC, updates.id DESC
        LIMIT 1
        """,
        (conversation_id,),
    ).fetchone()
    return _context_update_from_row(row) if row is not None else None


def get_latest_pending_video_context(
    conn: sqlite3.Connection,
    *,
    conversation_id: int,
) -> VideoContextUpdate | None:
    """Return the newest meaningful state not yet shown to its chat session."""
    row = conn.execute(
        """
        SELECT updates.*
        FROM video_context_updates AS updates
        JOIN video_reports AS reports ON reports.id = updates.report_id
        JOIN video_segments AS segments ON segments.id = reports.segment_id
        WHERE updates.changed = 1
          AND updates.injected_at IS NULL
          AND segments.conversation_id = ?
        ORDER BY segments.ended_at DESC, segments.id DESC, updates.id DESC
        LIMIT 1
        """,
        (conversation_id,),
    ).fetchone()
    return _context_update_from_row(row) if row is not None else None


def get_video_context_updates_for_conversation(
    conn: sqlite3.Connection,
    *,
    conversation_id: int,
    changed_only: bool = True,
) -> list[VideoContextUpdate]:
    """Return Butler decisions for one session in gameplay chronology."""
    changed_clause = "AND updates.changed = 1" if changed_only else ""
    rows = conn.execute(
        f"""
        SELECT updates.*
        FROM video_context_updates AS updates
        JOIN video_reports AS reports ON reports.id = updates.report_id
        JOIN video_segments AS segments ON segments.id = reports.segment_id
        WHERE segments.conversation_id = ?
          {changed_clause}
        ORDER BY segments.ended_at, segments.id, updates.id
        """,
        (conversation_id,),
    ).fetchall()
    return [_context_update_from_row(row) for row in rows]


def mark_video_context_injected_through(
    conn: sqlite3.Connection,
    *,
    conversation_id: int,
    update_id: int,
) -> None:
    """Acknowledge the selected state and every older state it supersedes."""
    target = conn.execute(
        """
        SELECT segments.ended_at, segments.id AS segment_id
        FROM video_context_updates AS updates
        JOIN video_reports AS reports ON reports.id = updates.report_id
        JOIN video_segments AS segments ON segments.id = reports.segment_id
        WHERE updates.id = ?
          AND segments.conversation_id = ?
        """,
        (update_id, conversation_id),
    ).fetchone()
    if target is None:
        raise ValueError(
            f"Video context update {update_id} does not belong to conversation "
            f"{conversation_id}."
        )
    now = datetime.now(timezone.utc).isoformat()
    with conn:
        conn.execute(
            """
            UPDATE video_context_updates
            SET injected_at = ?
            WHERE injected_at IS NULL
              AND id IN (
                  SELECT updates.id
                  FROM video_context_updates AS updates
                  JOIN video_reports AS reports ON reports.id = updates.report_id
                  JOIN video_segments AS segments ON segments.id = reports.segment_id
                  WHERE updates.changed = 1
                    AND segments.conversation_id = ?
                    AND (
                        segments.ended_at < ?
                        OR (
                            segments.ended_at = ?
                            AND segments.id <= ?
                        )
                    )
              )
            """,
            (
                now,
                conversation_id,
                target["ended_at"],
                target["ended_at"],
                target["segment_id"],
            ),
        )


def _normalize_file_path(file_path: str | Path) -> str:
    raw_path = str(file_path).strip()
    if not raw_path:
        raise ValueError("A video segment file path cannot be empty.")
    return str(Path(raw_path).expanduser().resolve(strict=False))


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
        recording_key=row["recording_key"],
        started_at=_parse_datetime(row["started_at"]),
        ended_at=_parse_datetime(row["ended_at"]),
        duration_seconds=row["duration_seconds"],
        status=VideoSegmentStatus(row["status"]),
        error=row["error"],
        created_at=_parse_datetime(row["created_at"]),
        updated_at=_parse_datetime(row["updated_at"]),
    )


def _report_from_row(row: sqlite3.Row) -> VideoReport:
    try:
        mode = VideoReportMode(row["mode"])
        content = json.loads(row["report_json"])
        if not isinstance(content, dict):
            raise TypeError("report_json must decode to an object")
        _validate_report_content(mode, content)
        return VideoReport(
            id=row["id"],
            segment_id=row["segment_id"],
            mode=mode,
            question=row["question"],
            request_key=row["request_key"],
            provider=row["provider"],
            model_name=row["model_name"],
            content=content,
            created_at=_parse_datetime(row["created_at"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, ValidationError) as exc:
        raise VideoStorageDataError("Stored video report is invalid.") from exc


def _context_update_from_row(row: sqlite3.Row) -> VideoContextUpdate:
    try:
        sources = json.loads(row["sources_json"])
        if not isinstance(sources, list) or not all(
            isinstance(source, str) for source in sources
        ):
            raise TypeError("sources_json must decode to a string list")
        changed = bool(row["changed"])
        if changed and (not row["context"].strip() or not sources):
            raise ValueError("changed context requires content and sources")
        return VideoContextUpdate(
            id=row["id"],
            report_id=row["report_id"],
            changed=changed,
            context=row["context"],
            sources=sources,
            uncertainty=row["uncertainty"],
            created_at=_parse_datetime(row["created_at"]),
            injected_at=(
                _parse_datetime(row["injected_at"])
                if row["injected_at"] is not None
                else None
            ),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VideoStorageDataError("Stored video context update is invalid.") from exc


def _validate_report_content(
    mode: VideoReportMode,
    content: Mapping[str, object],
) -> None:
    model = VideoOverview if mode is VideoReportMode.OVERVIEW else VideoDetailAnalysis
    model.model_validate(content)


def _metadata_recording_key(
    file_path: str,
    started_at: datetime,
    ended_at: datetime,
    conversation_id: int | None,
) -> str:
    raw = "\0".join(
        (
            file_path,
            started_at.isoformat(),
            ended_at.isoformat(),
            str(conversation_id),
        )
    )
    return sha256(raw.encode()).hexdigest()


def _migrate_legacy_video_schema(conn: sqlite3.Connection) -> None:
    """Upgrade the pre-versioned video tables without losing reports."""
    table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'video_segments'"
    ).fetchone()
    if table is None:
        return
    segment_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(video_segments)")
    }
    report_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(video_reports)")
    }
    has_unique_path = any(
        index["unique"]
        and [
            column["name"]
            for column in conn.execute(
                f"PRAGMA index_info({index['name']})"
            ).fetchall()
        ] == ["file_path"]
        for index in conn.execute("PRAGMA index_list(video_segments)").fetchall()
    )
    if (
        "recording_key" in segment_columns
        and "request_key" in report_columns
        and not has_unique_path
    ):
        return

    conn.commit()
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE video_segments_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER,
                    file_path TEXT NOT NULL,
                    recording_key TEXT NOT NULL UNIQUE,
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
            recording_key_sql = (
                "COALESCE(recording_key, 'legacy:' || id)"
                if "recording_key" in segment_columns
                else "'legacy:' || id"
            )
            conn.execute(
                f"""
                INSERT INTO video_segments_new (
                    id, conversation_id, file_path, recording_key, started_at,
                    ended_at, duration_seconds, status, error, created_at, updated_at
                )
                SELECT id, conversation_id, file_path, {recording_key_sql}, started_at,
                       ended_at, duration_seconds, status, error, created_at, updated_at
                FROM video_segments
                """
            )
            conn.execute(
                """
                CREATE TABLE video_reports_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    segment_id INTEGER NOT NULL,
                    mode TEXT NOT NULL CHECK (mode IN ('overview', 'detail')),
                    question TEXT,
                    request_key TEXT,
                    provider TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (segment_id)
                        REFERENCES video_segments_new (id) ON DELETE CASCADE
                )
                """
            )
            request_key_sql = (
                "request_key" if "request_key" in report_columns else "NULL"
            )
            conn.execute(
                f"""
                INSERT INTO video_reports_new (
                    id, segment_id, mode, question, request_key, provider,
                    model_name, report_json, created_at
                )
                SELECT id, segment_id, mode, question, {request_key_sql}, provider,
                       model_name, report_json, created_at
                FROM video_reports
                """
            )
            conn.execute(
                """
                CREATE TABLE video_context_updates_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id INTEGER NOT NULL UNIQUE,
                    changed INTEGER NOT NULL CHECK (changed IN (0, 1)),
                    context TEXT NOT NULL DEFAULT '',
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    uncertainty TEXT,
                    created_at TEXT NOT NULL,
                    injected_at TEXT,
                    FOREIGN KEY (report_id)
                        REFERENCES video_reports_new (id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                INSERT INTO video_context_updates_new (
                    id, report_id, changed, context, sources_json,
                    uncertainty, created_at, injected_at
                )
                SELECT id, report_id, changed, context, sources_json,
                       uncertainty, created_at, injected_at
                FROM video_context_updates
                """
            )
            conn.execute("DROP TABLE video_context_updates")
            conn.execute("DROP TABLE video_reports")
            conn.execute("DROP TABLE video_segments")
            conn.execute("ALTER TABLE video_segments_new RENAME TO video_segments")
            conn.execute("ALTER TABLE video_reports_new RENAME TO video_reports")
            conn.execute(
                "ALTER TABLE video_context_updates_new RENAME TO video_context_updates"
            )
    finally:
        conn.execute("PRAGMA foreign_keys = ON")
