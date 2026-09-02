"""SQLite metadata and report storage for locally recorded video segments."""

import json
import re
import sqlite3
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Mapping

from pydantic import ValidationError

from buddy.video.models import (
    EpisodeAnalysis,
    EpisodeObservation,
    StoredEpisodeObservation,
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
            game_document_id INTEGER,
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
                REFERENCES conversations (id) ON DELETE SET NULL,
            FOREIGN KEY (game_document_id)
                REFERENCES memory_documents (id) ON DELETE SET NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS video_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            segment_id INTEGER NOT NULL,
            mode TEXT NOT NULL CHECK (mode IN ('overview', 'episode', 'detail')),
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
        CREATE TABLE IF NOT EXISTS video_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id INTEGER NOT NULL,
            segment_id INTEGER NOT NULL,
            start_seconds REAL NOT NULL CHECK (start_seconds >= 0),
            end_seconds REAL NOT NULL CHECK (end_seconds >= start_seconds),
            activity TEXT NOT NULL,
            location TEXT,
            description TEXT NOT NULL,
            visible_objects_json TEXT NOT NULL DEFAULT '[]',
            visible_text_json TEXT NOT NULL DEFAULT '[]',
            uncertainties_json TEXT NOT NULL DEFAULT '[]',
            FOREIGN KEY (report_id)
                REFERENCES video_reports (id) ON DELETE CASCADE,
            FOREIGN KEY (segment_id)
                REFERENCES video_segments (id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS video_observation_search USING fts5(
            observation_id UNINDEXED,
            activity,
            location,
            description,
            visible_objects,
            visible_text,
            tokenize = 'unicode61 remove_diacritics 2'
        )
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS video_observations_after_delete
        AFTER DELETE ON video_observations
        BEGIN
            DELETE FROM video_observation_search
            WHERE observation_id = OLD.id;
        END
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
        CREATE INDEX IF NOT EXISTS idx_video_segments_game_started
        ON video_segments (game_document_id, started_at)
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
        CREATE INDEX IF NOT EXISTS idx_video_observations_segment_time
        ON video_observations (segment_id, start_seconds, end_seconds)
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
    game_document_id: int | None = None,
    recording_key: str | None = None,
) -> VideoSegment:
    """Persist one completed local recording and return its metadata."""
    normalized_path = _normalize_file_path(file_path)
    start = _normalize_datetime(started_at)
    end = _normalize_datetime(ended_at)
    duration_seconds = (end - start).total_seconds()
    if duration_seconds <= 0:
        raise ValueError("A video segment must end after it starts.")
    _validate_game_document_id(conn, game_document_id)
    normalized_recording_key = recording_key or _metadata_recording_key(
        normalized_path,
        start,
        end,
        conversation_id,
        game_document_id,
    )

    now = datetime.now(timezone.utc)
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO video_segments (
                conversation_id,
                game_document_id,
                file_path,
                recording_key,
                started_at,
                ended_at,
                duration_seconds,
                status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                game_document_id,
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


def assign_video_segment_game(
    conn: sqlite3.Connection,
    segment_id: int,
    game_document_id: int | None,
) -> VideoSegment:
    """Attach or clear one confirmed indexed game profile for a segment."""
    if get_video_segment(conn, segment_id) is None:
        raise ValueError(f"Unknown video segment: {segment_id}.")
    _validate_game_document_id(conn, game_document_id)
    with conn:
        conn.execute(
            "UPDATE video_segments SET game_document_id = ? WHERE id = ?",
            (game_document_id, segment_id),
        )
    updated = get_video_segment(conn, segment_id)
    if updated is None:
        raise RuntimeError("The updated video segment could not be reloaded.")
    return updated


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
    """Persist one validated episode, legacy overview, or detail report."""
    normalized_question = question.strip() if question and question.strip() else None
    if mode is VideoReportMode.DETAIL and normalized_question is None:
        raise ValueError("A detailed video report requires its question.")
    if mode is not VideoReportMode.DETAIL and normalized_question is not None:
        raise ValueError("A primary video report cannot have a question.")
    if not provider.strip() or not model_name.strip():
        raise ValueError("A video report requires provider and model names.")
    normalized_request_key = request_key.strip() if request_key else None
    if mode is not VideoReportMode.DETAIL and normalized_request_key is not None:
        raise ValueError("A primary report cannot have a detail request key.")

    segment = get_video_segment(conn, segment_id)
    if segment is None:
        raise ValueError(f"Unknown video segment: {segment_id}.")
    if mode in (VideoReportMode.OVERVIEW, VideoReportMode.EPISODE):
        if segment.status is not VideoSegmentStatus.PROCESSING:
            raise ValueError("A primary report requires a processing segment.")
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
        if mode is VideoReportMode.EPISODE:
            _save_episode_observations(
                conn,
                report_id=cursor.lastrowid,
                segment_id=segment_id,
                analysis=EpisodeAnalysis.model_validate(content),
            )
        if mode in (VideoReportMode.OVERVIEW, VideoReportMode.EPISODE):
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


def get_episode_observations(
    conn: sqlite3.Connection,
    segment_id: int,
) -> list[StoredEpisodeObservation]:
    """Return one episode's normalized observations in timeline order."""
    rows = conn.execute(
        """
        SELECT * FROM video_observations
        WHERE segment_id = ?
        ORDER BY start_seconds, end_seconds, id
        """,
        (segment_id,),
    ).fetchall()
    return [_episode_observation_from_row(row) for row in rows]


def search_episode_observations(
    conn: sqlite3.Connection,
    *,
    query: str,
    conversation_id: int | None = None,
    game_document_id: int | None = None,
    occurred_from: datetime | None = None,
    occurred_before: datetime | None = None,
    limit: int = 5,
) -> list[StoredEpisodeObservation]:
    """Search bounded episode evidence by text plus session/game/date filters."""
    fts_query = _build_fts_query(query)
    if not 1 <= limit <= 100:
        raise ValueError("Video observation search limit must be between 1 and 100.")

    filters: list[str] = []
    parameters: list[object] = [fts_query]
    if conversation_id is not None:
        filters.append("segments.conversation_id = ?")
        parameters.append(conversation_id)
    if game_document_id is not None:
        _validate_game_document_id(conn, game_document_id)
        filters.append("segments.game_document_id = ?")
        parameters.append(game_document_id)
    if occurred_from is not None:
        filters.append("segments.started_at >= ?")
        parameters.append(_normalize_datetime(occurred_from).isoformat())
    if occurred_before is not None:
        filters.append("segments.started_at < ?")
        parameters.append(_normalize_datetime(occurred_before).isoformat())
    where_filters = "".join(f" AND {item}" for item in filters)
    parameters.append(limit)

    rows = conn.execute(
        f"""
        SELECT observations.*
        FROM video_observation_search AS search
        JOIN video_observations AS observations
          ON observations.id = search.observation_id
        JOIN video_segments AS segments
          ON segments.id = observations.segment_id
        WHERE video_observation_search MATCH ?
          {where_filters}
        ORDER BY bm25(video_observation_search),
                 segments.started_at DESC,
                 observations.start_seconds
        LIMIT ?
        """,
        parameters,
    ).fetchall()
    return [_episode_observation_from_row(row) for row in rows]


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


def _save_episode_observations(
    conn: sqlite3.Connection,
    *,
    report_id: int | None,
    segment_id: int,
    analysis: EpisodeAnalysis,
) -> None:
    if report_id is None:
        raise RuntimeError("SQLite did not return an ID for the episode report.")
    for observation in analysis.observations:
        cursor = conn.execute(
            """
            INSERT INTO video_observations (
                report_id,
                segment_id,
                start_seconds,
                end_seconds,
                activity,
                location,
                description,
                visible_objects_json,
                visible_text_json,
                uncertainties_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report_id,
                segment_id,
                observation.start_seconds,
                observation.end_seconds,
                observation.activity,
                observation.location,
                observation.description,
                json.dumps(observation.visible_objects, ensure_ascii=False),
                json.dumps(observation.visible_text, ensure_ascii=False),
                json.dumps(observation.uncertainties, ensure_ascii=False),
            ),
        )
        observation_id = cursor.lastrowid
        if observation_id is None:
            raise RuntimeError("SQLite did not return an observation ID.")
        conn.execute(
            """
            INSERT INTO video_observation_search (
                observation_id,
                activity,
                location,
                description,
                visible_objects,
                visible_text
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                observation_id,
                observation.activity,
                observation.location or "",
                observation.description,
                "\n".join(observation.visible_objects),
                "\n".join(observation.visible_text),
            ),
        )


def _build_fts_query(query: str) -> str:
    tokens = re.findall(r"\w+", query.casefold(), flags=re.UNICODE)
    if not tokens:
        raise ValueError("Video observation search query must contain text.")
    return " ".join(f'"{token}"*' for token in tokens)


def _validate_game_document_id(
    conn: sqlite3.Connection,
    game_document_id: int | None,
) -> None:
    if game_document_id is None:
        return
    row = conn.execute(
        "SELECT document_type FROM memory_documents WHERE id = ?",
        (game_document_id,),
    ).fetchone()
    if row is None or row["document_type"] != "game":
        raise ValueError(
            f"Memory document {game_document_id} is not an indexed game profile."
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
        game_document_id=row["game_document_id"],
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


def _episode_observation_from_row(
    row: sqlite3.Row,
) -> StoredEpisodeObservation:
    try:
        visible_objects = _json_string_list(row["visible_objects_json"])
        visible_text = _json_string_list(row["visible_text_json"])
        uncertainties = _json_string_list(row["uncertainties_json"])
        validated = EpisodeObservation(
            start_seconds=row["start_seconds"],
            end_seconds=row["end_seconds"],
            activity=row["activity"],
            location=row["location"],
            description=row["description"],
            visible_objects=visible_objects,
            visible_text=visible_text,
            uncertainties=uncertainties,
        )
        return StoredEpisodeObservation(
            id=row["id"],
            report_id=row["report_id"],
            segment_id=row["segment_id"],
            **validated.model_dump(),
        )
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise VideoStorageDataError(
            "Stored episode observation is invalid."
        ) from exc


def _json_string_list(value: object) -> list[str]:
    if not isinstance(value, str):
        raise TypeError("Stored JSON list must be text.")
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) for item in parsed
    ):
        raise TypeError("Stored JSON value must contain a string list.")
    return parsed


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
    models = {
        VideoReportMode.OVERVIEW: VideoOverview,
        VideoReportMode.EPISODE: EpisodeAnalysis,
        VideoReportMode.DETAIL: VideoDetailAnalysis,
    }
    model = models[mode]
    model.model_validate(content)


def _metadata_recording_key(
    file_path: str,
    started_at: datetime,
    ended_at: datetime,
    conversation_id: int | None,
    game_document_id: int | None,
) -> str:
    raw = "\0".join(
        (
            file_path,
            started_at.isoformat(),
            ended_at.isoformat(),
            str(conversation_id),
            str(game_document_id),
        )
    )
    return sha256(raw.encode()).hexdigest()


def _migrate_legacy_video_schema(conn: sqlite3.Connection) -> None:
    """Upgrade older video tables without losing reports or stable IDs."""
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
    report_table = conn.execute(
        """
        SELECT sql FROM sqlite_master
        WHERE type = 'table' AND name = 'video_reports'
        """
    ).fetchone()
    report_table_sql = report_table["sql"] if report_table is not None else ""
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
        and "game_document_id" in segment_columns
        and "request_key" in report_columns
        and "'episode'" in report_table_sql
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
                    game_document_id INTEGER,
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
                        REFERENCES conversations (id) ON DELETE SET NULL,
                    FOREIGN KEY (game_document_id)
                        REFERENCES memory_documents (id) ON DELETE SET NULL
                )
                """
            )
            recording_key_sql = (
                "COALESCE(recording_key, 'legacy:' || id)"
                if "recording_key" in segment_columns
                else "'legacy:' || id"
            )
            game_document_sql = (
                "game_document_id" if "game_document_id" in segment_columns else "NULL"
            )
            conn.execute(
                f"""
                INSERT INTO video_segments_new (
                    id, conversation_id, game_document_id, file_path, recording_key,
                    started_at, ended_at, duration_seconds, status, error, created_at,
                    updated_at
                )
                SELECT id, conversation_id, {game_document_sql}, file_path,
                       {recording_key_sql}, started_at, ended_at, duration_seconds,
                       status, error, created_at, updated_at
                FROM video_segments
                """
            )
            conn.execute(
                """
                CREATE TABLE video_reports_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    segment_id INTEGER NOT NULL,
                    mode TEXT NOT NULL
                        CHECK (mode IN ('overview', 'episode', 'detail')),
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
