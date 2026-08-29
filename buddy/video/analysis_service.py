"""Application service coordinating persisted segments and video providers."""

import asyncio
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from pydantic import ValidationError

from buddy.config import VIDEO
from buddy.video.analyzer import VideoAnalyzer
from buddy.video.models import (
    VideoOverview,
    VideoReport,
    VideoReportMode,
    VideoSegment,
    VideoSegmentStatus,
)
from buddy.video.storage import (
    claim_video_segment_for_analysis,
    create_video_segment,
    get_video_reports,
    get_video_segment,
    get_video_segment_by_recording_key,
    save_video_report,
    update_video_segment_status,
)


class VideoSegmentStateError(RuntimeError):
    """A segment cannot be analyzed from its current persisted state."""


@dataclass(frozen=True, slots=True)
class VideoAnalysisOutcome:
    """Persisted result returned to a CLI, worker, or future context pipeline."""

    segment: VideoSegment
    report: VideoReport
    overview: VideoOverview
    reused_existing_report: bool


async def register_and_analyze_video(
    conn: sqlite3.Connection,
    analyzer: VideoAnalyzer,
    *,
    file_path: str | Path,
    started_at: datetime,
    ended_at: datetime,
    conversation_id: int | None = None,
) -> VideoAnalysisOutcome:
    """Register one local recording once, then run or reuse its overview."""
    path = Path(file_path).expanduser().resolve(strict=False)
    if (
        started_at.tzinfo is None
        or started_at.utcoffset() is None
        or ended_at.tzinfo is None
        or ended_at.utcoffset() is None
    ):
        raise ValueError("Video segment timestamps must include a timezone.")
    normalized_start = started_at.astimezone(timezone.utc)
    normalized_end = ended_at.astimezone(timezone.utc)
    recording_key = _recording_key(
        path,
        started_at=normalized_start,
        ended_at=normalized_end,
        conversation_id=conversation_id,
    )
    existing = get_video_segment_by_recording_key(conn, recording_key)
    segment = existing or create_video_segment(
        conn,
        file_path=path,
        started_at=normalized_start,
        ended_at=normalized_end,
        conversation_id=conversation_id,
        recording_key=recording_key,
    )
    return await analyze_video_segment(conn, analyzer, segment.id)


async def analyze_video_segment(
    conn: sqlite3.Connection,
    analyzer: VideoAnalyzer,
    segment_id: int,
) -> VideoAnalysisOutcome:
    """Analyze one segment exactly once and persist its structured overview."""
    current = get_video_segment(conn, segment_id)
    if current is None:
        raise ValueError(f"Unknown video segment: {segment_id}.")

    if current.status is VideoSegmentStatus.COMPLETED:
        return _load_completed_outcome(conn, current)

    claimed = claim_video_segment_for_analysis(conn, segment_id)
    if claimed is None:
        latest = get_video_segment(conn, segment_id)
        status = latest.status.value if latest is not None else "deleted"
        raise VideoSegmentStateError(
            f"Video segment {segment_id} cannot be claimed from status {status}."
        )

    try:
        overview = await analyzer.analyze_overview(claimed.file_path)
        _validate_overview_timeline(overview, claimed.duration_seconds)
        report = save_video_report(
            conn,
            segment_id=claimed.id,
            mode=VideoReportMode.OVERVIEW,
            provider=analyzer.provider,
            model_name=analyzer.model_name,
            content=overview.model_dump(mode="json"),
        )
    except asyncio.CancelledError:
        update_video_segment_status(
            conn,
            claimed.id,
            VideoSegmentStatus.PENDING,
        )
        raise
    except Exception as exc:
        update_video_segment_status(
            conn,
            claimed.id,
            VideoSegmentStatus.FAILED,
            error=_failure_reason(exc),
        )
        raise

    completed = get_video_segment(conn, claimed.id)
    if completed is None or completed.status is not VideoSegmentStatus.COMPLETED:
        raise RuntimeError("The analyzed video segment was not marked completed.")
    return VideoAnalysisOutcome(
        segment=completed,
        report=report,
        overview=overview,
        reused_existing_report=False,
    )


def _load_completed_outcome(
    conn: sqlite3.Connection,
    segment: VideoSegment,
) -> VideoAnalysisOutcome:
    reports = get_video_reports(
        conn,
        segment.id,
        mode=VideoReportMode.OVERVIEW,
    )
    if not reports:
        raise VideoSegmentStateError(
            f"Completed video segment {segment.id} has no overview report."
        )
    report = reports[-1]
    try:
        overview = VideoOverview.model_validate(report.content)
    except ValidationError as exc:
        raise VideoSegmentStateError(
            f"Completed video segment {segment.id} has an invalid overview report."
        ) from exc
    return VideoAnalysisOutcome(
        segment=segment,
        report=report,
        overview=overview,
        reused_existing_report=True,
    )


def _failure_reason(exc: Exception) -> str:
    message = str(exc).strip()
    reason = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
    return reason[: VIDEO.stored_error_chars]


def _recording_key(
    path: Path,
    *,
    started_at: datetime,
    ended_at: datetime,
    conversation_id: int | None,
) -> str:
    """Hash file bytes and capture metadata into a stable recording identity."""
    if not path.is_file():
        raise ValueError(f"Video file does not exist: {path}")
    digest = sha256()
    with path.open("rb") as video_file:
        while chunk := video_file.read(VIDEO.hash_chunk_bytes):
            digest.update(chunk)
    for value in (started_at.isoformat(), ended_at.isoformat(), str(conversation_id)):
        digest.update(b"\0")
        digest.update(value.encode())
    return digest.hexdigest()


def _validate_overview_timeline(
    overview: VideoOverview,
    duration_seconds: float,
) -> None:
    """Reject provider timestamps outside the persisted clip timeline."""
    for event in overview.events:
        if event.end_seconds > duration_seconds + 0.001:
            raise ValueError("Video overview event exceeds the segment duration.")
    for observation in overview.visible_text:
        if observation.timestamp_seconds > duration_seconds + 0.001:
            raise ValueError(
                "Video overview text observation exceeds the segment duration."
            )
