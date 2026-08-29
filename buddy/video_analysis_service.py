"""Application service coordinating persisted segments and video providers."""

import asyncio
from dataclasses import dataclass
import sqlite3

from pydantic import ValidationError

from buddy.video_analyzer import VideoAnalyzer
from buddy.video_models import (
    VideoOverview,
    VideoReport,
    VideoReportMode,
    VideoSegment,
    VideoSegmentStatus,
)
from buddy.video_storage import (
    claim_video_segment_for_analysis,
    get_video_reports,
    get_video_segment,
    save_video_report,
    update_video_segment_status,
)


MAX_STORED_VIDEO_ERROR_CHARS = 2_000


class VideoSegmentStateError(RuntimeError):
    """A segment cannot be analyzed from its current persisted state."""


@dataclass(frozen=True, slots=True)
class VideoAnalysisOutcome:
    """Persisted result returned to a CLI, worker, or future context pipeline."""

    segment: VideoSegment
    report: VideoReport
    overview: VideoOverview
    reused_existing_report: bool


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
    return reason[:MAX_STORED_VIDEO_ERROR_CHARS]
