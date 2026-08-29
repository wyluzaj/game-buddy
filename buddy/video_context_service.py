"""Persist idempotent Video Context Butler decisions for overview reports."""

import sqlite3

from pydantic import ValidationError
from pydantic_ai import RunUsage

from buddy.video_context_butler import (
    VideoContextButler,
    build_video_context_prompt,
    validate_video_context_result,
)
from buddy.video_models import (
    VideoContextUpdate,
    VideoOverview,
    VideoReportMode,
    VideoSegment,
)
from buddy.video_storage import (
    get_latest_video_context,
    get_video_context_update_for_report,
    get_video_report,
    get_video_segment,
    save_video_context_update,
)


class VideoContextStateError(RuntimeError):
    """Persisted video data cannot be processed into a context update."""


async def process_video_report_context(
    conn: sqlite3.Connection,
    butler: VideoContextButler,
    report_id: int,
    *,
    usage: RunUsage | None = None,
) -> VideoContextUpdate:
    """Reduce one overview into a bounded replacement state exactly once."""
    existing = get_video_context_update_for_report(conn, report_id)
    if existing is not None:
        return existing

    report = get_video_report(conn, report_id)
    if report is None:
        raise ValueError(f"Unknown video report: {report_id}.")
    if report.mode is not VideoReportMode.OVERVIEW:
        raise ValueError("Only overview reports can update current video context.")
    segment = get_video_segment(conn, report.segment_id)
    if segment is None:
        raise VideoContextStateError("The overview's source segment is missing.")
    try:
        overview = VideoOverview.model_validate(report.content)
    except ValidationError as exc:
        raise VideoContextStateError("The overview report is invalid.") from exc

    previous = get_latest_video_context(
        conn,
        conversation_id=segment.conversation_id,
    )
    if previous is not None and _is_newer_than_current(
        conn,
        previous,
        current_segment=segment,
    ):
        return save_video_context_update(
            conn,
            report_id=report.id,
            changed=False,
            context="",
            sources=[],
            uncertainty=(
                "Pominięto spóźniony raport, ponieważ nowszy fragment wideo "
                "zaktualizował już bieżący kontekst."
            ),
        )
    built = build_video_context_prompt(segment, overview, previous)
    run_result = await butler.run(built.prompt, usage=usage)
    validated = validate_video_context_result(
        run_result.output,
        allowed_sources=built.allowed_sources,
    )
    try:
        return save_video_context_update(
            conn,
            report_id=report.id,
            changed=validated.changed,
            context=validated.context,
            sources=validated.sources,
            uncertainty=validated.uncertainty,
        )
    except sqlite3.IntegrityError:
        # A concurrent worker may have persisted the same unique report marker.
        concurrent = get_video_context_update_for_report(conn, report.id)
        if concurrent is None:
            raise
        return concurrent


def _is_newer_than_current(
    conn: sqlite3.Connection,
    previous: VideoContextUpdate,
    *,
    current_segment: VideoSegment,
) -> bool:
    previous_report = get_video_report(conn, previous.report_id)
    if previous_report is None:
        raise VideoContextStateError("The previous context report is missing.")
    previous_segment = get_video_segment(conn, previous_report.segment_id)
    if previous_segment is None:
        raise VideoContextStateError("The previous context segment is missing.")
    return (previous_segment.ended_at, previous_segment.id) > (
        current_segment.ended_at,
        current_segment.id,
    )
