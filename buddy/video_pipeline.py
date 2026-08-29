"""End-to-end orchestration for overview analysis and compact video context."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sqlite3

from pydantic_ai import RunUsage

from buddy.video_analysis_service import (
    VideoAnalysisOutcome,
    register_and_analyze_video,
)
from buddy.video_analyzer import VideoAnalyzer
from buddy.video_context_butler import VideoContextButler
from buddy.video_context_service import process_video_report_context
from buddy.video_models import VideoContextUpdate


@dataclass(frozen=True, slots=True)
class VideoProcessingOutcome:
    """The durable full report and its bounded context projection."""

    analysis: VideoAnalysisOutcome
    context_update: VideoContextUpdate


async def process_video_file(
    conn: sqlite3.Connection,
    analyzer: VideoAnalyzer,
    context_butler: VideoContextButler,
    *,
    file_path: str | Path,
    started_at: datetime,
    ended_at: datetime,
    conversation_id: int | None = None,
    usage: RunUsage | None = None,
) -> VideoProcessingOutcome:
    """Create or reuse an overview, then create or reuse its context update."""
    analysis = await register_and_analyze_video(
        conn,
        analyzer,
        file_path=file_path,
        started_at=started_at,
        ended_at=ended_at,
        conversation_id=conversation_id,
    )
    context_update = await process_video_report_context(
        conn,
        context_butler,
        analysis.report.id,
        usage=usage,
    )
    return VideoProcessingOutcome(
        analysis=analysis,
        context_update=context_update,
    )
