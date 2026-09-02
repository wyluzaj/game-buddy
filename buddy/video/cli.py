"""Manual command-line entry point for end-to-end video overview testing."""

import argparse
import asyncio
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

from dotenv import load_dotenv
from pydantic_ai.exceptions import AgentRunError

from buddy.config import STORAGE, VIDEO
from buddy.database import init_db
from buddy.video.analyzer import VideoAnalysisError
from buddy.video.context_butler import VideoContextButler, build_video_context_butler
from buddy.video.gemini import GeminiVideoAnalyzer
from buddy.video.pipeline import VideoProcessingOutcome, process_video_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze one completed local gameplay video with Gemini.",
    )
    parser.add_argument("video_path", type=Path, help="Path to a completed video file.")
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=VIDEO.manual_duration_seconds,
        help="Recorded duration used for timeline metadata (default: 120).",
    )
    parser.add_argument(
        "--conversation-id",
        type=int,
        default=None,
        help="Optional existing conversation ID to associate with the segment.",
    )
    parser.add_argument(
        "--database",
        default=STORAGE.chat_database_path,
        help="SQLite database path (default: data/chatbot.db).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Analyze one file and print a concise human-readable report."""
    args = build_parser().parse_args(argv)
    if args.duration_seconds <= 0:
        print("Error: --duration-seconds must be positive.", file=sys.stderr)
        return 2

    video_path = args.video_path.expanduser().resolve(strict=False)
    if not video_path.is_file():
        print(f"Error: video file does not exist: {video_path}", file=sys.stderr)
        return 2

    ended_at = datetime.fromtimestamp(video_path.stat().st_mtime, tz=timezone.utc)
    started_at = ended_at - timedelta(seconds=args.duration_seconds)

    load_dotenv()
    conn = init_db(args.database)
    try:
        outcome = asyncio.run(
            _run_analysis(
                conn,
                video_path=video_path,
                started_at=started_at,
                ended_at=ended_at,
                conversation_id=args.conversation_id,
            )
        )
    except (
        AgentRunError,
        VideoAnalysisError,
        sqlite3.Error,
        OSError,
        ValueError,
    ) as exc:
        print(f"Video analysis failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    _print_outcome(outcome)
    return 0


async def _run_analysis(
    conn: sqlite3.Connection,
    *,
    video_path: Path,
    started_at: datetime,
    ended_at: datetime,
    conversation_id: int | None,
) -> VideoProcessingOutcome:
    context_butler = build_video_context_butler()
    async with GeminiVideoAnalyzer() as analyzer:
        return await process_video_file(
            conn,
            analyzer,
            cast(VideoContextButler, context_butler),
            file_path=video_path,
            started_at=started_at,
            ended_at=ended_at,
            conversation_id=conversation_id,
        )


def _print_outcome(outcome: VideoProcessingOutcome) -> None:
    analysis = outcome.analysis
    print(f"Segment: {analysis.segment.id} ({analysis.segment.source_uri})")
    print(f"Status: {analysis.segment.status.value}")
    print(f"Model: {analysis.report.provider}/{analysis.report.model_name}")
    print(
        "Report: "
        + (
            "reused existing result"
            if analysis.reused_existing_report
            else "created"
        )
    )
    if outcome.context_update.episode_summary:
        print(f"Episode summary: {outcome.context_update.episode_summary}")
    if analysis.episode.observations:
        print("Episode timeline:")
        for observation in analysis.episode.observations:
            print(
                f"  - {observation.start_seconds:.1f}-"
                f"{observation.end_seconds:.1f}s: {observation.description}"
            )
    if analysis.episode.uncertainties:
        print("Uncertainties:")
        for uncertainty in analysis.episode.uncertainties:
            print(f"  - {uncertainty}")
    context_update = outcome.context_update
    print(
        "Video context decision: "
        + ("meaningful change" if context_update.changed else "no meaningful change")
    )
    if context_update.context:
        print(context_update.context)
    if context_update.sources:
        print("Context sources:")
        for source in context_update.sources:
            print(f"  - {source}")
    if context_update.uncertainty:
        print(f"Context note: {context_update.uncertainty}")


if __name__ == "__main__":
    raise SystemExit(main())
