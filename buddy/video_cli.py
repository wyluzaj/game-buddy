"""Manual command-line entry point for end-to-end video overview testing."""

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import sys

from dotenv import load_dotenv

from buddy.database import init_db
from buddy.gemini_video_analyzer import GeminiVideoAnalyzer
from buddy.video_analyzer import VideoAnalysisError
from buddy.video_analysis_service import (
    VideoAnalysisOutcome,
    register_and_analyze_video,
)


DEFAULT_MANUAL_VIDEO_DURATION_SECONDS = 120.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze one completed local gameplay video with Gemini.",
    )
    parser.add_argument("video_path", type=Path, help="Path to a completed video file.")
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=DEFAULT_MANUAL_VIDEO_DURATION_SECONDS,
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
        default="data/chatbot.db",
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
    except (VideoAnalysisError, sqlite3.Error, OSError, ValueError) as exc:
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
) -> VideoAnalysisOutcome:
    async with GeminiVideoAnalyzer() as analyzer:
        return await register_and_analyze_video(
            conn,
            analyzer,
            file_path=video_path,
            started_at=started_at,
            ended_at=ended_at,
            conversation_id=conversation_id,
        )


def _print_outcome(outcome: VideoAnalysisOutcome) -> None:
    print(f"Segment: {outcome.segment.id} ({outcome.segment.source_uri})")
    print(f"Status: {outcome.segment.status.value}")
    print(f"Model: {outcome.report.provider}/{outcome.report.model_name}")
    print(
        "Report: "
        + ("reused existing result" if outcome.reused_existing_report else "created")
    )
    print(f"Summary: {outcome.overview.summary}")
    if outcome.overview.events:
        print("Events:")
        for event in outcome.overview.events:
            print(
                f"  - {event.start_seconds:.1f}-{event.end_seconds:.1f}s "
                f"[{event.importance.value}]: {event.description}"
            )
    if outcome.overview.uncertainties:
        print("Uncertainties:")
        for uncertainty in outcome.overview.uncertainties:
            print(f"  - {uncertainty}")


if __name__ == "__main__":
    raise SystemExit(main())
