"""Render bounded, source-aware video observations for completed sessions."""

import sqlite3

from buddy.config import VIDEO
from buddy.video.storage import (
    get_video_context_updates_for_conversation,
    get_video_report,
    get_video_segment,
)


def build_session_video_observations(
    conn: sqlite3.Connection,
    conversation_id: int,
) -> str:
    """Build chronological evidence without exposing technical DB metadata."""
    updates = get_video_context_updates_for_conversation(
        conn,
        conversation_id=conversation_id,
        changed_only=False,
    )[-VIDEO.session_observations:]
    blocks: list[str] = []
    for update in updates:
        report = get_video_report(conn, update.report_id)
        segment = get_video_segment(conn, report.segment_id) if report else None
        summary = update.episode_summary or update.context
        if not summary or segment is None:
            continue
        block = f"- {summary}\n  Sources: {segment.source_uri}"
        if update.uncertainty:
            block += f"\n  Uncertainty: {update.uncertainty}"
        blocks.append(block)
    selected: list[str] = []
    selected_chars = 0
    for block in reversed(blocks):
        separator_chars = 1 if selected else 0
        if (
            selected_chars + separator_chars + len(block)
            > VIDEO.session_observation_chars
        ):
            break
        selected.append(block)
        selected_chars += separator_chars + len(block)
    return "\n".join(reversed(selected))
