"""Render bounded, source-aware video observations for completed sessions."""

import sqlite3

from buddy.video_storage import get_video_context_updates_for_conversation


MAX_SESSION_VIDEO_OBSERVATIONS = 50
MAX_SESSION_VIDEO_OBSERVATIONS_CHARS = 30_000


def build_session_video_observations(
    conn: sqlite3.Connection,
    conversation_id: int,
) -> str:
    """Build chronological evidence without exposing technical DB metadata."""
    updates = get_video_context_updates_for_conversation(
        conn,
        conversation_id=conversation_id,
    )[-MAX_SESSION_VIDEO_OBSERVATIONS:]
    blocks: list[str] = []
    for update in updates:
        sources = ", ".join(update.sources)
        block = f"- {update.context}\n  Sources: {sources}"
        if update.uncertainty:
            block += f"\n  Uncertainty: {update.uncertainty}"
        blocks.append(block)
    rendered = "\n".join(blocks)
    if len(rendered) <= MAX_SESSION_VIDEO_OBSERVATIONS_CHARS:
        return rendered
    return rendered[-MAX_SESSION_VIDEO_OBSERVATIONS_CHARS:]
