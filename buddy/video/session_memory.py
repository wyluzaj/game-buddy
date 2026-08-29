"""Render bounded, source-aware video observations for completed sessions."""

import sqlite3

from buddy.config import VIDEO
from buddy.video.storage import get_video_context_updates_for_conversation


def build_session_video_observations(
    conn: sqlite3.Connection,
    conversation_id: int,
) -> str:
    """Build chronological evidence without exposing technical DB metadata."""
    updates = get_video_context_updates_for_conversation(
        conn,
        conversation_id=conversation_id,
    )[-VIDEO.session_observations:]
    blocks: list[str] = []
    for update in updates:
        sources = ", ".join(update.sources)
        block = f"- {update.context}\n  Sources: {sources}"
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
