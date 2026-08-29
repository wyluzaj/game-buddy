"""Persist completed short-memory snapshots as searchable session documents."""

from datetime import datetime, timezone
import re
import sqlite3

from pydantic_ai_harness.memory import Memory

from buddy.memory_index import (
    MemoryDocument,
    MemoryDocumentType,
    index_memory_document,
)


SESSION_MEMORY_PREFIX = "main/sessions"
MAX_SESSION_DOCUMENT_CHARS = 1_000_000
MAX_SESSION_INDEX_SUMMARY_CHARS = 500


def session_memory_path(conversation_id: int) -> str:
    """Return the stable Harness Memory path for one conversation report."""
    if conversation_id < 1:
        raise ValueError("A conversation ID must be positive.")
    return f"{SESSION_MEMORY_PREFIX}/session-{conversation_id}.md"


def build_session_document(
    *,
    conversation_id: int,
    started_at: datetime,
    ended_at: datetime,
    session_context: str,
) -> str:
    """Wrap the final short-memory snapshot in a small Markdown report."""
    context = session_context.strip()
    if not context:
        raise ValueError("A session report cannot be built from empty context.")

    return (
        f"# Session {conversation_id}\n\n"
        f"- Started: {_format_timestamp(started_at)}\n"
        f"- Ended: {_format_timestamp(ended_at)}\n\n"
        f"## Session Report\n\n"
        f"{context}\n"
    )


async def persist_session_document(
    conn: sqlite3.Connection,
    memory: Memory,
    *,
    conversation_id: int,
    started_at: datetime,
    ended_at: datetime,
    session_context: str,
) -> MemoryDocument:
    """Write one idempotent session report and refresh its search metadata."""
    path = session_memory_path(conversation_id)
    content = build_session_document(
        conversation_id=conversation_id,
        started_at=started_at,
        ended_at=ended_at,
        session_context=session_context,
    )
    current = await memory.store.read(path, max_chars=MAX_SESSION_DOCUMENT_CHARS)
    if current is None or current.truncated or current.content != content:
        await memory.store.write(
            path,
            content,
            expected_version=current.version if current is not None else None,
        )

    return index_memory_document(
        conn,
        document_type=MemoryDocumentType.SESSION,
        path=path,
        title=f"Sesja {conversation_id} — {ended_at.date().isoformat()}",
        summary=_index_summary(session_context),
        body=content,
        conversation_id=conversation_id,
        occurred_at=ended_at,
    )


def parse_database_timestamp(value: str | datetime) -> datetime:
    """Interpret SQLite CURRENT_TIMESTAMP values as UTC datetimes."""
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    """Render report metadata consistently in UTC."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def _index_summary(session_context: str) -> str:
    """Create bounded display metadata; the full body remains in FTS."""
    summary = re.sub(r"\s+", " ", session_context).strip()
    if len(summary) <= MAX_SESSION_INDEX_SUMMARY_CHARS:
        return summary
    return summary[: MAX_SESSION_INDEX_SUMMARY_CHARS - 1].rstrip() + "…"
