"""Consolidate completed sessions into long-term Harness Memory."""

from dataclasses import dataclass, field
import sqlite3
from typing import Protocol

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models import Model as PydanticModel
from pydantic_ai_harness.memory import Memory

from buddy.config import LONG_TERM_MEMORY_MODEL, build_model
from buddy.database import (
    get_conversations_pending_memory,
    get_messages,
    mark_memory_analyzed,
)
from buddy.long_term_memory import MEMORY_STORE_PATH
from buddy.prompts import (
    LONG_TERM_MEMORY_ANALYSIS_PROMPT,
    LONG_TERM_MEMORY_ANALYZER_INSTRUCTIONS,
)
from buddy.timed_compaction import format_messages


class MemoryAnalysisResult(BaseModel):
    """Structured decision returned by the long-term-memory curator."""

    changed: bool
    memory_markdown: str


class MemoryAnalyzerRunResult(Protocol):
    """Minimal result interface required from a memory analyzer."""

    @property
    def output(self) -> MemoryAnalysisResult: ...


class MemoryAnalyzer(Protocol):
    """Minimal async analyzer interface used by memory processing."""

    async def run(self, prompt: str) -> MemoryAnalyzerRunResult: ...


@dataclass
class MemoryProcessingReport:
    """Outcome of processing all sessions waiting for long-term memory."""

    processed: int = 0
    updated: int = 0
    failed_conversation_ids: list[int] = field(default_factory=list)


def build_memory_analyzer(
    model: str | PydanticModel | None = None,
) -> Agent[None, MemoryAnalysisResult]:
    """Build the dedicated agent that curates completed session memory."""
    analyzer_model = model if model is not None else build_model(LONG_TERM_MEMORY_MODEL)
    return Agent[None, MemoryAnalysisResult](
        model=analyzer_model,
        deps_type=type(None),
        instructions=LONG_TERM_MEMORY_ANALYZER_INSTRUCTIONS,
        output_type=MemoryAnalysisResult,
    )


def build_memory_analysis_prompt(
    current_memory: str,
    session_context: str,
) -> str:
    """Build a clearly delimited prompt for memory consolidation."""
    current = current_memory or "(MEMORY.md jest obecnie puste)"
    user_statements = "\n".join(
        line
        for line in session_context.splitlines()
        if line.lstrip().startswith("User:")
    )
    return LONG_TERM_MEMORY_ANALYSIS_PROMPT.format(
        current_memory=current,
        user_statements=user_statements,
        session_context=session_context,
    )


async def update_long_term_memory(
    analyzer: MemoryAnalyzer,
    memory: Memory,
    session_context: str,
) -> bool:
    """Merge one completed session into MEMORY.md and report whether it changed."""
    session_context = session_context.strip()
    if not session_context:
        return False
    if not any(
        line.lstrip().startswith("User:")
        for line in session_context.splitlines()
    ):
        return False

    memory_file = await memory.store.read(MEMORY_STORE_PATH, max_chars=100_000)
    current_memory = memory_file.content if memory_file is not None else ""
    prompt = build_memory_analysis_prompt(current_memory, session_context)

    result = await analyzer.run(prompt)
    analysis = result.output
    if not analysis.changed:
        return False

    updated_memory = analysis.memory_markdown.strip()
    if not updated_memory:
        raise ValueError("Memory analyzer returned an empty changed memory.")
    if updated_memory == current_memory.strip():
        return False

    await memory.store.write(
        MEMORY_STORE_PATH,
        updated_memory,
        expected_version=memory_file.version if memory_file is not None else None,
    )
    return True


def _deserialize_context_snapshot(snapshot: str) -> str:
    """Normalize current text snapshots and legacy ModelMessages JSON snapshots."""
    snapshot = snapshot.strip()
    if not snapshot:
        return ""
    try:
        messages = ModelMessagesTypeAdapter.validate_json(snapshot)
    except ValueError:
        return snapshot
    return format_messages(messages)


def _load_raw_conversation_context(
    conn: sqlite3.Connection,
    conversation_id: int,
) -> str:
    """Build analyzer input from raw messages when no final snapshot exists."""
    messages = []
    for raw_message_list in get_messages(conn, conversation_id):
        messages.extend(ModelMessagesTypeAdapter.validate_json(raw_message_list))
    return format_messages(messages)


async def process_pending_memories(
    conn: sqlite3.Connection,
    analyzer: MemoryAnalyzer,
    memory: Memory,
) -> MemoryProcessingReport:
    """Process ended sessions once, leaving failed sessions pending for retry."""
    report = MemoryProcessingReport()
    for row in get_conversations_pending_memory(conn):
        conversation_id = row["id"]
        try:
            snapshot = row["compacted_history"] or ""
            session_context = _deserialize_context_snapshot(snapshot)
            if not session_context:
                session_context = _load_raw_conversation_context(
                    conn,
                    conversation_id,
                )

            changed = await update_long_term_memory(
                analyzer,
                memory,
                session_context,
            )
            mark_memory_analyzed(conn, conversation_id)
            report.processed += 1
            if changed:
                report.updated += 1
        # One broken session must stay pending without blocking later sessions.
        # noinspection PyBroadException
        except Exception:
            report.failed_conversation_ids.append(conversation_id)

    return report
