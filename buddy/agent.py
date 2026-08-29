"""Pydantic AI Agent factory."""
import sqlite3

from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai_harness.compaction import ClearToolResults
from pydantic_ai_harness.memory import Memory

from buddy.config import (
    HISTORY_MINUTES,
    MIN_RECENT_MESSAGES,
    SHORT_TERM_MEMORY_MODEL,
    build_model,
)
from buddy.long_term_memory import create_long_term_memory
from buddy.memory_recall import create_recall_memory_tool
from buddy.prompts import CHAT_AGENT_INSTRUCTIONS
from buddy.timed_compaction import SummarizingCompactionTimed
from buddy.video_detail_tool import create_analyze_video_detail_tool


def build_agent(
    model,
    *,
    memory: Memory | None = None,
    memory_connection: sqlite3.Connection | None = None,
) -> Agent:
    """Builds and returns an Agent instance with the specified model and instructions."""
    long_term_memory = memory or create_long_term_memory()
    tools = [duckduckgo_search_tool()]
    if memory_connection is not None:
        tools.append(
            create_recall_memory_tool(memory_connection, long_term_memory)
        )
        tools.append(create_analyze_video_detail_tool(memory_connection))
    return Agent(
        model=model,
        instructions=CHAT_AGENT_INSTRUCTIONS,
        capabilities=[
            ClearToolResults(
                max_tokens=50_000,
                keep_pairs=3,
            ),
            SummarizingCompactionTimed(
                model=build_model(SHORT_TERM_MEMORY_MODEL),
                history_minutes=HISTORY_MINUTES,
                min_keep_messages=MIN_RECENT_MESSAGES,
            ),
            long_term_memory,
        ],
        tools=tools,
    )
