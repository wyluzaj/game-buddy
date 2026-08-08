"""Pydantic AI Agent factory."""
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai_harness.compaction import ClearToolResults
from pydantic_ai_harness.memory import Memory

from buddy.config import (
    HISTORY_MINUTES,
    MIN_RECENT_MESSAGES,
    SHORT_TERM_MEMORY_MODEL,
)
from buddy.long_term_memory import create_long_term_memory
from buddy.prompts import CHAT_AGENT_INSTRUCTIONS
from buddy.timed_compaction import SummarizingCompactionTimed


def build_agent(model, *, memory: Memory | None = None) -> Agent:
    """Builds and returns an Agent instance with the specified model and instructions."""
    long_term_memory = memory or create_long_term_memory()
    return Agent(
        model=model,
        instructions=CHAT_AGENT_INSTRUCTIONS,
        capabilities=[
            ClearToolResults(
                max_tokens=50_000,
                keep_pairs=3,
            ),
            SummarizingCompactionTimed(
                model=SHORT_TERM_MEMORY_MODEL.value,
                history_minutes=HISTORY_MINUTES,
                min_keep_messages=MIN_RECENT_MESSAGES,
            ),
            long_term_memory,
        ],
        tools=[duckduckgo_search_tool()],
    )
