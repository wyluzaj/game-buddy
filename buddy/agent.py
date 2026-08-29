"""Pydantic AI Agent factory."""
import sqlite3
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.models import Model as PydanticModel
from pydantic_ai_harness.compaction import ClearToolResults
from pydantic_ai_harness.memory import Memory

from buddy.config import AGENT, MODELS, build_model
from buddy.memory.compaction import SummarizingCompactionTimed
from buddy.memory.long_term import create_long_term_memory
from buddy.memory.recall import create_recall_memory_tool
from buddy.prompts import CHAT_AGENT_INSTRUCTIONS
from buddy.video.detail_tool import create_analyze_video_detail_tool


def build_agent(
    model: str | PydanticModel,
    *,
    memory: Memory | None = None,
    memory_connection: sqlite3.Connection | None = None,
) -> Agent:
    """Builds and returns an Agent instance with the specified model and instructions."""
    long_term_memory = memory or create_long_term_memory()
    tools: list[Any] = [duckduckgo_search_tool()]
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
                max_tokens=AGENT.clear_tool_results_max_tokens,
                keep_pairs=AGENT.clear_tool_results_keep_pairs,
            ),
            SummarizingCompactionTimed(
                model=build_model(MODELS.short_term_memory),
                history_minutes=AGENT.history_minutes,
                min_keep_messages=AGENT.min_recent_messages,
            ),
            long_term_memory,
        ],
        tools=tools,
    )
