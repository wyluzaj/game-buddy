"""Pydantic AI Agent factory."""
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai_harness.compaction import ClearToolResults
from pydantic_ai_harness.memory import Memory

from buddy.config import (
    HISTORY_MINUTES,
    MIN_RECENT_MESSAGES,
    Model,
)
from buddy.long_term_memory import create_long_term_memory
from buddy.timed_compaction import SummarizingCompactionTimed

INSTRUCTIONS = """
Jestes buddym do gadania podczas grania w gry.
Odpowiadaj krotko (max 2-3 zdania), naturalnie, po polsku, luzno.

Jesli nie znasz odpowiedzi na pytanie — wywolaj tool
`duckduckgo_search`. Nie zgaduj i nie mow "to sie jeszcze nie odbylo",
jesli mozesz to sprawdzic. Jesli wyszukiwanie nic sensownego nie
zwroci — przyznaj sie, ze nie wiesz.
""".strip()


def build_agent(model, *, memory: Memory | None = None) -> Agent:
    """Builds and returns an Agent instance with the specified model and instructions."""
    long_term_memory = memory or create_long_term_memory()
    return Agent(
        model=model,
        instructions=INSTRUCTIONS,
        capabilities=[
            ClearToolResults(
                max_tokens=50_000,
                keep_pairs=3,
            ),
            SummarizingCompactionTimed(
                model=Model.GEMINI_LITE.value,
                history_minutes=HISTORY_MINUTES,
                min_keep_messages=MIN_RECENT_MESSAGES,
            ),
            long_term_memory,
        ],
        tools=[duckduckgo_search_tool()],
    )
