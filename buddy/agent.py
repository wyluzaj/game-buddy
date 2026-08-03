"""Pydantic AI Agent factory and history filter."""
from datetime import datetime, timedelta, timezone
from pydantic_ai import Agent, ModelMessage
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from buddy.config import USE_LAST_N, USE_LAST_MINUTES, HISTORY_LIMIT, HISTORY_MINUTES

INSTRUCTIONS = """
Jestes buddym do gadania podczas grania w gry.
Odpowiadaj krotko (max 2-3 zdania), naturalnie, po polsku, luzno.

Jesli nie znasz odpowiedzi na pytanie — wywolaj tool
`duckduckgo_search`. Nie zgaduj i nie mow "to sie jeszcze nie odbylo",
jesli mozesz to sprawdzic. Jesli wyszukiwanie nic sensownego nie
zwroci — przyznaj sie, ze nie wiesz.
""".strip()


def keep_recent(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Return messages from the last HISTORY_MINUTES, or last HISTORY_LIMIT if fewer."""
    if USE_LAST_MINUTES:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=HISTORY_MINUTES)
        result = [m for m in messages if m.timestamp is not None and m.timestamp >= cutoff]
    else:
        result = list(messages)
    if USE_LAST_N and len(result) < HISTORY_LIMIT:
        result = messages[-HISTORY_LIMIT:]
    return result


def build_agent(model) -> Agent:
    """Builds and returns an Agent instance with the specified model and instructions."""
    return Agent(
        model=model,
        instructions=INSTRUCTIONS,
        capabilities=[
            ProcessHistory(keep_recent),
        ],
        tools=[duckduckgo_search_tool()],
    )
