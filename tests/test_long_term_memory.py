import asyncio

from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.memory import InMemoryStore, Memory

from buddy.agent import build_agent
from buddy.config import SHORT_TERM_MEMORY_MODEL
from buddy.long_term_memory import MEMORY_STORE_PATH, create_long_term_memory
from buddy.timed_compaction import SummarizingCompactionTimed


def test_long_term_memory_persists_between_store_instances(tmp_path):
    """MEMORY.md written by one store is readable after reopening the database."""
    async def write_and_reopen():
        database_path = tmp_path / "memory.db"
        first_memory = create_long_term_memory(database_path)
        await first_memory.store.write(
            MEMORY_STORE_PATH,
            "Użytkowniczka lubi eksplorację.",
            expected_version=None,
        )

        reopened_memory = create_long_term_memory(database_path)
        return await reopened_memory.store.read(MEMORY_STORE_PATH, max_chars=10_000)

    memory_file = asyncio.run(write_and_reopen())

    assert memory_file is not None
    assert memory_file.content == "Użytkowniczka lubi eksplorację."


def test_agent_contains_injected_memory_capability():
    """The agent uses the supplied Memory capability instance."""
    memory = Memory(InMemoryStore())

    agent = build_agent(TestModel(), memory=memory)

    assert memory in agent.root_capability.capabilities


def test_agent_uses_configured_short_term_memory_model():
    """Short-term summarization uses the model selected in project config."""
    agent = build_agent(TestModel(), memory=Memory(InMemoryStore()))

    short_term_memory = next(
        capability
        for capability in agent.root_capability.capabilities
        if isinstance(capability, SummarizingCompactionTimed)
    )

    assert short_term_memory.model == SHORT_TERM_MEMORY_MODEL.value
