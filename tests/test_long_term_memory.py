import asyncio

from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.memory import InMemoryStore, Memory

from buddy.agent import build_agent
from buddy.config import MODEL_TIERS, MODELS
from buddy.memory.compaction import SummarizingCompactionTimed
from buddy.memory.long_term import (
    MEMORY_STORE_PATH,
    InjectionOnlyMemory,
    create_long_term_memory,
)


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


def test_application_memory_injects_without_native_memory_tools(tmp_path):
    """Nested session/game documents are accessed only through recall_memory."""
    memory = create_long_term_memory(tmp_path / "memory.db")

    assert isinstance(memory, InjectionOnlyMemory)
    assert memory.inject_memory is True
    assert memory.get_toolset() is None


def test_agent_contains_injected_memory_capability(monkeypatch):
    """The agent uses the supplied Memory capability instance."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    memory = Memory(InMemoryStore())

    agent = build_agent(TestModel(), memory=memory)

    assert memory in agent.root_capability.capabilities


def test_agent_injects_behavior_and_response_language_separately(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")

    agent = build_agent(TestModel(), memory=Memory(InMemoryStore()))

    assert len(agent._instructions) == 2
    assert "friendly buddy" in agent._instructions[0]
    assert "configured default response language" in agent._instructions[1]


def test_agent_uses_configured_short_term_memory_fallback(monkeypatch):
    """Short-term summarization uses its model and same-tier fallback."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    agent = build_agent(TestModel(), memory=Memory(InMemoryStore()))

    short_term_memory = next(
        capability
        for capability in agent.root_capability.capabilities
        if isinstance(capability, SummarizingCompactionTimed)
    )

    assert isinstance(short_term_memory.model, FallbackModel)
    used_names = [model.model_name for model in short_term_memory.model.models]
    assert used_names == [
        MODELS.short_term_memory.value.split(":", 1)[1]
    ] + [
        model.value.split(":", 1)[1]
        for model, tier in MODEL_TIERS.items()
        if tier == MODEL_TIERS[MODELS.short_term_memory]
        and model is not MODELS.short_term_memory
    ]
