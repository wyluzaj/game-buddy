import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest
from pydantic_ai import RunUsage
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai_harness.memory import InMemoryStore, Memory

from buddy.config import CONTEXT_BUTLER_MODEL, MODEL_TIERS
from buddy.context_butler import ContextButlerResult, build_context_butler
from buddy.game_memory import ensure_game_document, link_session_to_games
from buddy.memory_recall import (
    MemoryRecallCoordinator,
    MemoryRecallService,
    MemoryRecallType,
)
from buddy.session_memory import persist_session_document


@dataclass
class StubButlerRunResult:
    output: ContextButlerResult


class StubContextButler:
    def __init__(self, output: ContextButlerResult):
        self.output = output
        self.prompts: list[str] = []
        self.usages: list[RunUsage | None] = []

    async def run(
        self,
        prompt: str,
        *,
        usage: RunUsage | None = None,
    ) -> StubButlerRunResult:
        self.prompts.append(prompt)
        self.usages.append(usage)
        return StubButlerRunResult(self.output)


class FailIfCalledContextButler:
    async def run(
        self,
        prompt: str,
        *,
        usage: RunUsage | None = None,
    ) -> StubButlerRunResult:
        raise AssertionError("Context Butler must not run without memory matches.")


async def _add_session(
    db,
    memory: Memory,
    *,
    conversation_id: int,
    game_name: str,
    day: int,
    context: str,
):
    started = datetime(2026, 8, day, 18, tzinfo=timezone.utc)
    ended = datetime(2026, 8, day, 20, tzinfo=timezone.utc)
    db.execute(
        """
        INSERT INTO conversations (id, created_at, ended_at, compacted_history)
        VALUES (?, ?, ?, ?)
        """,
        (conversation_id, started.isoformat(), ended.isoformat(), context),
    )
    db.commit()
    session = await persist_session_document(
        db,
        memory,
        conversation_id=conversation_id,
        started_at=started,
        ended_at=ended,
        session_context=context,
    )
    game = await ensure_game_document(db, memory, game_name)
    await link_session_to_games(
        db,
        memory,
        session_document=session,
        game_names=[game_name],
    )
    return session, game


def test_recall_returns_game_profile_and_only_its_matching_sessions(db):
    async def scenario():
        memory = Memory(InMemoryStore())
        elden_session, elden_game = await _add_session(
            db,
            memory,
            conversation_id=1,
            game_name="Elden Ring",
            day=20,
            context=(
                f"User: {'Długi wcześniejszy kontekst. ' * 180}"
                "Zamknięte drzwi są pod zieloną wieżą."
            ),
        )
        await _add_session(
            db,
            memory,
            conversation_id=2,
            game_name="Cyberpunk 2077",
            day=21,
            context="User: Zamknięte drzwi są przy magazynie.",
        )

        result = await MemoryRecallService(db, memory).recall(
            query="zamknięte drzwi",
            game="Elden Ring",
            memory_type=MemoryRecallType.ALL,
            limit=3,
        )
        return result, elden_session.path, elden_game.path

    result, session_path, game_path = asyncio.run(scenario())

    assert result.used_recent_fallback is False
    assert [match.path for match in result.matches] == [game_path, session_path]
    assert "zieloną wieżą" in result.matches[1].content
    assert result.matches[1].truncated is True
    assert result.matches[1].games == ["Elden Ring"]
    assert all("cyberpunk" not in match.path for match in result.matches)


def test_recall_marks_lexical_fallback_and_enforces_its_public_limit(db):
    async def scenario():
        memory = Memory(InMemoryStore())
        session, _ = await _add_session(
            db,
            memory,
            conversation_id=1,
            game_name="Elden Ring",
            day=20,
            context="User: Wyszłam z zamku i dotarłam do mostu.",
        )
        service = MemoryRecallService(db, memory)
        result = await service.recall(
            query="zamek",
            game="Elden Ring",
            memory_type=MemoryRecallType.SESSION,
            date_from="2026-08-20",
            date_to="2026-08-20",
            limit=1,
        )
        with pytest.raises(ValueError, match="Recall limit"):
            await service.recall(limit=6)
        return result, session.path

    result, session_path = asyncio.run(scenario())

    assert result.used_recent_fallback is True
    assert [match.path for match in result.matches] == [session_path]
    assert "No exact text match" in result.message


def test_context_butler_uses_explicit_configured_model(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")

    butler = build_context_butler()

    assert isinstance(butler.model, FallbackModel)
    expected_models = [
        model.value.split(":", 1)[1]
        for model, tier in MODEL_TIERS.items()
        if tier == MODEL_TIERS[CONTEXT_BUTLER_MODEL]
    ]
    assert [model.model_name for model in butler.model.models] == expected_models


def test_coordinator_returns_only_reduced_context_and_forwards_usage(db):
    async def scenario():
        memory = Memory(InMemoryStore())
        session, _ = await _add_session(
            db,
            memory,
            conversation_id=1,
            game_name="Elden Ring",
            day=20,
            context=(
                "User: Zamknięte drzwi są pod zieloną wieżą. "
                "User: Nie lubię spoilerów."
            ),
        )
        butler = StubContextButler(
            ContextButlerResult(
                enough_evidence=True,
                context="Ostatnie zamknięte drzwi były pod zieloną wieżą.",
                sources=[session.path],
            )
        )
        usage = RunUsage()
        coordinator = MemoryRecallCoordinator(
            MemoryRecallService(db, memory),
            butler,
        )
        result = await coordinator.recall(
            question="Gdzie były ostatnie zamknięte drzwi?",
            query="zamknięte drzwi",
            game="Elden Ring",
            memory_type=MemoryRecallType.SESSION,
            usage=usage,
        )
        return result, butler, usage

    result, butler, usage = asyncio.run(scenario())

    assert result.context == "Ostatnie zamknięte drzwi były pod zieloną wieżą."
    assert not hasattr(result, "matches")
    assert "Nie lubię spoilerów" in butler.prompts[0]
    assert "Gdzie były ostatnie zamknięte drzwi?" in butler.prompts[0]
    assert butler.usages == [usage]


def test_coordinator_rejects_an_invented_source(db):
    async def scenario():
        memory = Memory(InMemoryStore())
        await _add_session(
            db,
            memory,
            conversation_id=1,
            game_name="Elden Ring",
            day=20,
            context="User: Zamknięte drzwi są pod zieloną wieżą.",
        )
        butler = StubContextButler(
            ContextButlerResult(
                enough_evidence=True,
                context="Wymyślona odpowiedź.",
                sources=["main/sessions/nie-istnieje.md"],
            )
        )
        coordinator = MemoryRecallCoordinator(
            MemoryRecallService(db, memory),
            butler,
        )
        return await coordinator.recall(
            question="Gdzie były drzwi?",
            query="zamknięte drzwi",
            game="Elden Ring",
            memory_type=MemoryRecallType.SESSION,
        )

    result = asyncio.run(scenario())

    assert result.enough_evidence is False
    assert result.context == ""
    assert result.sources == []
    assert result.uncertainty is not None
    assert "spoza wyników" in result.uncertainty


def test_coordinator_skips_delegate_when_nothing_was_found(db):
    async def scenario():
        memory = Memory(InMemoryStore())
        coordinator = MemoryRecallCoordinator(
            MemoryRecallService(db, memory),
            FailIfCalledContextButler(),
        )
        return await coordinator.recall(
            question="Co wydarzyło się wczoraj?",
            query="nieistniejące wydarzenie",
            memory_type=MemoryRecallType.SESSION,
        )

    result = asyncio.run(scenario())

    assert result.enough_evidence is False
    assert result.context == ""
    assert result.sources == []
    assert result.uncertainty == "No memory documents matched the supplied filters."
