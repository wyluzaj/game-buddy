import asyncio
from datetime import datetime, timezone

import pytest
from pydantic_ai_harness.memory import InMemoryStore, Memory

from buddy.game_memory import ensure_game_document, link_session_to_games
from buddy.memory_recall import MemoryRecallService, MemoryRecallType
from buddy.session_memory import persist_session_document


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
