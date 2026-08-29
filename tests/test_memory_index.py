"""Behavior tests for searchable memory metadata and session/game links."""

from datetime import datetime, timezone

import pytest

from buddy.database import create_conversation
from buddy.memory_index import (
    MemoryDocumentType,
    MemoryRelationType,
    delete_memory_document,
    get_memory_document,
    index_memory_document,
    link_memory_documents,
    search_memory_documents,
)


def _time(day: int, hour: int = 20) -> datetime:
    return datetime(2026, 8, day, hour, tzinfo=timezone.utc)


def _index_game(db, *, slug: str, title: str):
    return index_memory_document(
        db,
        document_type=MemoryDocumentType.GAME,
        path=f"main/games/{slug}.md",
        title=title,
        summary=f"Profil gry {title}.",
        body=f"# {title}\nPreferencje i aktualne cele.",
    )


def _index_session(db, *, number: int, body: str, day: int):
    return index_memory_document(
        db,
        document_type=MemoryDocumentType.SESSION,
        path=f"main/sessions/session-{number}.md",
        title=f"Sesja {number}",
        summary=body,
        body=body,
        occurred_at=_time(day),
    )


def _link_session_to_game(db, session_id: int, game_id: int) -> None:
    link_memory_documents(
        db,
        source_document_id=session_id,
        relation_type=MemoryRelationType.BELONGS_TO_GAME,
        target_document_id=game_id,
    )


def test_indexed_document_can_be_reloaded_and_searched(db):
    """Indexing persists metadata and makes the document body searchable."""
    conversation_id = create_conversation(db)
    document = index_memory_document(
        db,
        document_type=MemoryDocumentType.SESSION,
        path="main/sessions/session-42.md",
        title="Sesja Cyberpunk",
        summary="Infiltracja magazynu.",
        body="Użytkowniczka znalazła czerwone drzwi obok magazynu.",
        conversation_id=conversation_id,
        occurred_at=_time(18),
    )

    assert get_memory_document(db, document.path) == document
    assert search_memory_documents(db, query="czerwone drzwi") == [document]
    assert document.occurred_at == "2026-08-18 20:00:00"


def test_reindexing_replaces_stale_search_content_but_not_document_type(db):
    """A living game profile updates in place without changing its identity."""
    original = index_memory_document(
        db,
        document_type=MemoryDocumentType.GAME,
        path="main/games/elden-ring.md",
        title="Elden Ring",
        summary="Poszukiwanie klucza.",
        body="# Elden Ring\nTrzeba znaleźć bursztynowy klucz.",
    )
    updated = index_memory_document(
        db,
        document_type=MemoryDocumentType.GAME,
        path=original.path,
        title="Elden Ring",
        summary="Otwarcie zamkniętej wieży.",
        body="Klucz odnaleziony. Teraz trzeba otworzyć zieloną wieżę.",
    )

    assert updated.id == original.id
    assert search_memory_documents(db, query="zieloną wieżę") == [updated]
    assert search_memory_documents(db, query="bursztynowy") == []

    with pytest.raises(ValueError, match="cannot change its type"):
        index_memory_document(
            db,
            document_type=MemoryDocumentType.SESSION,
            path=original.path,
            title="Invalid session",
            summary="Invalid",
            body="Invalid",
        )


def test_one_session_can_be_related_to_multiple_games_without_duplicates(db):
    """Changing games mid-conversation creates two idempotent relationships."""
    elden_ring = _index_game(db, slug="elden-ring", title="Elden Ring")
    cyberpunk = _index_game(db, slug="cyberpunk-2077", title="Cyberpunk 2077")
    session = _index_session(
        db,
        number=11,
        body="Zmiana gry z Elden Ring na Cyberpunk.",
        day=25,
    )

    _link_session_to_game(db, session.id, elden_ring.id)
    _link_session_to_game(db, session.id, cyberpunk.id)
    _link_session_to_game(db, session.id, cyberpunk.id)

    assert search_memory_documents(
        db,
        document_type=MemoryDocumentType.SESSION,
        game_document_id=elden_ring.id,
    ) == [session]
    assert search_memory_documents(
        db,
        document_type=MemoryDocumentType.SESSION,
        game_document_id=cyberpunk.id,
    ) == [session]
    assert db.execute("SELECT COUNT(*) FROM memory_relations").fetchone()[0] == 2


def test_search_combines_topic_game_and_time_without_reading_other_sessions(db):
    """Only matching Cyberpunk sessions from the requested week are returned."""
    elden_ring = _index_game(db, slug="elden-ring", title="Elden Ring")
    cyberpunk = _index_game(db, slug="cyberpunk-2077", title="Cyberpunk 2077")
    session_2 = _index_session(
        db,
        number=2,
        body="Cyberpunk: zamknięte drzwi przy magazynie.",
        day=18,
    )
    session_5 = _index_session(
        db,
        number=5,
        body="Elden Ring: zamknięte drzwi w zamku.",
        day=19,
    )
    session_7 = _index_session(
        db,
        number=7,
        body="Cyberpunk: zamknięte drzwi obok informatora.",
        day=21,
    )
    session_8 = _index_session(
        db,
        number=8,
        body="Cyberpunk: zamknięte drzwi w Night City.",
        day=25,
    )
    _link_session_to_game(db, session_2.id, cyberpunk.id)
    _link_session_to_game(db, session_5.id, elden_ring.id)
    _link_session_to_game(db, session_7.id, cyberpunk.id)
    _link_session_to_game(db, session_8.id, cyberpunk.id)

    results = search_memory_documents(
        db,
        query="zamknięte drzwi",
        document_type=MemoryDocumentType.SESSION,
        game_document_id=cyberpunk.id,
        occurred_from=_time(17, 0),
        occurred_before=_time(24, 0),
    )

    assert results == [session_7, session_2]


def test_search_handles_arbitrary_query_text_and_enforces_result_limit(db):
    """User text cannot become FTS syntax and a valid limit bounds results."""
    for number, day in [(1, 18), (2, 19), (3, 20)]:
        _index_session(db, number=number, body="Eksploracja zamku.", day=day)

    assert search_memory_documents(db, query='" OR ) *') == []
    assert len(search_memory_documents(db, query="zamku", limit=2)) == 2

    with pytest.raises(ValueError, match="search limit"):
        search_memory_documents(db, limit=0)


def test_search_rejects_an_inverted_time_range(db):
    """A reversed interval fails explicitly instead of returning confusing data."""
    with pytest.raises(ValueError, match="earlier than"):
        search_memory_documents(
            db,
            occurred_from=_time(24),
            occurred_before=_time(17),
        )


def test_index_rejects_paths_outside_the_main_memory_scope(db):
    """Indexed paths cannot escape the logical Harness Memory namespace."""
    with pytest.raises(ValueError, match="under main/"):
        index_memory_document(
            db,
            document_type=MemoryDocumentType.GAME,
            path="main/../secrets.md",
            title="Invalid",
            summary="Invalid",
            body="Invalid",
        )


def test_game_relationship_rejects_non_session_sources(db):
    """The session/game relation cannot silently connect two game profiles."""
    first_game = _index_game(db, slug="elden-ring", title="Elden Ring")
    second_game = _index_game(db, slug="cyberpunk-2077", title="Cyberpunk 2077")

    with pytest.raises(ValueError, match="session to a game"):
        _link_session_to_game(db, first_game.id, second_game.id)


def test_deleting_document_removes_its_relations_and_search_entry(db):
    """Deleting metadata cannot leave stale graph or full-text records behind."""
    game = _index_game(db, slug="elden-ring", title="Elden Ring")
    session = _index_session(
        db,
        number=1,
        body="Spotkanie przy zielonej wieży.",
        day=18,
    )
    _link_session_to_game(db, session.id, game.id)

    assert delete_memory_document(db, session.path) is True
    assert delete_memory_document(db, session.path) is False
    assert get_memory_document(db, session.path) is None
    assert search_memory_documents(db, query="zielonej wieży") == []
    assert db.execute("SELECT COUNT(*) FROM memory_relations").fetchone()[0] == 0
