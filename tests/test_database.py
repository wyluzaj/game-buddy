import sqlite3

from buddy.database import (
    create_conversation,
    finish_conversation,
    get_conversations_pending_memory,
    get_messages,
    init_db,
    mark_memory_analyzed,
    save_messages,
)


def test_get_messages_returns_them_in_insertion_order(db):
    """Verify SELECT ORDER BY id ASC preserves the exact insertion order across multiple rows."""
    conv_id = create_conversation(db)
    save_messages(db, conv_id, "first")
    save_messages(db, conv_id, "second")
    save_messages(db, conv_id, "third")

    assert get_messages(db, conv_id) == ["first", "second", "third"]


def test_get_messages_isolates_conversations(db):
    """Verify the WHERE conversation_id filter prevents messages from leaking between conversations."""
    conv_a = create_conversation(db)
    conv_b = create_conversation(db)
    save_messages(db, conv_a, "A1")
    save_messages(db, conv_b, "B1")
    save_messages(db, conv_a, "A2")

    assert get_messages(db, conv_a) == ["A1", "A2"]
    assert get_messages(db, conv_b) == ["B1"]


def test_finish_conversation_persists_compacted_history(db):
    """An ended session retains the final rolling summary and recent tail."""
    conversation_id = create_conversation(db)

    finish_conversation(db, conversation_id, '[{"kind":"summary"}]')

    row = db.execute(
        """
        SELECT ended_at, compacted_history, memory_analyzed_at
        FROM conversations
        WHERE id = ?
        """,
        (conversation_id,),
    ).fetchone()
    assert row["ended_at"] is not None
    assert row["compacted_history"] == '[{"kind":"summary"}]'
    assert row["memory_analyzed_at"] is None


def test_only_ended_unanalyzed_conversations_are_pending(db):
    """Open and already analyzed sessions are excluded from pending memory work."""
    open_conversation = create_conversation(db)
    pending_conversation = create_conversation(db)
    analyzed_conversation = create_conversation(db)
    finish_conversation(db, pending_conversation, "pending")
    finish_conversation(db, analyzed_conversation, "analyzed")
    mark_memory_analyzed(db, analyzed_conversation)

    pending = get_conversations_pending_memory(db)

    assert [row["id"] for row in pending] == [pending_conversation]
    assert open_conversation not in [row["id"] for row in pending]


def test_init_db_migrates_existing_conversations_table(tmp_path):
    """Existing chatbot databases receive the long-term-memory state columns."""
    database_path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(database_path)
    legacy.execute(
        """
        CREATE TABLE conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    legacy.commit()
    legacy.close()

    migrated = init_db(str(database_path))
    columns = {
        row["name"]
        for row in migrated.execute("PRAGMA table_info(conversations)").fetchall()
    }
    migrated.close()

    assert {"ended_at", "compacted_history", "memory_analyzed_at"} <= columns
