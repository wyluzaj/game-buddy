from buddy.database import create_conversation, save_messages, get_messages


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
