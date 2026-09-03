import asyncio

from pydantic_ai_harness.memory import Memory

from buddy.audit import export_audit
from buddy.database import create_conversation, init_db, save_messages
from buddy.memory.long_term import create_long_term_memory


def test_audit_export_preserves_unicode_as_utf8(tmp_path):
    chat_database = tmp_path / "chatbot.db"
    memory_database = tmp_path / "memory.db"
    connection = init_db(str(chat_database))
    conversation_id = create_conversation(connection)
    save_messages(connection, conversation_id, "Zażółć gęślą jaźń")
    connection.close()

    memory: Memory = create_long_term_memory(memory_database)

    async def write_memory():
        await memory.store.write(
            "main/MEMORY.md",
            "# User Memory\n\n- The user's exact quote was: Zażółć gęślą jaźń.",
            expected_version=None,
        )

    asyncio.run(write_memory())

    chat_output, memory_output = export_audit(
        database_path=chat_database,
        memory_database_path=memory_database,
        output_directory=tmp_path / "audit",
    )

    assert "Zażółć gęślą jaźń" in chat_output.read_text(encoding="utf-8")
    assert "Zażółć gęślą jaźń" in memory_output.read_text(encoding="utf-8")
    assert memory_output.read_bytes().decode("utf-8").startswith(
        "=== main/MEMORY.md ==="
    )


def test_audit_export_requires_existing_databases(tmp_path):
    try:
        export_audit(
            database_path=tmp_path / "missing-chat.db",
            memory_database_path=tmp_path / "missing-memory.db",
            output_directory=tmp_path / "audit",
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Missing database files should not be created by audit.")
