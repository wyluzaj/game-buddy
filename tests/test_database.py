import sqlite3
from datetime import datetime, timedelta, timezone

from buddy.database import (
    LATEST_SCHEMA_VERSION,
    create_conversation,
    finish_conversation,
    get_conversations_pending_memory,
    get_messages,
    init_db,
    mark_memory_analyzed,
    save_messages,
)
from buddy.video.storage import create_video_segment


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

    finished = finish_conversation(
        db,
        conversation_id,
        '[{"kind":"summary"}]',
    )

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
    assert finished["id"] == conversation_id
    assert finished["created_at"] is not None
    assert finished["ended_at"] == row["ended_at"]


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
    """Existing databases receive conversation state and the memory index."""
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
    memory_tables = {
        row["name"]
        for row in migrated.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE name IN (
                'memory_documents',
                'memory_relations',
                'memory_search',
                'video_segments',
                'video_reports',
                'video_context_updates',
                'video_observations',
                'video_observation_search'
            )
            """
        ).fetchall()
    }
    migrated.close()

    assert {"ended_at", "compacted_history", "memory_analyzed_at"} <= columns
    assert memory_tables == {
        "memory_documents",
        "memory_relations",
        "memory_search",
        "video_segments",
        "video_reports",
        "video_context_updates",
        "video_observations",
        "video_observation_search",
    }


def test_init_db_versions_and_migrates_legacy_video_schema(tmp_path):
    database_path = tmp_path / "legacy-video.db"
    legacy = sqlite3.connect(database_path)
    legacy.executescript(
        """
        CREATE TABLE conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE video_segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            file_path TEXT NOT NULL UNIQUE,
            started_at TEXT NOT NULL,
            ended_at TEXT NOT NULL,
            duration_seconds REAL NOT NULL,
            status TEXT NOT NULL,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE video_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            segment_id INTEGER NOT NULL,
            mode TEXT NOT NULL,
            question TEXT,
            provider TEXT NOT NULL,
            model_name TEXT NOT NULL,
            report_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE video_context_updates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id INTEGER NOT NULL UNIQUE,
            changed INTEGER NOT NULL,
            context TEXT NOT NULL DEFAULT '',
            sources_json TEXT NOT NULL DEFAULT '[]',
            uncertainty TEXT,
            created_at TEXT NOT NULL,
            injected_at TEXT
        );
        INSERT INTO video_segments (
            id, file_path, started_at, ended_at, duration_seconds,
            status, created_at, updated_at
        ) VALUES (
            1, 'same.mp4', '2026-08-29T18:00:00+00:00',
            '2026-08-29T18:02:00+00:00', 120, 'pending',
            '2026-08-29T18:02:00+00:00', '2026-08-29T18:02:00+00:00'
        );
        """
    )
    legacy.commit()
    legacy.close()

    migrated = init_db(str(database_path))
    segment_columns = {
        row["name"]
        for row in migrated.execute("PRAGMA table_info(video_segments)")
    }
    report_columns = {
        row["name"]
        for row in migrated.execute("PRAGMA table_info(video_reports)")
    }
    context_columns = {
        row["name"]
        for row in migrated.execute("PRAGMA table_info(video_context_updates)")
    }
    legacy_row = migrated.execute(
        "SELECT recording_key FROM video_segments WHERE id = 1"
    ).fetchone()
    started_at = datetime(2026, 8, 29, 19, 0, tzinfo=timezone.utc)
    second = create_video_segment(
        migrated,
        file_path="same.mp4",
        started_at=started_at,
        ended_at=started_at + timedelta(minutes=2),
    )
    schema_version = migrated.execute("PRAGMA user_version").fetchone()[0]
    foreign_key_errors = migrated.execute("PRAGMA foreign_key_check").fetchall()
    report_schema = migrated.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'video_reports'"
    ).fetchone()["sql"]
    migrated.close()

    assert "recording_key" in segment_columns
    assert "game_document_id" in segment_columns
    assert "request_key" in report_columns
    assert "episode_summary" in context_columns
    assert "'episode'" in report_schema
    assert legacy_row["recording_key"] == "legacy:1"
    assert second.id != 1
    assert schema_version == LATEST_SCHEMA_VERSION
    assert foreign_key_errors == []
