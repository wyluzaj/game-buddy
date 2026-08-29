"""SQLite persistence layer for conversations and messages."""
import sqlite3
from pathlib import Path

from buddy.config import STORAGE
from buddy.memory.index import init_memory_index
from buddy.memory.journal import init_memory_journal
from buddy.video.storage import init_video_storage

_CONVERSATION_COLUMNS: dict[str, str] = {
    "ended_at": "TIMESTAMP",
    "compacted_history": "TEXT",
    "memory_analyzed_at": "TIMESTAMP",
}
LATEST_SCHEMA_VERSION = 3


def _open_connection(db_path: str) -> sqlite3.Connection:
    """Make a new connection to the database and enable foreign keys."""
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = STORAGE.chat_database_path) -> sqlite3.Connection:
    """Initialize the database and return a connection."""
    conn = _open_connection(db_path)

    conn.execute('''
                 CREATE TABLE IF NOT EXISTS conversations
                 (
                     id         INTEGER PRIMARY KEY AUTOINCREMENT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     ended_at TIMESTAMP,
                     compacted_history TEXT,
                     memory_analyzed_at TIMESTAMP
                 )
                 ''')

    existing_columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(conversations)").fetchall()
    }
    for name, column_type in _CONVERSATION_COLUMNS.items():
        if name not in existing_columns:
            conn.execute(
                f"ALTER TABLE conversations ADD COLUMN {name} {column_type}"
            )

    conn.execute('''
                 CREATE TABLE IF NOT EXISTS messages
                 (
                     id              INTEGER PRIMARY KEY AUTOINCREMENT,
                     conversation_id INTEGER NOT NULL,
                     message_list    TEXT    NOT NULL,
                     created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     FOREIGN KEY (conversation_id) REFERENCES conversations (id)
                 )
                 ''')

    init_memory_index(conn)
    init_memory_journal(conn)
    init_video_storage(conn)
    conn.execute(f"PRAGMA user_version = {LATEST_SCHEMA_VERSION}")
    conn.commit()
    return conn


def create_conversation(conn: sqlite3.Connection) -> int:
    """Create a new conversation and return its ID."""
    cursor = conn.execute("INSERT INTO conversations DEFAULT VALUES")
    conn.commit()
    conversation_id = cursor.lastrowid
    if conversation_id is None:
        raise RuntimeError("SQLite did not return an ID for the new conversation.")
    return conversation_id


def save_messages(conn: sqlite3.Connection, conversation_id: int, messages_json: str) -> None:
    """Save a JSON of messages for a given conversation."""
    conn.execute(
        "INSERT INTO messages (conversation_id, message_list) VALUES (?, ?)",
        (conversation_id, messages_json),
    )
    conn.commit()


def get_messages(conn: sqlite3.Connection, conversation_id: int) -> list[str]:
    """Retrieve all messages for a given conversation, ordered by insertion."""
    cursor = conn.execute(
        "SELECT message_list FROM messages WHERE conversation_id = ? ORDER BY id",
        (conversation_id,),
    )
    rows = cursor.fetchall()
    return [row["message_list"] for row in rows]


def finish_conversation(
    conn: sqlite3.Connection,
    conversation_id: int,
    compacted_history: str,
) -> sqlite3.Row:
    """End a conversation and return metadata needed for durable session memory."""
    conn.execute(
        """
        UPDATE conversations
        SET ended_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP,
            compacted_history = ?
        WHERE id = ?
        """,
        (compacted_history, conversation_id),
    )
    conn.commit()
    row = conn.execute(
        """
        SELECT id, created_at, ended_at, compacted_history
        FROM conversations
        WHERE id = ?
        """,
        (conversation_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError("The finished conversation could not be reloaded.")
    return row


def get_conversations_pending_memory(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return ended conversations that have not updated long-term memory yet."""
    cursor = conn.execute(
        """
        SELECT id, created_at, ended_at, compacted_history
        FROM conversations
        WHERE ended_at IS NOT NULL
          AND memory_analyzed_at IS NULL
        ORDER BY id
        """
    )
    return cursor.fetchall()


def mark_memory_analyzed(conn: sqlite3.Connection, conversation_id: int) -> None:
    """Mark a conversation as successfully processed into long-term memory."""
    conn.execute(
        """
        UPDATE conversations
        SET memory_analyzed_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (conversation_id,),
    )
    conn.commit()
