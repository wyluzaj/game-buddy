"""SQLite persistence layer for conversations and messages."""
import sqlite3
from pathlib import Path


def _open_connection(db_path: str) -> sqlite3.Connection:
    """Make a new connection to the database and enable foreign keys."""
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = "data/chatbot.db") -> sqlite3.Connection:
    """Initialize the database and return a connection."""
    conn = _open_connection(db_path)

    conn.execute('''
                 CREATE TABLE IF NOT EXISTS conversations
                 (
                     id         INTEGER PRIMARY KEY AUTOINCREMENT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                 )
                 ''')

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

    conn.commit()
    return conn


def create_conversation(conn: sqlite3.Connection) -> int:
    """Create a new conversation and return its ID."""
    cursor = conn.execute("INSERT INTO conversations DEFAULT VALUES")
    conn.commit()
    return cursor.lastrowid


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
        "SELECT message_list FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (conversation_id,),
    )
    rows = cursor.fetchall()
    return [row["message_list"] for row in rows]
