"""Bounded SQLite indexing and relationships for persisted memory documents."""

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath

from buddy.config import MEMORY


class MemoryDocumentType(str, Enum):
    """Kinds of memory documents currently supported by the application."""

    GLOBAL = "global"
    GAME = "game"
    SESSION = "session"


class MemoryRelationType(str, Enum):
    """Explicit relationships between indexed memory documents."""

    BELONGS_TO_GAME = "belongs_to_game"


@dataclass(frozen=True, slots=True)
class MemoryDocument:
    """Searchable metadata pointing to a document stored in Harness Memory."""

    id: int
    document_type: MemoryDocumentType
    path: str
    title: str
    summary: str
    conversation_id: int | None
    occurred_at: str | None


def init_memory_index(conn: sqlite3.Connection) -> None:
    """Create document metadata, relationships, and their full-text index."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_type TEXT NOT NULL,
            path TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            summary TEXT NOT NULL DEFAULT '',
            conversation_id INTEGER,
            occurred_at TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id)
                REFERENCES conversations (id) ON DELETE SET NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_document_id INTEGER NOT NULL,
            relation_type TEXT NOT NULL,
            target_document_id INTEGER NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_document_id)
                REFERENCES memory_documents (id) ON DELETE CASCADE,
            FOREIGN KEY (target_document_id)
                REFERENCES memory_documents (id) ON DELETE CASCADE,
            UNIQUE (source_document_id, relation_type, target_document_id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memory_documents_type_occurred
        ON memory_documents (document_type, occurred_at)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memory_documents_conversation
        ON memory_documents (conversation_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memory_relations_target
        ON memory_relations (target_document_id, relation_type)
        """
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_search USING fts5(
            document_id UNINDEXED,
            title,
            summary,
            body,
            tokenize = 'unicode61 remove_diacritics 2'
        )
        """
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_search_en USING fts5(
            document_id UNINDEXED,
            title,
            summary,
            body,
            tokenize = 'porter unicode61 remove_diacritics 2'
        )
        """
    )
    conn.execute(
        """
        INSERT INTO memory_search_en (document_id, title, summary, body)
        SELECT legacy.document_id, legacy.title, legacy.summary, legacy.body
        FROM memory_search AS legacy
        WHERE NOT EXISTS (
            SELECT 1 FROM memory_search_en AS canonical
            WHERE canonical.document_id = legacy.document_id
        )
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS memory_documents_after_delete
        AFTER DELETE ON memory_documents
        BEGIN
            DELETE FROM memory_search WHERE document_id = OLD.id;
            DELETE FROM memory_search_en WHERE document_id = OLD.id;
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS memory_documents_after_delete_en
        AFTER DELETE ON memory_documents
        BEGIN
            DELETE FROM memory_search_en WHERE document_id = OLD.id;
        END
        """
    )


def index_memory_document(
    conn: sqlite3.Connection,
    *,
    document_type: MemoryDocumentType,
    path: str,
    title: str,
    summary: str,
    body: str,
    conversation_id: int | None = None,
    occurred_at: datetime | None = None,
) -> MemoryDocument:
    """Insert or refresh a document and its searchable content atomically."""
    _validate_memory_path(path)
    if not title.strip():
        raise ValueError("A memory document title cannot be empty.")
    normalized_occurred_at = _normalize_timestamp(occurred_at)

    with conn:
        existing = conn.execute(
            "SELECT document_type FROM memory_documents WHERE path = ?",
            (path,),
        ).fetchone()
        if (
            existing is not None
            and MemoryDocumentType(existing["document_type"]) is not document_type
        ):
            raise ValueError("An indexed memory document cannot change its type.")

        conn.execute(
            """
            INSERT INTO memory_documents (
                document_type,
                path,
                title,
                summary,
                conversation_id,
                occurred_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                document_type = excluded.document_type,
                title = excluded.title,
                summary = excluded.summary,
                conversation_id = excluded.conversation_id,
                occurred_at = excluded.occurred_at,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                document_type.value,
                path,
                title.strip(),
                summary.strip(),
                conversation_id,
                normalized_occurred_at,
            ),
        )
        row = conn.execute(
            "SELECT * FROM memory_documents WHERE path = ?",
            (path,),
        ).fetchone()
        if row is None:
            raise RuntimeError("The indexed memory document could not be reloaded.")

        document = _document_from_row(row)
        conn.execute(
            "DELETE FROM memory_search WHERE document_id = ?",
            (document.id,),
        )
        conn.execute(
            "DELETE FROM memory_search_en WHERE document_id = ?",
            (document.id,),
        )
        conn.execute(
            """
            INSERT INTO memory_search (document_id, title, summary, body)
            VALUES (?, ?, ?, ?)
            """,
            (document.id, document.title, document.summary, body),
        )
        conn.execute(
            """
            INSERT INTO memory_search_en (document_id, title, summary, body)
            VALUES (?, ?, ?, ?)
            """,
            (document.id, document.title, document.summary, body),
        )

    return document


def delete_memory_document(conn: sqlite3.Connection, path: str) -> bool:
    """Delete document metadata, its relations, and full-text index entry."""
    with conn:
        cursor = conn.execute(
            "DELETE FROM memory_documents WHERE path = ?",
            (path,),
        )
    return cursor.rowcount > 0


def get_memory_document(
    conn: sqlite3.Connection,
    path: str,
) -> MemoryDocument | None:
    """Return metadata for one exact Harness Memory document path."""
    row = conn.execute(
        "SELECT * FROM memory_documents WHERE path = ?",
        (path,),
    ).fetchone()
    return _document_from_row(row) if row is not None else None


def get_session_game_documents(
    conn: sqlite3.Connection,
    session_document_id: int,
) -> list[MemoryDocument]:
    """Return game profiles directly linked to one session document."""
    rows = conn.execute(
        """
        SELECT games.*
        FROM memory_relations AS relations
        JOIN memory_documents AS games
          ON games.id = relations.target_document_id
        WHERE relations.source_document_id = ?
          AND relations.relation_type = ?
          AND games.document_type = ?
        ORDER BY games.title, games.id
        """,
        (
            session_document_id,
            MemoryRelationType.BELONGS_TO_GAME.value,
            MemoryDocumentType.GAME.value,
        ),
    ).fetchall()
    return [_document_from_row(row) for row in rows]


def link_memory_documents(
    conn: sqlite3.Connection,
    *,
    source_document_id: int,
    relation_type: MemoryRelationType,
    target_document_id: int,
) -> None:
    """Create an idempotent, validated relationship between documents."""
    if relation_type is MemoryRelationType.BELONGS_TO_GAME:
        _validate_session_game_relation(
            conn,
            source_document_id=source_document_id,
            target_document_id=target_document_id,
        )

    with conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO memory_relations (
                source_document_id,
                relation_type,
                target_document_id
            )
            VALUES (?, ?, ?)
            """,
            (source_document_id, relation_type.value, target_document_id),
        )


def search_memory_documents(
    conn: sqlite3.Connection,
    *,
    query: str | None = None,
    document_type: MemoryDocumentType | None = None,
    game_document_id: int | None = None,
    occurred_from: datetime | None = None,
    occurred_before: datetime | None = None,
    limit: int = 5,
) -> list[MemoryDocument]:
    """Find bounded documents by text, type, game relation, and date."""
    if not 1 <= limit <= MEMORY.search_results:
        raise ValueError(
            f"The memory search limit must be between 1 and "
            f"{MEMORY.search_results}."
        )

    normalized_from = _normalize_timestamp(occurred_from)
    normalized_before = _normalize_timestamp(occurred_before)
    if (
        normalized_from is not None
        and normalized_before is not None
        and normalized_from >= normalized_before
    ):
        raise ValueError("occurred_from must be earlier than occurred_before.")

    conditions: list[str] = []
    params: list[str | int] = []
    join = ""

    if query is not None:
        normalized_query = _build_fts_query(query)
        if normalized_query is None:
            return []
        join = "JOIN memory_search_en ON memory_search_en.document_id = documents.id"
        conditions.append("memory_search_en MATCH ?")
        params.append(normalized_query)

    if document_type is not None:
        conditions.append("documents.document_type = ?")
        params.append(document_type.value)

    if game_document_id is not None:
        conditions.append(
            """
            EXISTS (
                SELECT 1
                FROM memory_relations
                WHERE source_document_id = documents.id
                  AND relation_type = ?
                  AND target_document_id = ?
            )
            """
        )
        params.extend(
            [MemoryRelationType.BELONGS_TO_GAME.value, game_document_id]
        )

    if normalized_from is not None:
        conditions.append("documents.occurred_at >= ?")
        params.append(normalized_from)

    if normalized_before is not None:
        conditions.append("documents.occurred_at < ?")
        params.append(normalized_before)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    order = (
        "ORDER BY bm25(memory_search_en, 0.0, 5.0, 3.0, 1.0), "
        "documents.occurred_at DESC, documents.id DESC"
        if query is not None
        else "ORDER BY documents.occurred_at DESC, documents.id DESC"
    )
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT documents.*
        FROM memory_documents AS documents
        {join}
        {where}
        {order}
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [_document_from_row(row) for row in rows]


def _document_from_row(row: sqlite3.Row) -> MemoryDocument:
    """Convert a SQLite row into a typed document metadata record."""
    return MemoryDocument(
        id=row["id"],
        document_type=MemoryDocumentType(row["document_type"]),
        path=row["path"],
        title=row["title"],
        summary=row["summary"],
        conversation_id=row["conversation_id"],
        occurred_at=row["occurred_at"],
    )


def _validate_memory_path(path: str) -> None:
    """Reject paths outside the application's logical main memory scope."""
    parsed = PurePosixPath(path)
    if (
        "\\" in path
        or parsed.is_absolute()
        or ".." in parsed.parts
        or len(parsed.parts) < 2
        or parsed.parts[0] != "main"
        or parsed.suffix.lower() != ".md"
    ):
        raise ValueError("Memory documents must stay under main/ and end in .md.")


def _validate_session_game_relation(
    conn: sqlite3.Connection,
    *,
    source_document_id: int,
    target_document_id: int,
) -> None:
    """Ensure belongs_to_game links only connect sessions to games."""
    rows = conn.execute(
        """
        SELECT id, document_type
        FROM memory_documents
        WHERE id IN (?, ?)
        """,
        (source_document_id, target_document_id),
    ).fetchall()
    types_by_id = {
        row["id"]: MemoryDocumentType(row["document_type"])
        for row in rows
    }
    if (
        types_by_id.get(source_document_id) is not MemoryDocumentType.SESSION
        or types_by_id.get(target_document_id) is not MemoryDocumentType.GAME
    ):
        raise ValueError("A belongs_to_game relation must connect a session to a game.")


def _build_fts_query(query: str) -> str | None:
    """Turn arbitrary user text into safe literal FTS5 search terms."""
    terms = re.findall(r"\w+", query, flags=re.UNICODE)
    if not terms:
        return None
    return " AND ".join(f'"{term}"' for term in terms[:16])


def _normalize_timestamp(value: datetime | None) -> str | None:
    """Store comparable ISO timestamps, converting aware values to naive UTC."""
    if value is None:
        return None
    if not isinstance(value, datetime):
        raise TypeError("Memory timestamps must be datetime values.")
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    timespec = "microseconds" if value.microsecond else "seconds"
    return value.isoformat(sep=" ", timespec=timespec)
