"""Create stable game memory documents and link them to session reports."""

import re
import sqlite3
import unicodedata
from hashlib import sha256

from pydantic_ai_harness.memory import Memory

from buddy.config import MEMORY, STORAGE
from buddy.memory.index import (
    MemoryDocument,
    MemoryDocumentType,
    MemoryRelationType,
    get_memory_document,
    index_memory_document,
    link_memory_documents,
)


def normalize_game_name(name: str) -> str:
    """Normalize model output into a bounded, single-line display name."""
    normalized = " ".join(name.split()).strip()
    if not normalized:
        raise ValueError("A game name cannot be empty.")
    return normalized[: MEMORY.game_name_chars]


def game_memory_path(name: str) -> str:
    """Return a deterministic path for a normalized game title."""
    normalized = normalize_game_name(name)
    ascii_name = unicodedata.normalize("NFKD", normalized).encode(
        "ascii",
        "ignore",
    ).decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.casefold()).strip("-")
    if not slug:
        digest = sha256(normalized.casefold().encode()).hexdigest()[:12]
        slug = f"game-{digest}"
    return f"{STORAGE.game_memory_prefix}/{slug}.md"


def resolve_game_memory_path(conn: sqlite3.Connection, name: str) -> str:
    """Resolve a stable path without merging distinct titles on slug collision."""
    title = normalize_game_name(name)
    base_path = game_memory_path(title)
    existing = get_memory_document(conn, base_path)
    if existing is None or existing.title.casefold() == title.casefold():
        return base_path

    digest = sha256(title.casefold().encode()).hexdigest()[:12]
    stem = base_path.removesuffix(".md")
    return f"{stem}-{digest}.md"


async def ensure_game_document(
    conn: sqlite3.Connection,
    memory: Memory,
    name: str,
) -> MemoryDocument:
    """Create a minimal game profile once and ensure it remains indexed."""
    title = normalize_game_name(name)
    path = resolve_game_memory_path(conn, title)
    memory_file = await memory.store.read(path, max_chars=MEMORY.game_document_chars)
    if memory_file is None:
        content = f"# {title}\n"
        await memory.store.write(path, content, expected_version=None)
    else:
        content = memory_file.content

    return index_memory_document(
        conn,
        document_type=MemoryDocumentType.GAME,
        path=path,
        title=title,
        summary=f"Pamięć dotycząca gry {title}.",
        body=content,
    )


async def link_session_to_games(
    conn: sqlite3.Connection,
    memory: Memory,
    *,
    session_document: MemoryDocument,
    game_names: list[str],
) -> list[MemoryDocument]:
    """Ensure distinct game profiles and link one session to all of them."""
    linked_games: list[MemoryDocument] = []
    seen_paths: set[str] = set()
    for name in game_names:
        game_document = await ensure_game_document(conn, memory, name)
        if game_document.path in seen_paths:
            continue
        seen_paths.add(game_document.path)
        link_memory_documents(
            conn,
            source_document_id=session_document.id,
            relation_type=MemoryRelationType.BELONGS_TO_GAME,
            target_document_id=game_document.id,
        )
        linked_games.append(game_document)
    return linked_games
