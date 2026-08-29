"""Bounded retrieval of game profiles and related session memories."""

import re
import sqlite3
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from typing import Awaitable, Callable, cast

from pydantic import BaseModel
from pydantic_ai import RunContext, RunUsage
from pydantic_ai_harness.memory import Memory

from buddy.config import MEMORY
from buddy.memory.context_butler import (
    ContextButler,
    ContextButlerResult,
    build_context_butler,
    build_memory_context_prompt,
    validate_butler_sources,
)
from buddy.memory.game import resolve_game_memory_path
from buddy.memory.index import (
    MemoryDocument,
    MemoryDocumentType,
    get_memory_document,
    get_session_game_documents,
    search_memory_documents,
)


class MemoryRecallType(str, Enum):
    """Document scopes exposed to the chat agent."""

    ALL = "all"
    GAME = "game"
    SESSION = "session"


class RecalledMemory(BaseModel):
    """One bounded memory document returned to the model."""

    path: str
    document_type: MemoryDocumentType
    title: str
    occurred_at: str | None
    games: list[str]
    summary: str
    content: str
    truncated: bool


class MemoryRecallResult(BaseModel):
    """Structured result with explicit retrieval diagnostics."""

    message: str
    used_recent_fallback: bool = False
    matches: list[RecalledMemory]


class MemoryRecallService:
    """Search indexed metadata and read only the selected Harness documents."""

    def __init__(self, conn: sqlite3.Connection, memory: Memory):
        self._conn = conn
        self._memory = memory

    async def recall(
        self,
        *,
        query: str | None = None,
        game: str | None = None,
        memory_type: MemoryRecallType = MemoryRecallType.ALL,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 3,
    ) -> MemoryRecallResult:
        """Return a bounded profile/session set matching explicit filters."""
        if not 1 <= limit <= MEMORY.recall_results:
            raise ValueError(
                f"Recall limit must be between 1 and {MEMORY.recall_results}."
            )
        occurred_from, occurred_before = _parse_date_range(date_from, date_to)
        normalized_query = query.strip() if query and query.strip() else None
        normalized_game = game.strip() if game and game.strip() else None

        game_document = None
        if normalized_game is not None:
            game_document, error = self._resolve_game(normalized_game)
            if error is not None:
                return MemoryRecallResult(message=error, matches=[])

        documents: list[MemoryDocument] = []
        if (
            game_document is not None
            and memory_type in (MemoryRecallType.ALL, MemoryRecallType.GAME)
            and occurred_from is None
            and occurred_before is None
        ):
            documents.append(game_document)

        remaining = limit - len(documents)
        used_recent_fallback = False
        if remaining > 0 and memory_type in (
            MemoryRecallType.ALL,
            MemoryRecallType.SESSION,
        ):
            sessions = search_memory_documents(
                self._conn,
                query=normalized_query,
                document_type=MemoryDocumentType.SESSION,
                game_document_id=(
                    game_document.id if game_document is not None else None
                ),
                occurred_from=occurred_from,
                occurred_before=occurred_before,
                limit=remaining,
            )
            if (
                normalized_query is not None
                and not sessions
                and game_document is not None
            ):
                sessions = search_memory_documents(
                    self._conn,
                    document_type=MemoryDocumentType.SESSION,
                    game_document_id=game_document.id,
                    occurred_from=occurred_from,
                    occurred_before=occurred_before,
                    limit=remaining,
                )
                used_recent_fallback = bool(sessions)
            documents.extend(sessions)

        if (
            remaining > 0
            and game_document is None
            and memory_type is MemoryRecallType.GAME
        ):
            documents.extend(
                search_memory_documents(
                    self._conn,
                    query=normalized_query,
                    document_type=MemoryDocumentType.GAME,
                    limit=remaining,
                )
            )

        matches = [
            await self._read_document(document, normalized_query)
            for document in documents
        ]
        if not matches:
            return MemoryRecallResult(
                message="No memory documents matched the supplied filters.",
                matches=[],
            )
        message = f"Found {len(matches)} bounded memory document(s)."
        if used_recent_fallback:
            message += (
                " No exact text match was found; returning the most recent "
                "sessions for the selected game."
            )
        return MemoryRecallResult(
            message=message,
            used_recent_fallback=used_recent_fallback,
            matches=matches,
        )

    def _resolve_game(
        self,
        game: str,
    ) -> tuple[MemoryDocument | None, str | None]:
        """Resolve an exact stable path, then a unique indexed title match."""
        exact = get_memory_document(
            self._conn,
            resolve_game_memory_path(self._conn, game),
        )
        if exact is not None and exact.document_type is MemoryDocumentType.GAME:
            return exact, None

        candidates = search_memory_documents(
            self._conn,
            query=game,
            document_type=MemoryDocumentType.GAME,
            limit=MEMORY.recall_results,
        )
        if len(candidates) == 1:
            return candidates[0], None
        if not candidates:
            return None, f"No indexed game matched {game!r}."
        titles = ", ".join(candidate.title for candidate in candidates)
        return None, f"Game name {game!r} is ambiguous. Candidates: {titles}."

    async def _read_document(
        self,
        document: MemoryDocument,
        query: str | None,
    ) -> RecalledMemory:
        """Read a single selected document with a strict character bound."""
        games = (
            [
                game.title
                for game in get_session_game_documents(self._conn, document.id)
            ]
            if document.document_type is MemoryDocumentType.SESSION
            else []
        )
        memory_file = await self._memory.store.read(
            document.path,
            max_chars=MEMORY.recall_source_chars,
        )
        if memory_file is None:
            content = "(Indexed document is missing from Harness Memory.)"
            truncated = False
        else:
            content, excerpted = _bounded_excerpt(memory_file.content, query)
            truncated = memory_file.truncated or excerpted
        return RecalledMemory(
            path=document.path,
            document_type=document.document_type,
            title=document.title,
            occurred_at=document.occurred_at,
            games=games,
            summary=document.summary,
            content=content,
            truncated=truncated,
        )


class MemoryRecallCoordinator:
    """Retrieve bounded documents and delegate their reduction to a cheap model."""

    def __init__(
        self,
        service: MemoryRecallService,
        context_butler: ContextButler,
    ):
        self._service = service
        self._context_butler = context_butler

    async def recall(
        self,
        *,
        question: str,
        query: str | None = None,
        game: str | None = None,
        memory_type: MemoryRecallType = MemoryRecallType.ALL,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 3,
        usage: RunUsage | None = None,
    ) -> ContextButlerResult:
        """Return only compact, source-validated evidence to the parent agent."""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty.")

        recalled = await self._service.recall(
            query=query,
            game=game,
            memory_type=memory_type,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )
        if not recalled.matches:
            return ContextButlerResult(
                enough_evidence=False,
                context="",
                sources=[],
                uncertainty=recalled.message,
            )

        prompt = build_memory_context_prompt(
            normalized_question,
            recalled.model_dump_json(indent=2),
        )
        analyzed = await self._context_butler.run(prompt, usage=usage)
        return validate_butler_sources(
            analyzed.output,
            allowed_sources={match.path for match in recalled.matches},
        )


RecallMemoryTool = Callable[..., Awaitable[ContextButlerResult]]


def create_recall_memory_tool(
    conn: sqlite3.Connection,
    memory: Memory,
    context_butler: ContextButler | None = None,
) -> RecallMemoryTool:
    """Create a Pydantic AI tool bound to this process's stores."""
    service = MemoryRecallService(conn, memory)
    coordinator = MemoryRecallCoordinator(
        service,
        context_butler or cast(ContextButler, build_context_butler()),
    )

    async def recall_memory(
        ctx: RunContext,
        question: str,
        query: str | None = None,
        game: str | None = None,
        memory_type: MemoryRecallType = MemoryRecallType.ALL,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 3,
    ) -> ContextButlerResult:
        """Recall and reduce prior game/session memory for one question.

        `question` is the user's complete information need. Use short distinctive
        keywords in `query`. `game` should be a title explicitly known from the
        conversation. Dates use YYYY-MM-DD and are inclusive. Only compact,
        source-validated evidence is returned to the parent agent.
        """
        return await coordinator.recall(
            question=question,
            query=query,
            game=game,
            memory_type=memory_type,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            usage=ctx.usage,
        )

    return recall_memory


def _parse_date_range(
    date_from: str | None,
    date_to: str | None,
) -> tuple[datetime | None, datetime | None]:
    """Convert inclusive ISO dates into a half-open UTC datetime interval."""
    start = _parse_date(date_from, "date_from") if date_from else None
    end = _parse_date(date_to, "date_to") if date_to else None
    if start is not None and end is not None and start > end:
        raise ValueError("date_from must not be later than date_to.")
    occurred_from = (
        datetime.combine(start, time.min, tzinfo=timezone.utc)
        if start is not None
        else None
    )
    occurred_before = (
        datetime.combine(end + timedelta(days=1), time.min, tzinfo=timezone.utc)
        if end is not None
        else None
    )
    return occurred_from, occurred_before


def _parse_date(value: str, field_name: str) -> date:
    """Parse a strict YYYY-MM-DD tool argument with a useful error."""
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must use YYYY-MM-DD format.") from error


def _bounded_excerpt(content: str, query: str | None) -> tuple[str, bool]:
    """Return a bounded excerpt around the first matching query term."""
    if len(content) <= MEMORY.recall_document_chars:
        return content, False

    match_index = 0
    if query is not None:
        folded_content = content.casefold()
        terms = re.findall(r"\w+", query.casefold(), flags=re.UNICODE)
        positions = [
            position
            for term in terms
            if (position := folded_content.find(term)) >= 0
        ]
        if positions:
            match_index = min(positions)

    start = max(0, match_index - 500)
    end = min(len(content), start + MEMORY.recall_document_chars)
    if end - start < MEMORY.recall_document_chars:
        start = max(0, end - MEMORY.recall_document_chars)
    excerpt = content[start:end]
    if start > 0:
        excerpt = "…" + excerpt[1:]
    if end < len(content):
        excerpt = excerpt[:-1] + "…"
    return excerpt, True
