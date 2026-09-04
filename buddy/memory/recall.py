"""Bounded retrieval of game profiles and related session memories."""

import re
import sqlite3
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from typing import Awaitable, Callable, cast

from pydantic import BaseModel, Field
from pydantic_ai import RunContext, RunUsage
from pydantic_ai_harness.memory import Memory

from buddy.config import MEMORY, VIDEO
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
from buddy.video.models import StoredEpisodeObservation, StoredEpisodeSummary
from buddy.video.storage import (
    get_episode_observations,
    get_recent_episode_summaries,
    get_video_segment,
    search_episode_observations,
)


class MemoryRecallType(str, Enum):
    """Document scopes exposed to the chat agent."""

    ALL = "all"
    GAME = "game"
    SESSION = "session"
    VIDEO = "video"


class RecalledMemory(BaseModel):
    """One bounded memory document returned to the model."""

    path: str
    document_type: str
    title: str
    occurred_at: str | None
    games: list[str]
    summary: str
    content: str
    truncated: bool
    sources: list[str] = Field(default_factory=list)


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
        query_en: str | None = None,
        literal_terms: list[str] | None = None,
        game: str | None = None,
        memory_type: MemoryRecallType = MemoryRecallType.ALL,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 3,
    ) -> MemoryRecallResult:
        """Return bounded memory documents and episode evidence."""
        if not 1 <= limit <= MEMORY.recall_results:
            raise ValueError(
                f"Recall limit must be between 1 and {MEMORY.recall_results}."
            )
        occurred_from, occurred_before = _parse_date_range(date_from, date_to)
        normalized_query = query_en.strip() if query_en and query_en.strip() else None
        normalized_literals = _normalize_literal_terms(literal_terms)
        search_queries = list(
            dict.fromkeys(
                query
                for query in [normalized_query, *normalized_literals]
                if query is not None
            )
        )
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
        video_matches: list[RecalledMemory] = []
        if remaining > 0 and memory_type in (
            MemoryRecallType.ALL,
            MemoryRecallType.VIDEO,
        ):
            if search_queries:
                observations: list[StoredEpisodeObservation] = []
                seen_segments: set[int] = set()
                for search_query in search_queries:
                    found = search_episode_observations(
                        self._conn,
                        query=search_query,
                        game_document_id=(
                            game_document.id if game_document is not None else None
                        ),
                        occurred_from=occurred_from,
                        occurred_before=occurred_before,
                        limit=remaining,
                    )
                    for observation in found:
                        if observation.segment_id in seen_segments:
                            continue
                        seen_segments.add(observation.segment_id)
                        observations.append(observation)
                        if len(observations) >= remaining:
                            break
                    if len(observations) >= remaining:
                        break
                video_matches = self._expand_candidate_episode_timelines(
                    observations,
                    limit=remaining,
                )
            if not video_matches and game_document is not None:
                summaries = get_recent_episode_summaries(
                    self._conn,
                    game_document_id=game_document.id,
                    occurred_from=occurred_from,
                    occurred_before=occurred_before,
                    limit=remaining,
                )
                video_matches = [
                    self._read_episode_summary(summary) for summary in summaries
                ]
                used_recent_fallback = bool(search_queries and summaries)
            elif not search_queries:
                summaries = get_recent_episode_summaries(
                    self._conn,
                    occurred_from=occurred_from,
                    occurred_before=occurred_before,
                    limit=remaining,
                )
                video_matches = [
                    self._read_episode_summary(summary) for summary in summaries
                ]
            remaining -= len(video_matches)
        if remaining > 0 and memory_type in (
            MemoryRecallType.ALL,
            MemoryRecallType.SESSION,
        ):
            sessions: list[MemoryDocument] = []
            seen_documents: set[int] = set()
            for search_query in search_queries or [None]:
                found = search_memory_documents(
                    self._conn,
                    query=search_query,
                    document_type=MemoryDocumentType.SESSION,
                    game_document_id=(
                        game_document.id if game_document is not None else None
                    ),
                    occurred_from=occurred_from,
                    occurred_before=occurred_before,
                    limit=remaining,
                )
                for document in found:
                    if document.id in seen_documents:
                        continue
                    seen_documents.add(document.id)
                    sessions.append(document)
                    if len(sessions) >= remaining:
                        break
                if len(sessions) >= remaining:
                    break
            if (
                search_queries
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

        memory_matches = [
            await self._read_document(
                document,
                normalized_query or (normalized_literals[0] if normalized_literals else None),
            )
            for document in documents
        ]
        matches = [*memory_matches, *video_matches]
        if not matches:
            return MemoryRecallResult(
                message="No memory documents matched the supplied filters.",
                matches=[],
            )
        message = f"Found {len(matches)} bounded memory/video source(s)."
        if used_recent_fallback:
            message += (
                " No exact text match was found; returning recent evidence "
                "for the selected game."
            )
        return MemoryRecallResult(
            message=message,
            used_recent_fallback=used_recent_fallback,
            matches=matches,
        )

    def _expand_candidate_episode_timelines(
        self,
        matches: list[StoredEpisodeObservation],
        *,
        limit: int,
    ) -> list[RecalledMemory]:
        """Use matching observations to select bounded complete episode timelines."""
        candidates: list[RecalledMemory] = []
        seen_segments: set[int] = set()
        for match in matches:
            if match.segment_id in seen_segments:
                continue
            seen_segments.add(match.segment_id)
            candidates.append(self._read_episode_timeline(match))
            if len(candidates) >= limit:
                break
        return candidates

    def _read_episode_timeline(
        self,
        matched_observation: StoredEpisodeObservation,
    ) -> RecalledMemory:
        segment = get_video_segment(self._conn, matched_observation.segment_id)
        if segment is None:
            raise RuntimeError("A searched video observation lost its segment.")
        game_title = self._game_title(segment.game_document_id)
        observations = get_episode_observations(self._conn, segment.id)
        content, truncated, included_sources = self._render_bounded_episode_timeline(
            observations,
            max_chars=VIDEO.recall_episode_chars,
        )
        sources = [segment.source_uri]
        sources.extend(included_sources)
        return RecalledMemory(
            path=segment.source_uri,
            document_type="episode_timeline",
            title=f"Episode {segment.id} timeline",
            occurred_at=segment.started_at.isoformat(),
            games=[game_title] if game_title else [],
            summary=matched_observation.description,
            content=content,
            truncated=truncated,
            sources=list(dict.fromkeys(sources)),
        )

    @staticmethod
    def _render_video_observation(observation: StoredEpisodeObservation) -> str:
        details = [
            f"Source: {observation.source_uri}",
            f"Activity: {observation.activity}",
            f"Location: {observation.location or '(unknown)'}",
            f"Description: {observation.description}",
        ]
        if observation.visible_objects:
            details.append("Visible objects: " + ", ".join(observation.visible_objects))
        if observation.visible_text:
            details.append("Visible text: " + ", ".join(observation.visible_text))
        if observation.uncertainties:
            details.append("Uncertainties: " + ", ".join(observation.uncertainties))
        return "\n".join(details)

    @classmethod
    def _render_bounded_episode_timeline(
        cls,
        observations: list[StoredEpisodeObservation],
        *,
        max_chars: int,
    ) -> tuple[str, bool, list[str]]:
        """Render only complete source-backed blocks that fit the recall bound."""
        blocks: list[str] = []
        sources: list[str] = []
        used_chars = 0
        for observation in observations:
            block = cls._render_video_observation(observation)
            separator_chars = 2 if blocks else 0
            if used_chars + separator_chars + len(block) > max_chars:
                if not blocks:
                    block, _ = _bounded_text(block, max_chars)
                    blocks.append(block)
                    sources.append(observation.source_uri)
                return "\n\n".join(blocks), True, sources
            blocks.append(block)
            sources.append(observation.source_uri)
            used_chars += separator_chars + len(block)
        return "\n\n".join(blocks), False, sources

    @staticmethod
    def _read_episode_summary(summary: StoredEpisodeSummary) -> RecalledMemory:
        return RecalledMemory(
            path=summary.source_uri,
            document_type="episode_summary",
            title=f"Episode {summary.segment_id}",
            occurred_at=summary.started_at.isoformat(),
            games=[summary.game_title] if summary.game_title else [],
            summary=summary.summary,
            content=summary.summary,
            truncated=False,
            sources=[summary.source_uri],
        )

    def _game_title(self, game_document_id: int | None) -> str | None:
        if game_document_id is None:
            return None
        row = self._conn.execute(
            """
            SELECT title FROM memory_documents
            WHERE id = ? AND document_type = 'game'
            """,
            (game_document_id,),
        ).fetchone()
        return row["title"] if row is not None else None

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
            sources=[document.path],
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
        query_en: str | None = None,
        literal_terms: list[str] | None = None,
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
            query_en=query_en,
            literal_terms=literal_terms,
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
            allowed_sources={
                source
                for match in recalled.matches
                for source in (match.sources or [match.path])
            },
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
        query_en: str | None = None,
        literal_terms: list[str] | None = None,
        game: str | None = None,
        memory_type: MemoryRecallType = MemoryRecallType.ALL,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 3,
    ) -> ContextButlerResult:
        """Recall and reduce prior game, session, or video memory for one question.

        `question` is the user's complete information need in its original
        language. Translate short distinctive search terms into canonical English
        and pass them in `query_en`. Put exact original-language OCR, labels, and
        proper names in `literal_terms`; do not translate those values. `game`
        should be a title explicitly known from the conversation. Use
        `memory_type="video"` for earlier gameplay
        footage; exact observations are preferred and recent episode summaries
        are the bounded fallback for a known game. Dates use YYYY-MM-DD and are
        inclusive. Only compact, source-validated evidence is returned.
        """
        return await coordinator.recall(
            question=question,
            query_en=query_en,
            literal_terms=literal_terms,
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


def _normalize_literal_terms(literal_terms: list[str] | None) -> list[str]:
    if literal_terms is None:
        return []
    if len(literal_terms) > MEMORY.recall_literal_terms:
        raise ValueError(
            f"literal_terms accepts at most {MEMORY.recall_literal_terms} values."
        )
    normalized: list[str] = []
    seen: set[str] = set()
    for term in literal_terms:
        clean = " ".join(term.split()).strip()
        if not clean:
            continue
        if len(clean) > MEMORY.recall_literal_term_chars:
            raise ValueError(
                "A literal search term cannot exceed "
                f"{MEMORY.recall_literal_term_chars} characters."
            )
        key = clean.casefold()
        if key not in seen:
            seen.add(key)
            normalized.append(clean)
    return normalized


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


def _bounded_text(content: str, max_chars: int) -> tuple[str, bool]:
    """Bound a retrieval payload and mark that it no longer contains all data."""
    if len(content) <= max_chars:
        return content, False
    return content[: max_chars - 1].rstrip() + "…", True
