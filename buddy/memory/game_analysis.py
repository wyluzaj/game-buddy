"""Curate game-specific long-term profiles from completed session reports."""

from dataclasses import dataclass
import re
import sqlite3
from typing import Protocol

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model as PydanticModel
from pydantic_ai_harness.memory import Memory

from buddy.config import MEMORY, MODELS, build_model
from buddy.memory.index import (
    MemoryDocument,
    MemoryDocumentType,
    index_memory_document,
)
from buddy.memory.compaction import extract_user_evidence
from buddy.prompts import (
    GAME_MEMORY_ANALYSIS_PROMPT,
    GAME_MEMORY_ANALYZER_INSTRUCTIONS,
)


class GameProfileUpdate(BaseModel):
    """One model-proposed replacement of an existing game profile."""

    path: str
    changed: bool
    memory_markdown: str


class GameMemoryAnalysisResult(BaseModel):
    """Bounded profile updates returned from one completed session."""

    updates: list[GameProfileUpdate] = Field(
        default_factory=list,
        max_length=MEMORY.game_profile_updates,
    )


class GameMemoryAnalyzerRunResult(Protocol):
    """Minimal result interface required from a game-memory analyzer."""

    @property
    def output(self) -> GameMemoryAnalysisResult: ...


class GameMemoryAnalyzer(Protocol):
    """Minimal async analyzer interface used by profile processing."""

    async def run(self, prompt: str) -> GameMemoryAnalyzerRunResult: ...


@dataclass(frozen=True, slots=True)
class CurrentGameProfile:
    """An allowed profile together with its current versioned memory file."""

    document: MemoryDocument
    content: str
    version: str


def build_game_memory_analyzer(
    model: str | PydanticModel | None = None,
) -> Agent[None, GameMemoryAnalysisResult]:
    """Build the dedicated game curator using long-term-memory fallback config."""
    analyzer_model = model if model is not None else build_model(MODELS.long_term_memory)
    return Agent[None, GameMemoryAnalysisResult](
        model=analyzer_model,
        deps_type=type(None),
        instructions=GAME_MEMORY_ANALYZER_INSTRUCTIONS,
        output_type=GameMemoryAnalysisResult,
    )


async def update_game_profiles(
    conn: sqlite3.Connection,
    analyzer: GameMemoryAnalyzer,
    memory: Memory,
    *,
    game_documents: list[MemoryDocument],
    user_evidence_context: str,
    video_evidence_context: str = "",
) -> int:
    """Merge one session into only its allowed game profiles."""
    if not game_documents:
        return 0

    profiles = await _load_current_profiles(memory, game_documents)
    prompt = _build_game_memory_prompt(
        profiles,
        user_evidence_context=user_evidence_context,
        video_evidence_context=video_evidence_context,
    )
    result = await analyzer.run(prompt)
    updates_by_path = _validate_updates(result.output.updates, profiles)

    changed = 0
    for path, update in updates_by_path.items():
        if not update.changed:
            continue
        profile = profiles[path]
        content = update.memory_markdown.strip()
        if not content:
            raise ValueError("A changed game profile cannot be empty.")
        expected_heading = f"# {profile.document.title}"
        if content.splitlines()[0].strip() != expected_heading:
            raise ValueError(
                f"Game profile {path!r} must start with {expected_heading!r}."
            )
        if content == profile.content.strip():
            continue

        await memory.store.write(
            path,
            content + "\n",
            expected_version=profile.version,
        )
        index_memory_document(
            conn,
            document_type=MemoryDocumentType.GAME,
            path=path,
            title=profile.document.title,
            summary=_index_summary(content),
            body=content,
        )
        changed += 1
    return changed


async def _load_current_profiles(
    memory: Memory,
    game_documents: list[MemoryDocument],
) -> dict[str, CurrentGameProfile]:
    """Read complete contents for the bounded set of detected games."""
    profiles: dict[str, CurrentGameProfile] = {}
    for document in game_documents:
        memory_file = await memory.store.read(
            document.path,
            max_chars=MEMORY.game_document_chars,
        )
        if memory_file is None:
            raise RuntimeError(f"Game profile {document.path!r} is missing.")
        if memory_file.truncated:
            raise ValueError(f"Game profile {document.path!r} is too large to update.")
        profiles[document.path] = CurrentGameProfile(
            document=document,
            content=memory_file.content,
            version=memory_file.version,
        )
    return profiles


def _build_game_memory_prompt(
    profiles: dict[str, CurrentGameProfile],
    *,
    user_evidence_context: str,
    video_evidence_context: str = "",
) -> str:
    """Build bounded, clearly delimited input for game-profile consolidation."""
    current_profiles = "\n\n".join(
        (
            f'<game_profile path="{path}">\n'
            f"{profile.content.strip()}\n"
            f"</game_profile>"
        )
        for path, profile in profiles.items()
    )
    user_statements = extract_user_evidence(user_evidence_context)
    return GAME_MEMORY_ANALYSIS_PROMPT.format(
        current_game_profiles=current_profiles,
        user_statements=user_statements,
        video_observations=video_evidence_context.strip(),
    )


def _validate_updates(
    updates: list[GameProfileUpdate],
    profiles: dict[str, CurrentGameProfile],
) -> dict[str, GameProfileUpdate]:
    """Reject unknown or duplicate model-selected profile paths."""
    validated: dict[str, GameProfileUpdate] = {}
    for update in updates:
        if update.path not in profiles:
            raise ValueError(f"Analyzer returned unknown game path {update.path!r}.")
        if update.path in validated:
            raise ValueError(f"Analyzer returned duplicate game path {update.path!r}.")
        validated[update.path] = update
    return validated


def _index_summary(content: str) -> str:
    """Create bounded display metadata while indexing the complete profile body."""
    summary = re.sub(r"\s+", " ", content).strip()
    if len(summary) <= MEMORY.game_index_summary_chars:
        return summary
    return summary[: MEMORY.game_index_summary_chars - 1].rstrip() + "…"
