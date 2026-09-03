import asyncio
from dataclasses import dataclass

import pytest
from pydantic_ai_harness.memory import InMemoryStore, Memory

from buddy.memory.game import ensure_game_document
from buddy.memory.game_analysis import (
    GameMemoryAnalysisResult,
    GameProfileUpdate,
    update_game_profiles,
)
from buddy.memory.long_term import MEMORY_STORE_PATH
from buddy.prompts import GAME_MEMORY_ANALYZER_INSTRUCTIONS


@dataclass
class StubRunResult:
    output: GameMemoryAnalysisResult


class StubAnalyzer:
    async def run(self, _prompt: str) -> StubRunResult:
        return StubRunResult(
            output=GameMemoryAnalysisResult(
                updates=[
                    GameProfileUpdate(
                        path=MEMORY_STORE_PATH,
                        changed=True,
                        memory_markdown="# Invalid overwrite",
                    )
                ]
            )
        )


class CapturingUnchangedAnalyzer:
    def __init__(self):
        self.prompts: list[str] = []

    async def run(self, prompt: str) -> StubRunResult:
        self.prompts.append(prompt)
        return StubRunResult(output=GameMemoryAnalysisResult(updates=[]))


def test_game_analyzer_cannot_update_a_path_outside_detected_profiles(db):
    """Model output is restricted to the exact game documents supplied to it."""
    async def scenario():
        memory = Memory(InMemoryStore())
        game = await ensure_game_document(db, memory, "Elden Ring")

        with pytest.raises(ValueError, match="unknown game path"):
            await update_game_profiles(
                db,
                StubAnalyzer(),
                memory,
                game_documents=[game],
                user_evidence_context="User: W Elden Ring gram magiem.",
            )

        return await memory.store.read(MEMORY_STORE_PATH, max_chars=10_000)

    assert asyncio.run(scenario()) is None


def test_game_profile_analyzer_receives_only_user_evidence(db):
    """Assistant guesses and tool output cannot become new game-profile facts."""
    async def scenario():
        memory = Memory(InMemoryStore())
        game = await ensure_game_document(db, memory, "Elden Ring")
        analyzer = CapturingUnchangedAnalyzer()

        await update_game_profiles(
            db,
            analyzer,
            memory,
            game_documents=[game],
            user_evidence_context=(
                "User: Znalazłam zamknięte drzwi i nie mam klucza.\n"
                "Assistant: To na pewno zagadka z trzema żółwiami.\n"
                "Tool: Drzwi nie wymagają klucza."
            ),
        )
        return analyzer.prompts[0]

    prompt = asyncio.run(scenario())

    assert "Znalazłam zamknięte drzwi" in prompt
    assert "trzema żółwiami" not in prompt
    assert "Drzwi nie wymagają klucza" not in prompt


def test_distinct_game_titles_do_not_share_a_colliding_slug(db):
    async def scenario():
        memory = Memory(InMemoryStore())
        slash_title = await ensure_game_document(db, memory, "A/B")
        spaced_title = await ensure_game_document(db, memory, "A B")
        return slash_title, spaced_title

    slash_title, spaced_title = asyncio.run(scenario())

    assert slash_title.path == "main/games/a-b.md"
    assert spaced_title.path.startswith("main/games/a-b-")
    assert spaced_title.path.endswith(".md")
    assert slash_title.path != spaced_title.path
    assert slash_title.title == "A/B"
    assert spaced_title.title == "A B"


def test_game_memory_instructions_do_not_promote_video_behavior_to_preference():
    assert "nie dowodzi preferencji" in GAME_MEMORY_ANALYZER_INSTRUCTIONS
    assert "Preferencje zapisuj wyłącznie" in GAME_MEMORY_ANALYZER_INSTRUCTIONS
    assert "Nie wzmacniaj znaczenia dowodu" in GAME_MEMORY_ANALYZER_INSTRUCTIONS
