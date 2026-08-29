import asyncio
from dataclasses import dataclass

import pytest
from pydantic_ai_harness.memory import InMemoryStore, Memory

from buddy.game_memory import ensure_game_document
from buddy.game_memory_analysis import (
    GameMemoryAnalysisResult,
    GameProfileUpdate,
    update_game_profiles,
)
from buddy.long_term_memory import MEMORY_STORE_PATH


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
                session_context="User: W Elden Ring gram magiem.",
                user_evidence_context="User: W Elden Ring gram magiem.",
            )

        return await memory.store.read(MEMORY_STORE_PATH, max_chars=10_000)

    assert asyncio.run(scenario()) is None
