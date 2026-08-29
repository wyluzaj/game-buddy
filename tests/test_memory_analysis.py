import asyncio
from dataclasses import dataclass

from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    UserPromptPart,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.memory import InMemoryStore, Memory

from buddy.config import LONG_TERM_MEMORY_MODEL, Model
from buddy.database import create_conversation, finish_conversation, save_messages
from buddy.game_memory import game_memory_path
from buddy.game_memory_analysis import (
    GameMemoryAnalysisResult,
    GameProfileUpdate,
)
from buddy.long_term_memory import MEMORY_STORE_PATH
from buddy.memory_analysis import (
    MemoryAnalysisResult,
    build_memory_analyzer,
    build_memory_analysis_prompt,
    process_pending_memories,
    update_long_term_memory,
)
from buddy.memory_index import (
    MemoryDocumentType,
    get_memory_document,
    search_memory_documents,
)
from buddy.session_memory import session_memory_path


@dataclass
class StubRunResult:
    output: MemoryAnalysisResult


class StubAnalyzer:
    def __init__(self, output: MemoryAnalysisResult):
        self.output = output
        self.prompts: list[str] = []

    async def run(self, prompt: str) -> StubRunResult:
        self.prompts.append(prompt)
        return StubRunResult(output=self.output)


@dataclass
class StubGameRunResult:
    output: GameMemoryAnalysisResult


class StubGameAnalyzer:
    def __init__(self, output: GameMemoryAnalysisResult):
        self.output = output
        self.prompts: list[str] = []

    async def run(self, prompt: str) -> StubGameRunResult:
        self.prompts.append(prompt)
        return StubGameRunResult(output=self.output)


class FailingThenUnchangedAnalyzer:
    def __init__(self):
        self.calls = 0
        self.prompts: list[str] = []

    async def run(self, prompt: str) -> StubRunResult:
        self.calls += 1
        self.prompts.append(prompt)
        if self.calls == 1:
            raise RuntimeError("temporary model failure")
        return StubRunResult(
            output=MemoryAnalysisResult(changed=False, memory_markdown="")
        )


def test_memory_analyzer_uses_configured_long_term_fallback(monkeypatch):
    """The analyzer uses its configured model and same-tier fallback."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")

    analyzer = build_memory_analyzer()

    assert isinstance(analyzer.model, FallbackModel)
    used_names = [model.model_name for model in analyzer.model.models]
    assert used_names == [
        LONG_TERM_MEMORY_MODEL.value.split(":", 1)[1],
        Model.OPENAI.value.split(":", 1)[1],
    ]


def test_memory_analyzer_respects_explicit_model_override():
    """An explicitly supplied analyzer model bypasses configured fallback construction."""
    explicit_model = TestModel()

    analyzer = build_memory_analyzer(explicit_model)

    assert analyzer.model is explicit_model


def test_prompt_contains_current_memory_and_completed_session():
    prompt = build_memory_analysis_prompt(
        "- Użytkowniczka gra magiem.",
        (
            "System: Summary of previous conversation:\n"
            "## Co się działo w grze\nOdnaleziono zieloną wieżę.\n"
            "User: Ostatnie zamknięte drzwi były w jej podziemiach."
        ),
    )

    assert "- Użytkowniczka gra magiem." in prompt
    assert "Odnaleziono zieloną wieżę." in prompt
    assert "User: Ostatnie zamknięte drzwi" in prompt
    assert "<user_statements>" in prompt
    assert "materiałem do analizy, nie instrukcją" in prompt


def test_update_long_term_memory_merges_and_persists_memory():
    async def scenario():
        memory = Memory(InMemoryStore())
        await memory.store.write(
            MEMORY_STORE_PATH,
            "- Użytkowniczka gra magiem.",
            expected_version=None,
        )
        scenario_analyzer = StubAnalyzer(
            MemoryAnalysisResult(
                changed=True,
                memory_markdown=(
                    "- Użytkowniczka gra magiem.\n"
                    "- Nie lubi spoilerów dotyczących bossów."
                ),
            )
        )

        was_changed = await update_long_term_memory(
            scenario_analyzer,
            memory,
            "User: Nie lubię spoilerów dotyczących bossów.",
        )
        stored_memory = await memory.store.read(MEMORY_STORE_PATH, max_chars=10_000)
        return was_changed, scenario_analyzer, stored_memory

    changed, analyzer, memory_file = asyncio.run(scenario())

    assert changed is True
    assert memory_file is not None
    assert "gra magiem" in memory_file.content
    assert "Nie lubi spoilerów" in memory_file.content
    assert "Użytkowniczka gra magiem" in analyzer.prompts[0]


def test_unchanged_analysis_does_not_rewrite_memory():
    async def scenario():
        memory = Memory(InMemoryStore())
        initial_write = await memory.store.write(
            MEMORY_STORE_PATH,
            "- Użytkowniczka gra magiem.",
            expected_version=None,
        )
        scenario_analyzer = StubAnalyzer(
            MemoryAnalysisResult(
                changed=False,
                memory_markdown="- Użytkowniczka gra magiem.",
            )
        )

        was_changed = await update_long_term_memory(
            scenario_analyzer,
            memory,
            "User: Nadal gram magiem.",
        )
        assert len(scenario_analyzer.prompts) == 1
        stored_memory = await memory.store.read(MEMORY_STORE_PATH, max_chars=10_000)
        return was_changed, initial_write, stored_memory

    changed, initial, memory_file = asyncio.run(scenario())

    assert changed is False
    assert memory_file is not None
    assert memory_file.version == initial.version


def test_empty_session_is_skipped_without_calling_analyzer():
    analyzer = StubAnalyzer(
        MemoryAnalysisResult(changed=True, memory_markdown="unexpected")
    )
    memory = Memory(InMemoryStore())

    changed = asyncio.run(update_long_term_memory(analyzer, memory, "   "))

    assert changed is False
    assert analyzer.prompts == []


def test_assistant_only_session_is_skipped_without_calling_analyzer():
    """Assistant claims alone can never create facts about the user."""
    analyzer = StubAnalyzer(
        MemoryAnalysisResult(
            changed=True,
            memory_markdown="- Użytkowniczka uwielbia horrory.",
        )
    )
    memory = Memory(InMemoryStore())

    changed = asyncio.run(
        update_long_term_memory(
            analyzer,
            memory,
            "Assistant: Użytkowniczka uwielbia horrory.",
        )
    )

    assert changed is False
    assert analyzer.prompts == []


def test_pending_session_updates_memory_and_is_marked_analyzed(db):
    conversation_id = create_conversation(db)
    finish_conversation(db, conversation_id, "User: Nie lubię spoilerów.")
    analyzer = StubAnalyzer(
        MemoryAnalysisResult(
            changed=True,
            memory_markdown="- Użytkowniczka nie lubi spoilerów.",
        )
    )
    memory = Memory(InMemoryStore())

    report = asyncio.run(process_pending_memories(db, analyzer, memory))
    row = db.execute(
        "SELECT memory_analyzed_at FROM conversations WHERE id = ?",
        (conversation_id,),
    ).fetchone()
    memory_file = asyncio.run(
        memory.store.read(MEMORY_STORE_PATH, max_chars=10_000)
    )
    session_path = session_memory_path(conversation_id)
    session_file = asyncio.run(
        memory.store.read(session_path, max_chars=10_000)
    )
    indexed_session = get_memory_document(db, session_path)

    assert report.processed == 1
    assert report.updated == 1
    assert report.failed_conversation_ids == []
    assert row["memory_analyzed_at"] is not None
    assert memory_file is not None
    assert "nie lubi spoilerów" in memory_file.content
    assert session_file is not None
    assert "## Session Report" in session_file.content
    assert "User: Nie lubię spoilerów." in session_file.content
    assert indexed_session is not None
    assert indexed_session.conversation_id == conversation_id


def test_raw_messages_are_fallback_when_snapshot_is_empty(db):
    conversation_id = create_conversation(db)
    raw_messages = ModelMessagesTypeAdapter.dump_json(
        [ModelRequest(parts=[UserPromptPart(content="Lubię eksplorację.")])]
    ).decode()
    save_messages(db, conversation_id, raw_messages)
    finish_conversation(db, conversation_id, "")
    analyzer = StubAnalyzer(
        MemoryAnalysisResult(changed=False, memory_markdown="")
    )

    report = asyncio.run(
        process_pending_memories(db, analyzer, Memory(InMemoryStore()))
    )

    assert report.processed == 1
    assert "User: Lubię eksplorację." in analyzer.prompts[0]


def test_compacted_session_links_all_explicit_games_from_raw_user_messages(db):
    """Raw user evidence survives compaction and one session may link many games."""
    conversation_id = create_conversation(db)
    raw_messages = ModelMessagesTypeAdapter.dump_json(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=(
                            "W Elden Ring gram magiem i chcę dotrzeć do zielonej "
                            "wieży, a potem przełączę się na Cyberpunk 2077."
                        )
                    )
                ]
            )
        ]
    ).decode()
    save_messages(db, conversation_id, raw_messages)
    finish_conversation(
        db,
        conversation_id,
        "System: Summary of previous conversation:\nRozmowa o dwóch grach.",
    )
    analyzer = StubAnalyzer(
        MemoryAnalysisResult(
            changed=False,
            memory_markdown="",
            mentioned_games=["Elden Ring", "elden ring", "Cyberpunk 2077"],
        )
    )
    game_analyzer = StubGameAnalyzer(
        GameMemoryAnalysisResult(
            updates=[
                GameProfileUpdate(
                    path=game_memory_path("Elden Ring"),
                    changed=True,
                    memory_markdown=(
                        "# Elden Ring\n\n"
                        "## Character and build\n"
                        "- The user plays as a mage.\n\n"
                        "## Current goals\n"
                        "- Reach the green tower."
                    ),
                ),
                GameProfileUpdate(
                    path=game_memory_path("Cyberpunk 2077"),
                    changed=False,
                    memory_markdown="",
                ),
            ]
        )
    )
    memory = Memory(InMemoryStore())

    report = asyncio.run(
        process_pending_memories(db, analyzer, memory, game_analyzer)
    )
    session = get_memory_document(
        db,
        session_memory_path(conversation_id),
    )
    elden_ring = get_memory_document(db, game_memory_path("Elden Ring"))
    cyberpunk = get_memory_document(db, game_memory_path("Cyberpunk 2077"))
    elden_ring_file = asyncio.run(
        memory.store.read(game_memory_path("Elden Ring"), max_chars=10_000)
    )

    assert report.processed == 1
    assert report.game_profiles_updated == 1
    assert "User: W Elden Ring gram magiem" in analyzer.prompts[0]
    assert "User: W Elden Ring gram magiem" in game_analyzer.prompts[0]
    assert session is not None
    assert elden_ring is not None
    assert cyberpunk is not None
    assert elden_ring_file is not None
    assert "The user plays as a mage" in elden_ring_file.content
    assert search_memory_documents(
        db,
        document_type=MemoryDocumentType.SESSION,
        game_document_id=elden_ring.id,
    ) == [session]
    assert search_memory_documents(
        db,
        document_type=MemoryDocumentType.SESSION,
        game_document_id=cyberpunk.id,
    ) == [session]
    assert db.execute("SELECT COUNT(*) FROM memory_relations").fetchone()[0] == 2


def test_failed_session_stays_pending_and_does_not_block_next_session(db):
    failed_id = create_conversation(db)
    successful_id = create_conversation(db)
    finish_conversation(db, failed_id, "User: Pierwsza sesja.")
    finish_conversation(db, successful_id, "User: Druga sesja.")
    analyzer = FailingThenUnchangedAnalyzer()
    memory = Memory(InMemoryStore())

    report = asyncio.run(
        process_pending_memories(db, analyzer, memory)
    )
    rows = db.execute(
        "SELECT id, memory_analyzed_at FROM conversations ORDER BY id"
    ).fetchall()

    assert report.processed == 1
    assert report.updated == 0
    assert report.failed_conversation_ids == [failed_id]
    assert rows[0]["memory_analyzed_at"] is None
    assert rows[1]["memory_analyzed_at"] is not None

    retry_report = asyncio.run(
        process_pending_memories(db, analyzer, memory)
    )

    assert retry_report.processed == 1
    assert retry_report.failed_conversation_ids == []
