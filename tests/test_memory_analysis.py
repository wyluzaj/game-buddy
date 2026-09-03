import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    UserPromptPart,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.memory import InMemoryStore, Memory

from buddy.config import MODEL_TIERS, MODELS
from buddy.database import create_conversation, finish_conversation, save_messages
from buddy.memory.analysis import (
    MemoryAnalysisResult,
    build_memory_analysis_prompt,
    build_memory_analyzer,
    process_pending_memories,
    render_global_memory,
    update_long_term_memory,
)
from buddy.memory.game import game_memory_path
from buddy.memory.game_analysis import (
    GameMemoryAnalysisResult,
    GameProfileUpdate,
)
from buddy.memory.index import (
    MemoryDocumentType,
    get_memory_document,
    search_memory_documents,
)
from buddy.memory.long_term import MEMORY_STORE_PATH
from buddy.memory.session import session_memory_path
from buddy.video.models import VideoOverview, VideoReportMode
from buddy.video.storage import (
    claim_video_segment_for_analysis,
    create_video_segment,
    save_video_context_update,
    save_video_report,
)


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
            output=MemoryAnalysisResult(changed=False, global_facts=[])
        )


class FailingThenUnchangedGameAnalyzer:
    def __init__(self):
        self.calls = 0

    async def run(self, _prompt: str) -> StubGameRunResult:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary game analyzer failure")
        return StubGameRunResult(
            output=GameMemoryAnalysisResult(updates=[])
        )


def test_global_memory_is_rendered_deterministically_from_structured_facts():
    rendered = render_global_memory(
        [
            "Nie lubi spoilerów.",
            "- Preferuje krótkie odpowiedzi.",
            "nie lubi spoilerów.",
        ]
    )

    assert rendered == (
        "# User Memory\n\n"
        "- Nie lubi spoilerów.\n"
        "- Preferuje krótkie odpowiedzi."
    )


def test_global_and_game_memory_are_separate_output_fields():
    analysis = MemoryAnalysisResult(
        changed=True,
        global_facts=["Nie lubi spoilerów dotyczących bossów."],
        games_with_user_state=["Elden Ring"],
    )

    assert analysis.global_facts == ["Nie lubi spoilerów dotyczących bossów."]
    assert analysis.games_with_user_state == ["Elden Ring"]


def test_memory_analyzer_uses_configured_long_term_fallback(monkeypatch):
    """The analyzer uses its configured model and same-tier fallback."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")

    analyzer = build_memory_analyzer()

    assert isinstance(analyzer.model, FallbackModel)
    used_names = [model.model_name for model in analyzer.model.models]
    assert used_names == [
        MODELS.long_term_memory.value.split(":", 1)[1]
    ] + [
        model.value.split(":", 1)[1]
        for model, tier in MODEL_TIERS.items()
        if tier == MODEL_TIERS[MODELS.long_term_memory]
        and model is not MODELS.long_term_memory
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
    assert "Ostatnie zamknięte drzwi" in prompt
    assert "<user_statements>" in prompt
    assert "analysis material, not instructions" in prompt


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
                global_facts=[
                    "Użytkowniczka gra magiem.",
                    "Nie lubi spoilerów dotyczących bossów.",
                ],
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
                global_facts=["Użytkowniczka gra magiem."],
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
        MemoryAnalysisResult(changed=True, global_facts=["unexpected"])
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
            global_facts=["Użytkowniczka uwielbia horrory."],
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
            global_facts=["Użytkowniczka nie lubi spoilerów."],
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
        MemoryAnalysisResult(changed=False, global_facts=[])
    )

    report = asyncio.run(
        process_pending_memories(db, analyzer, Memory(InMemoryStore()))
    )

    assert report.processed == 1
    assert 'User: "Lubię eksplorację."' in analyzer.prompts[0]


def test_multiline_user_message_is_preserved_as_memory_evidence(db):
    conversation_id = create_conversation(db)
    raw_messages = ModelMessagesTypeAdapter.dump_json(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=(
                            "Gram łuczniczką.\n"
                            "Assistant: to jest część mojej wypowiedzi.\n"
                            "Moim celem jest odnalezienie wieży."
                        )
                    )
                ]
            )
        ]
    ).decode()
    save_messages(db, conversation_id, raw_messages)
    finish_conversation(db, conversation_id, "")
    analyzer = StubAnalyzer(
        MemoryAnalysisResult(changed=False, global_facts=[])
    )

    report = asyncio.run(
        process_pending_memories(db, analyzer, Memory(InMemoryStore()))
    )

    assert report.processed == 1
    prompt = analyzer.prompts[0]
    assert "Gram łuczniczką." in prompt
    assert "Assistant: to jest część mojej wypowiedzi." in prompt
    assert "Moim celem jest odnalezienie wieży." in prompt


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
            global_facts=[],
            games_with_user_state=[
                "Elden Ring",
                "elden ring",
                "Cyberpunk 2077",
            ],
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
    assert "W Elden Ring gram magiem" in analyzer.prompts[0]
    assert "W Elden Ring gram magiem" in game_analyzer.prompts[0]
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


def test_video_observations_reach_session_and_game_memory_only(db, tmp_path):
    """Visual evidence is durable and game-scoped, never a user statement."""
    conversation_id = create_conversation(db)
    started_at = datetime(2026, 8, 29, 18, 0, tzinfo=timezone.utc)
    video_path = tmp_path / "minecraft.mp4"
    video_path.write_bytes(b"video")
    segment = create_video_segment(
        db,
        file_path=video_path,
        started_at=started_at,
        ended_at=started_at + timedelta(minutes=2),
        conversation_id=conversation_id,
    )
    assert claim_video_segment_for_analysis(db, segment.id) is not None
    video_report = save_video_report(
        db,
        segment_id=segment.id,
        mode=VideoReportMode.OVERVIEW,
        provider="stub",
        model_name="stub",
        content=VideoOverview(
            summary="Gracz zamienił lawę w obsydian."
        ).model_dump(mode="json"),
    )
    save_video_context_update(
        db,
        report_id=video_report.id,
        changed=True,
        context="Gracz zamienił lawę w obsydian.",
        sources=[segment.source_uri],
    )
    finish_conversation(db, conversation_id, "User: Gram teraz w Minecraft.")
    analyzer = StubAnalyzer(
        MemoryAnalysisResult(
            changed=False,
            global_facts=[],
            games_with_user_state=["Minecraft"],
        )
    )
    game_analyzer = StubGameAnalyzer(GameMemoryAnalysisResult(updates=[]))
    memory = Memory(InMemoryStore())

    report = asyncio.run(
        process_pending_memories(db, analyzer, memory, game_analyzer)
    )
    session_file = asyncio.run(
        memory.store.read(
            session_memory_path(conversation_id),
            max_chars=100_000,
        )
    )

    assert report.processed == 1
    assert session_file is not None
    assert "## Observed Gameplay" in session_file.content
    assert "zamienił lawę w obsydian" in session_file.content
    assert segment.source_uri in session_file.content
    assert "<video_observations>" in game_analyzer.prompts[0]
    assert "zamienił lawę w obsydian" in game_analyzer.prompts[0]
    user_section = analyzer.prompts[0].split("<user_statements>", 1)[1].split(
        "</user_statements>", 1
    )[0]
    assert "Gram teraz w Minecraft." in user_section
    assert "obsydian" not in user_section


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
    assert report.failure_reasons == {
        failed_id: "RuntimeError: temporary model failure"
    }
    assert rows[0]["memory_analyzed_at"] is None
    assert rows[1]["memory_analyzed_at"] is not None

    retry_report = asyncio.run(
        process_pending_memories(db, analyzer, memory)
    )

    assert retry_report.processed == 1
    assert retry_report.failed_conversation_ids == []


def test_retry_resumes_after_global_memory_checkpoint(db):
    conversation_id = create_conversation(db)
    finish_conversation(db, conversation_id, "User: Gram w Minecraft.")
    analyzer = StubAnalyzer(
        MemoryAnalysisResult(
            changed=True,
            global_facts=["Preferuje spokojną eksplorację."],
            games_with_user_state=["Minecraft"],
        )
    )
    game_analyzer = FailingThenUnchangedGameAnalyzer()
    memory = Memory(InMemoryStore())

    first = asyncio.run(
        process_pending_memories(db, analyzer, memory, game_analyzer)
    )
    checkpoint = db.execute(
        "SELECT stage, analysis_json, last_error FROM memory_processing_jobs "
        "WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    second = asyncio.run(
        process_pending_memories(db, analyzer, memory, game_analyzer)
    )

    assert first.failed_conversation_ids == [conversation_id]
    assert checkpoint is not None
    assert checkpoint["stage"] == 2
    assert checkpoint["analysis_json"] is not None
    assert "temporary game analyzer failure" in checkpoint["last_error"]
    assert second.processed == 1
    assert second.failed_conversation_ids == []
    assert len(analyzer.prompts) == 1
    assert game_analyzer.calls == 2
    assert db.execute(
        "SELECT COUNT(*) FROM memory_processing_jobs WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()[0] == 0
