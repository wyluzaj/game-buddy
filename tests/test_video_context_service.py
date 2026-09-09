import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from pydantic_ai import RunUsage
from pydantic_ai.models.fallback import FallbackModel

from buddy.config import MODEL_TIERS, MODELS
from buddy.database import create_conversation
from buddy.prompts import VIDEO_CONTEXT_BUTLER_INSTRUCTIONS
from buddy.video.context_butler import (
    VideoContextButlerResult,
    build_video_context_butler,
)
from buddy.video.context_service import process_video_report_context
from buddy.video.models import EpisodeAnalysis, EpisodeObservation, VideoReportMode
from buddy.video.storage import (
    claim_video_segment_for_analysis,
    create_video_segment,
    get_latest_video_context,
    save_video_report,
)


@dataclass
class StubVideoContextRunResult:
    output: VideoContextButlerResult


class StubVideoContextButler:
    def __init__(self, outputs: list[VideoContextButlerResult]) -> None:
        self.outputs = outputs
        self.prompts: list[str] = []
        self.usages: list[RunUsage | None] = []

    async def run(
        self,
        prompt: str,
        *,
        usage: RunUsage | None = None,
    ) -> StubVideoContextRunResult:
        self.prompts.append(prompt)
        self.usages.append(usage)
        return StubVideoContextRunResult(self.outputs.pop(0))


def test_video_context_prompt_requires_brief_but_concrete_output():
    instructions = " ".join(VIDEO_CONTEXT_BUTLER_INSTRUCTIONS.split())

    assert "concise but information-dense" in instructions
    assert "what was found, where it was, and what changed" in instructions
    assert '"the player explored"' in instructions
    assert "Do not omit an important fact" in instructions
    assert "present location" in instructions
    assert "Use specific nouns and outcomes" in instructions


def _episode_report(
    db,
    tmp_path,
    *,
    name: str,
    conversation_id: int | None = None,
    summary: str,
    event_start: float,
    event_end: float,
    start_offset_minutes: int = 0,
):
    path = tmp_path / f"{name}.mp4"
    path.write_bytes(b"gameplay")
    started_at = datetime(2026, 8, 29, 18, 0, tzinfo=timezone.utc) + timedelta(
        minutes=start_offset_minutes
    )
    segment = create_video_segment(
        db,
        conversation_id=conversation_id,
        file_path=path,
        started_at=started_at,
        ended_at=started_at + timedelta(minutes=2),
    )
    assert claim_video_segment_for_analysis(db, segment.id) is not None
    episode = EpisodeAnalysis(
        observations=[
            EpisodeObservation(
                start_seconds=event_start,
                end_seconds=event_end,
                activity="Testowa aktywność.",
                description=summary,
            )
        ],
    )
    report = save_video_report(
        db,
        segment_id=segment.id,
        mode=VideoReportMode.EPISODE,
        provider="google",
        model_name="gemini-3.1-flash-lite",
        content=episode.model_dump(mode="json"),
    )
    return segment, report


def test_video_context_butler_uses_its_explicit_configured_model(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")

    butler = build_video_context_butler()

    assert isinstance(butler.model, FallbackModel)
    expected = [MODELS.video_context_butler.value.split(":", 1)[1]] + [
        model.value.split(":", 1)[1]
        for model, tier in MODEL_TIERS.items()
        if tier == MODEL_TIERS[MODELS.video_context_butler]
        and model is not MODELS.video_context_butler
    ]
    assert [model.model_name for model in butler.model.models] == expected


def test_context_is_a_source_validated_replacement_state(db, tmp_path):
    """The next prompt receives prior state and persists one newer snapshot."""
    conversation_id = create_conversation(db)
    first_segment, first_report = _episode_report(
        db,
        tmp_path,
        name="first",
        summary="Znaleziono zamknięte drzwi.",
        event_start=12,
        event_end=18,
        conversation_id=conversation_id,
    )
    first_source = f"{first_segment.source_uri}?t=12-18"
    first_butler = StubVideoContextButler(
        [
            VideoContextButlerResult(
                episode_summary="Gracz znalazł zamknięte drzwi.",
                changed=True,
                context="Użytkowniczka znalazła zamknięte drzwi.",
                sources=[first_source],
            )
        ]
    )
    usage = RunUsage()
    first = asyncio.run(
        process_video_report_context(
            db,
            first_butler,
            first_report.id,
            usage=usage,
        )
    )

    second_segment, second_report = _episode_report(
        db,
        tmp_path,
        name="second",
        summary="Znaleziono klucz przy schodach.",
        event_start=30,
        event_end=35,
        conversation_id=conversation_id,
    )
    second_source = f"{second_segment.source_uri}?t=30-35"
    second_butler = StubVideoContextButler(
        [
            VideoContextButlerResult(
                episode_summary="Gracz znalazł klucz przy schodach.",
                changed=True,
                context=(
                    "Użytkowniczka znalazła zamknięte drzwi, a później klucz "
                    "przy schodach."
                ),
                sources=[first_source, second_source],
            )
        ]
    )
    second = asyncio.run(
        process_video_report_context(db, second_butler, second_report.id)
    )

    assert first_butler.usages == [usage]
    assert first.changed is True
    assert first.sources == [first_source]
    assert "zamknięte drzwi" in second_butler.prompts[0]
    assert first_source in second_butler.prompts[0]
    assert second.sources == [first_source, second_source]
    assert get_latest_video_context(db, conversation_id=conversation_id) == second


def test_no_change_is_persisted_once_without_polluting_context(db, tmp_path):
    """An irrelevant overview gets an idempotency marker but no context text."""
    segment, report = _episode_report(
        db,
        tmp_path,
        name="walking",
        summary="Postać szła pustym korytarzem.",
        event_start=1,
        event_end=20,
    )
    butler = StubVideoContextButler(
        [
            VideoContextButlerResult(
                episode_summary="Postać przeszła pustym korytarzem.",
                changed=False,
                context="Ten tekst ma zostać usunięty.",
                sources=[segment.source_uri],
            )
        ]
    )

    first = asyncio.run(process_video_report_context(db, butler, report.id))
    second = asyncio.run(process_video_report_context(db, butler, report.id))

    assert first == second
    assert first.changed is False
    assert first.context == ""
    assert first.sources == []
    assert len(butler.prompts) == 1


def test_unassigned_clips_do_not_share_previous_context(db, tmp_path):
    first_segment, first_report = _episode_report(
        db,
        tmp_path,
        name="unassigned-first",
        summary="Pierwszy niezależny klip.",
        event_start=1,
        event_end=2,
    )
    first_butler = StubVideoContextButler(
        [
            VideoContextButlerResult(
                episode_summary="Pierwszy niezależny klip.",
                changed=True,
                context="Stan pierwszego klipu.",
                sources=[f"{first_segment.source_uri}?t=1-2"],
            )
        ]
    )
    asyncio.run(process_video_report_context(db, first_butler, first_report.id))

    _, second_report = _episode_report(
        db,
        tmp_path,
        name="unassigned-second",
        summary="Drugi niezależny klip.",
        event_start=3,
        event_end=4,
    )
    second_butler = StubVideoContextButler(
        [
            VideoContextButlerResult(
                episode_summary="Drugi niezależny klip.",
                changed=False,
            )
        ]
    )
    asyncio.run(process_video_report_context(db, second_butler, second_report.id))

    assert "Stan pierwszego klipu" not in second_butler.prompts[0]
    assert '"context": ""' in second_butler.prompts[0]


def test_invented_video_source_is_rejected(db, tmp_path):
    """The delegate cannot attach context to a video it never received."""
    _, report = _episode_report(
        db,
        tmp_path,
        name="invented",
        summary="Znaleziono skrzynię.",
        event_start=5,
        event_end=8,
    )
    butler = StubVideoContextButler(
        [
            VideoContextButlerResult(
                episode_summary="Gracz znalazł skrzynię.",
                changed=True,
                context="Znaleziono ważną skrzynię.",
                sources=["video://segment/999?t=1-2"],
            )
        ]
    )

    update = asyncio.run(process_video_report_context(db, butler, report.id))

    assert update.changed is False
    assert update.context == ""
    assert update.sources == []
    assert update.uncertainty is not None
    assert "unknown video source" in update.uncertainty


def test_late_older_report_cannot_replace_newer_video_context(db, tmp_path):
    """Out-of-order provider completion cannot move observed state backwards."""
    conversation_id = create_conversation(db)
    newer_segment, newer_report = _episode_report(
        db,
        tmp_path,
        name="newer",
        summary="Użytkowniczka dotarła do wieży.",
        event_start=20,
        event_end=25,
        start_offset_minutes=2,
        conversation_id=conversation_id,
    )
    newer_butler = StubVideoContextButler(
        [
            VideoContextButlerResult(
                episode_summary="Gracz dotarł do wieży.",
                changed=True,
                context="Użytkowniczka dotarła do wieży.",
                sources=[f"{newer_segment.source_uri}?t=20-25"],
            )
        ]
    )
    asyncio.run(process_video_report_context(db, newer_butler, newer_report.id))

    _, older_report = _episode_report(
        db,
        tmp_path,
        name="older",
        summary="Użytkowniczka była jeszcze przed mostem.",
        event_start=10,
        event_end=15,
        conversation_id=conversation_id,
    )
    older_butler = StubVideoContextButler(
        [
            VideoContextButlerResult(
                episode_summary="Gracz był jeszcze przed mostem.",
                changed=True,
                context="Gracz jest przed mostem.",
                sources=["video://segment/2?t=10-15"],
            )
        ]
    )

    skipped = asyncio.run(
        process_video_report_context(db, older_butler, older_report.id)
    )

    assert skipped.changed is False
    assert skipped.uncertainty is not None
    assert "late report" in skipped.uncertainty
    assert skipped.episode_summary == "Gracz był jeszcze przed mostem."
    assert len(older_butler.prompts) == 1
