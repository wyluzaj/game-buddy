from datetime import datetime, timedelta, timezone

import pytest

from buddy.database import create_conversation
from buddy.memory.index import MemoryDocumentType, index_memory_document
from buddy.video.models import (
    EpisodeAnalysis,
    EpisodeObservation,
    VideoReportMode,
    VideoSegmentStatus,
)
from buddy.video.storage import (
    VideoStorageDataError,
    claim_video_segment_for_analysis,
    create_video_segment,
    get_episode_observations,
    get_video_report,
    get_video_reports,
    get_video_segment,
    save_video_report,
    search_episode_observations,
    update_video_segment_status,
)


def _recorded_segment(db):
    conversation_id = create_conversation(db)
    started_at = datetime(2026, 8, 29, 18, 0, tzinfo=timezone.utc)
    return create_video_segment(
        db,
        conversation_id=conversation_id,
        file_path="data/videos/2026-08-29/segment-1.mp4",
        started_at=started_at,
        ended_at=started_at + timedelta(minutes=2),
    )


def test_segment_and_overview_report_preserve_the_analysis_lifecycle(db):
    """A local recording becomes a stable source and completes with its overview."""
    segment = _recorded_segment(db)

    assert segment.duration_seconds == 120
    assert segment.status is VideoSegmentStatus.PENDING
    assert segment.source_uri == f"video://segment/{segment.id}"

    processing = update_video_segment_status(
        db,
        segment.id,
        VideoSegmentStatus.PROCESSING,
    )
    assert processing.status is VideoSegmentStatus.PROCESSING

    report = save_video_report(
        db,
        segment_id=segment.id,
        mode=VideoReportMode.OVERVIEW,
        provider="google",
        model_name="gemini-3.1-flash-lite",
        content={"summary": "Znaleziono zamknięte drzwi.", "events": []},
    )

    completed = get_video_segment(db, segment.id)
    assert completed is not None
    assert completed.status is VideoSegmentStatus.COMPLETED
    assert report.content["summary"] == "Znaleziono zamknięte drzwi."
    assert get_video_reports(db, segment.id) == [report]


def test_segment_rejects_ambiguous_or_inverted_timestamps(db):
    """Persisted timelines require unambiguous UTC-convertible time ranges."""
    aware = datetime(2026, 8, 29, 18, 0, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="timezone"):
        create_video_segment(
            db,
            file_path="naive.mp4",
            started_at=aware.replace(tzinfo=None),
            ended_at=(aware + timedelta(minutes=2)).replace(tzinfo=None),
        )

    with pytest.raises(ValueError, match="must end after"):
        create_video_segment(
            db,
            file_path="inverted.mp4",
            started_at=aware,
            ended_at=aware - timedelta(seconds=1),
        )


def test_detail_report_requires_a_question_and_can_be_filtered(db):
    """On-demand reports retain the question that explains why they exist."""
    segment = _recorded_segment(db)
    assert claim_video_segment_for_analysis(db, segment.id) is not None
    save_video_report(
        db,
        segment_id=segment.id,
        mode=VideoReportMode.OVERVIEW,
        provider="google",
        model_name="gemini-3.1-flash-lite",
        content={"summary": "Overview before detail."},
    )

    with pytest.raises(ValueError, match="requires its question"):
        save_video_report(
            db,
            segment_id=segment.id,
            mode=VideoReportMode.DETAIL,
            provider="google",
            model_name="gemini-3.1-flash-lite",
            content={"answer": "Nie wiadomo."},
        )

    detail = save_video_report(
        db,
        segment_id=segment.id,
        mode=VideoReportMode.DETAIL,
        provider="google",
        model_name="gemini-3.1-flash-lite",
        question="Czy obok drzwi był symbol?",
        content={"answer": "Tak, czerwony symbol.", "timestamp_seconds": 74},
    )

    assert get_video_reports(
        db,
        segment.id,
        mode=VideoReportMode.DETAIL,
    ) == [detail]
    assert detail.question == "Czy obok drzwi był symbol?"


def test_deleting_a_segment_cascades_to_its_reports(db):
    """Removing video metadata cannot leave orphaned analysis reports."""
    segment = _recorded_segment(db)
    assert claim_video_segment_for_analysis(db, segment.id) is not None
    save_video_report(
        db,
        segment_id=segment.id,
        mode=VideoReportMode.OVERVIEW,
        provider="google",
        model_name="gemini-3.1-flash-lite",
        content={"summary": "Test."},
    )

    db.execute("DELETE FROM video_segments WHERE id = ?", (segment.id,))
    db.commit()

    count = db.execute(
        "SELECT COUNT(*) AS count FROM video_reports WHERE segment_id = ?",
        (segment.id,),
    ).fetchone()["count"]
    assert count == 0


def test_invalid_segment_state_transitions_are_rejected(db):
    segment = _recorded_segment(db)

    with pytest.raises(ValueError, match="pending -> completed"):
        update_video_segment_status(
            db,
            segment.id,
            VideoSegmentStatus.COMPLETED,
        )
    with pytest.raises(ValueError, match="requires a processing segment"):
        save_video_report(
            db,
            segment_id=segment.id,
            mode=VideoReportMode.OVERVIEW,
            provider="stub",
            model_name="stub",
            content={"summary": "Invalid direct completion."},
        )

    assert claim_video_segment_for_analysis(db, segment.id) is not None
    save_video_report(
        db,
        segment_id=segment.id,
        mode=VideoReportMode.OVERVIEW,
        provider="stub",
        model_name="stub",
        content={"summary": "Valid completion."},
    )
    with pytest.raises(ValueError, match="completed -> processing"):
        update_video_segment_status(
            db,
            segment.id,
            VideoSegmentStatus.PROCESSING,
        )


def test_corrupt_report_json_raises_a_storage_boundary_error(db):
    segment = _recorded_segment(db)
    assert claim_video_segment_for_analysis(db, segment.id) is not None
    report = save_video_report(
        db,
        segment_id=segment.id,
        mode=VideoReportMode.OVERVIEW,
        provider="stub",
        model_name="stub",
        content={"summary": "Valid before corruption."},
    )
    db.execute(
        "UPDATE video_reports SET report_json = ? WHERE id = ?",
        ("not-json", report.id),
    )
    db.commit()

    with pytest.raises(VideoStorageDataError, match="Stored video report"):
        get_video_report(db, report.id)


def test_episode_report_persists_and_indexes_detailed_observations(db):
    segment = _recorded_segment(db)
    assert claim_video_segment_for_analysis(db, segment.id) is not None
    analysis = EpisodeAnalysis(
        observations=[
            EpisodeObservation(
                start_seconds=18,
                end_seconds=32,
                activity="Wejście do nowej lokacji.",
                location="Wyspa otoczona wodą",
                description="Gracz mija czerwone schody przy wejściu do tunelu.",
                visible_objects=["czerwone schody", "kamienny tunel"],
            )
        ],
        possible_game="Big Walk",
        confidence=0.9,
    )

    report = save_video_report(
        db,
        segment_id=segment.id,
        mode=VideoReportMode.EPISODE,
        provider="google",
        model_name="gemini-3.1-flash-lite",
        content=analysis.model_dump(mode="json"),
    )

    completed = get_video_segment(db, segment.id)
    observations = get_episode_observations(db, segment.id)
    matches = search_episode_observations(db, query="czerwone schody")
    assert completed is not None
    assert completed.status is VideoSegmentStatus.COMPLETED
    assert report.mode is VideoReportMode.EPISODE
    assert observations == matches
    assert observations[0].source_uri == f"{segment.source_uri}?t=18-32"


def test_episode_search_can_be_limited_to_a_confirmed_game(db):
    first_game = index_memory_document(
        db,
        document_type=MemoryDocumentType.GAME,
        path="main/games/big-walk.md",
        title="Big Walk",
        summary="Big Walk profile",
        body="Big Walk",
    )
    second_game = index_memory_document(
        db,
        document_type=MemoryDocumentType.GAME,
        path="main/games/other-game.md",
        title="Other Game",
        summary="Other profile",
        body="Other Game",
    )
    started_at = datetime(2026, 8, 29, 18, 0, tzinfo=timezone.utc)
    segments = [
        create_video_segment(
            db,
            file_path=f"game-{game_id}.mp4",
            started_at=started_at + timedelta(minutes=index * 2),
            ended_at=started_at + timedelta(minutes=index * 2 + 2),
            game_document_id=game_id,
        )
        for index, game_id in enumerate((first_game.id, second_game.id))
    ]
    for segment in segments:
        assert claim_video_segment_for_analysis(db, segment.id) is not None
        save_video_report(
            db,
            segment_id=segment.id,
            mode=VideoReportMode.EPISODE,
            provider="stub",
            model_name="stub",
            content=EpisodeAnalysis(
                observations=[
                    EpisodeObservation(
                        start_seconds=0,
                        end_seconds=10,
                        activity="Eksploracja.",
                        description="Widoczny czerwony balon.",
                        visible_objects=["czerwony balon"],
                    )
                ]
            ).model_dump(mode="json"),
        )

    matches = search_episode_observations(
        db,
        query="czerwony balon",
        game_document_id=first_game.id,
    )

    assert [match.segment_id for match in matches] == [segments[0].id]


def test_video_segment_rejects_a_non_game_memory_document(db):
    session = index_memory_document(
        db,
        document_type=MemoryDocumentType.SESSION,
        path="main/sessions/session-1.md",
        title="Session 1",
        summary="Session",
        body="Session",
    )
    started_at = datetime(2026, 8, 29, 18, 0, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="is not an indexed game profile"):
        create_video_segment(
            db,
            file_path="invalid-game.mp4",
            started_at=started_at,
            ended_at=started_at + timedelta(minutes=2),
            game_document_id=session.id,
        )
