import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from buddy.memory.index import MemoryDocumentType, index_memory_document
from buddy.video.analysis_service import (
    VideoSegmentStateError,
    analyze_video_segment,
    register_and_analyze_video,
)
from buddy.video.analyzer import VideoProviderError
from buddy.video.models import (
    EpisodeAnalysis,
    EpisodeObservation,
    VideoSegmentStatus,
)
from buddy.video.storage import (
    claim_video_segment_for_analysis,
    create_video_segment,
    get_video_reports,
    get_video_segment,
)


class StubVideoAnalyzer:
    provider = "stub"
    model_name = "stub-video-model"

    def __init__(
        self,
        results: list[EpisodeAnalysis | Exception],
    ) -> None:
        self.results = results
        self.calls: list[Path] = []

    async def analyze_episode(self, video_path: Path) -> EpisodeAnalysis:
        self.calls.append(video_path)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _segment(db, tmp_path):
    video_path = tmp_path / "segment.mp4"
    video_path.write_bytes(b"video without an audio track")
    started_at = datetime(2026, 8, 29, 18, 0, tzinfo=timezone.utc)
    return create_video_segment(
        db,
        file_path=video_path,
        started_at=started_at,
        ended_at=started_at + timedelta(minutes=2),
    )


def _episode(description: str) -> EpisodeAnalysis:
    return EpisodeAnalysis(
        observations=[
            EpisodeObservation(
                start_seconds=0,
                end_seconds=10,
                activity="Eksploracja.",
                description=description,
            )
        ]
    )


def _game_document(db, title: str = "Big Walk"):
    return index_memory_document(
        db,
        document_type=MemoryDocumentType.GAME,
        path="main/games/big-walk.md",
        title=title,
        summary=f"Pamięć dotycząca gry {title}.",
        body=f"# {title}\n",
    )


def test_successful_analysis_persists_episode_and_completes_segment(db, tmp_path):
    """The service owns the complete pending-to-report lifecycle."""
    segment = _segment(db, tmp_path)
    episode = _episode("Znaleziono zamknięte drzwi.")
    analyzer = StubVideoAnalyzer([episode])

    outcome = asyncio.run(analyze_video_segment(db, analyzer, segment.id))

    assert outcome.episode == episode
    assert outcome.reused_existing_report is False
    assert outcome.segment.status is VideoSegmentStatus.COMPLETED
    assert outcome.report.provider == "stub"
    assert outcome.report.model_name == "stub-video-model"
    assert analyzer.calls == [segment.file_path]
    assert len(get_video_reports(db, segment.id)) == 1


def test_failed_analysis_is_retryable_and_clears_previous_error(db, tmp_path):
    """A provider outage is recorded but does not permanently poison the clip."""
    segment = _segment(db, tmp_path)
    analyzer = StubVideoAnalyzer(
        [
            VideoProviderError("temporary quota error"),
            _episode("Drugie podejście zadziałało."),
        ]
    )

    with pytest.raises(VideoProviderError, match="temporary quota error"):
        asyncio.run(analyze_video_segment(db, analyzer, segment.id))

    failed = get_video_segment(db, segment.id)
    assert failed is not None
    assert failed.status is VideoSegmentStatus.FAILED
    assert failed.error == "VideoProviderError: temporary quota error"

    retried = asyncio.run(analyze_video_segment(db, analyzer, segment.id))

    assert retried.segment.status is VideoSegmentStatus.COMPLETED
    assert retried.segment.error is None
    assert len(analyzer.calls) == 2


def test_completed_segment_reuses_report_without_second_provider_call(db, tmp_path):
    """Idempotent processing avoids duplicate reports and API usage."""
    segment = _segment(db, tmp_path)
    episode = _episode("Pierwszy raport.")
    analyzer = StubVideoAnalyzer([episode])
    first = asyncio.run(analyze_video_segment(db, analyzer, segment.id))

    second = asyncio.run(analyze_video_segment(db, analyzer, segment.id))

    assert second.reused_existing_report is True
    assert second.report == first.report
    assert second.episode == first.episode
    assert analyzer.calls == [segment.file_path]
    assert len(get_video_reports(db, segment.id)) == 1


def test_processing_segment_cannot_be_claimed_by_a_second_worker(db, tmp_path):
    """An atomic status transition prevents duplicate concurrent submissions."""
    segment = _segment(db, tmp_path)
    claimed = claim_video_segment_for_analysis(db, segment.id)
    assert claimed is not None
    analyzer = StubVideoAnalyzer([_episode("Nie powinno być użyte.")])

    with pytest.raises(VideoSegmentStateError, match="status processing"):
        asyncio.run(analyze_video_segment(db, analyzer, segment.id))

    assert analyzer.calls == []


def test_registering_the_same_file_twice_reuses_segment_and_report(db, tmp_path):
    """A repeated CLI invocation is safe and does not duplicate API work."""
    video_path = tmp_path / "manual-clip.mp4"
    video_path.write_bytes(b"gameplay clip")
    started_at = datetime(2026, 8, 29, 18, 0, tzinfo=timezone.utc)
    ended_at = started_at + timedelta(minutes=2)
    analyzer = StubVideoAnalyzer([_episode("Gotowy raport.")])

    first = asyncio.run(
        register_and_analyze_video(
            db,
            analyzer,
            file_path=video_path,
            started_at=started_at,
            ended_at=ended_at,
        )
    )
    second = asyncio.run(
        register_and_analyze_video(
            db,
            analyzer,
            file_path=video_path,
            started_at=started_at,
            ended_at=ended_at,
        )
    )

    segment_count = db.execute(
        "SELECT COUNT(*) AS count FROM video_segments"
    ).fetchone()["count"]
    assert segment_count == 1
    assert second.segment.id == first.segment.id
    assert second.reused_existing_report is True
    assert analyzer.calls == [video_path]


def test_overwritten_file_path_creates_a_new_recording(db, tmp_path):
    video_path = tmp_path / "reused-name.mp4"
    started_at = datetime(2026, 8, 29, 18, 0, tzinfo=timezone.utc)
    ended_at = started_at + timedelta(minutes=2)
    video_path.write_bytes(b"first recording")
    analyzer = StubVideoAnalyzer(
        [
            _episode("Pierwsze nagranie."),
            _episode("Drugie nagranie."),
        ]
    )

    first = asyncio.run(
        register_and_analyze_video(
            db,
            analyzer,
            file_path=video_path,
            started_at=started_at,
            ended_at=ended_at,
        )
    )
    video_path.write_bytes(b"different second recording")
    second = asyncio.run(
        register_and_analyze_video(
            db,
            analyzer,
            file_path=video_path,
            started_at=started_at,
            ended_at=ended_at,
        )
    )

    assert first.segment.id != second.segment.id
    assert first.segment.recording_key != second.segment.recording_key
    assert second.episode.observations[0].description == "Drugie nagranie."
    assert db.execute("SELECT COUNT(*) FROM video_segments").fetchone()[0] == 2
    assert len(analyzer.calls) == 2


def test_high_confidence_game_detection_links_existing_profile(db, tmp_path):
    game = _game_document(db)
    video_path = tmp_path / "detected-game.mp4"
    video_path.write_bytes(b"gameplay clip")
    started_at = datetime(2026, 8, 29, 18, 0, tzinfo=timezone.utc)
    episode = _episode("Eksploracja.").model_copy(
        update={"possible_game": "  big WALK ", "confidence": 0.95}
    )

    outcome = asyncio.run(
        register_and_analyze_video(
            db,
            StubVideoAnalyzer([episode]),
            file_path=video_path,
            started_at=started_at,
            ended_at=started_at + timedelta(minutes=2),
        )
    )

    assert outcome.segment.game_document_id == game.id


def test_low_confidence_game_detection_does_not_link_profile(db, tmp_path):
    _game_document(db)
    video_path = tmp_path / "uncertain-game.mp4"
    video_path.write_bytes(b"gameplay clip")
    started_at = datetime(2026, 8, 29, 18, 0, tzinfo=timezone.utc)
    episode = _episode("Eksploracja.").model_copy(
        update={"possible_game": "Big Walk", "confidence": 0.6}
    )

    outcome = asyncio.run(
        register_and_analyze_video(
            db,
            StubVideoAnalyzer([episode]),
            file_path=video_path,
            started_at=started_at,
            ended_at=started_at + timedelta(minutes=2),
        )
    )

    assert outcome.segment.game_document_id is None


@pytest.mark.parametrize(
    "episode",
    [
        EpisodeAnalysis(
            observations=[
                EpisodeObservation(
                    start_seconds=119,
                    end_seconds=121,
                    activity="Za późno.",
                    description="Za późno.",
                )
            ],
        ),
        EpisodeAnalysis(
            observations=[
                EpisodeObservation(
                    start_seconds=121,
                    end_seconds=121,
                    activity="Odczyt tekstu.",
                    description="Za późno.",
                    visible_text=["Za późno"],
                )
            ],
        ),
    ],
)
def test_episode_timestamps_outside_segment_are_dropped(
    db,
    tmp_path,
    episode,
):
    segment = _segment(db, tmp_path)
    analyzer = StubVideoAnalyzer([episode])

    outcome = asyncio.run(analyze_video_segment(db, analyzer, segment.id))

    assert outcome.segment.status is VideoSegmentStatus.COMPLETED
    assert outcome.episode.observations == []
    assert outcome.episode.uncertainties[-1] == (
        "Number of skipped observations with timestamps exceeding the "
        "clip length: 1 (clip time: 120.000 s)."
    )
