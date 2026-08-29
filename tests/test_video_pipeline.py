import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from buddy.video_context_butler import VideoContextButlerResult
from buddy.video_models import VideoOverview
from buddy.video_pipeline import process_video_file
from buddy.video_storage import (
    get_video_context_update_for_report,
    get_video_reports,
)


class StubVideoAnalyzer:
    provider = "stub"
    model_name = "stub-video-model"

    def __init__(self, overview: VideoOverview) -> None:
        self.overview = overview
        self.calls: list[Path] = []

    async def analyze_overview(self, video_path: Path) -> VideoOverview:
        self.calls.append(video_path)
        return self.overview


class StubRunResult:
    def __init__(self, output: VideoContextButlerResult) -> None:
        self.output = output


class StubContextButler:
    def __init__(self, outputs: list[VideoContextButlerResult | Exception]) -> None:
        self.outputs = outputs
        self.prompts: list[str] = []

    async def run(self, prompt: str, *, usage=None) -> StubRunResult:
        self.prompts.append(prompt)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return StubRunResult(output)


def _video(tmp_path: Path) -> tuple[Path, datetime, datetime]:
    video_path = tmp_path / "segment.mp4"
    video_path.write_bytes(b"gameplay")
    started_at = datetime(2026, 8, 29, 18, 0, tzinfo=timezone.utc)
    return video_path, started_at, started_at + timedelta(minutes=2)


def test_pipeline_persists_overview_then_compact_context(db, tmp_path):
    """One call produces both durable layers and links them by report ID."""
    video_path, started_at, ended_at = _video(tmp_path)
    analyzer = StubVideoAnalyzer(VideoOverview(summary="Otwarto mapę."))
    butler = StubContextButler(
        [
            VideoContextButlerResult(
                changed=True,
                context="Gracz sprawdza mapę.",
                sources=["video://segment/1"],
            )
        ]
    )

    outcome = asyncio.run(
        process_video_file(
            db,
            analyzer,
            butler,
            file_path=video_path,
            started_at=started_at,
            ended_at=ended_at,
        )
    )

    assert outcome.analysis.overview.summary == "Otwarto mapę."
    assert outcome.context_update.context == "Gracz sprawdza mapę."
    assert outcome.context_update.report_id == outcome.analysis.report.id
    assert get_video_context_update_for_report(
        db, outcome.analysis.report.id
    ) == outcome.context_update


def test_retry_after_butler_failure_reuses_paid_overview(db, tmp_path):
    """A failed reduction can resume without uploading the video a second time."""
    video_path, started_at, ended_at = _video(tmp_path)
    analyzer = StubVideoAnalyzer(VideoOverview(summary="Znaleziono drzwi."))
    failure = RuntimeError("temporary Butler failure")
    failing_butler = StubContextButler([failure])

    with pytest.raises(RuntimeError, match="temporary Butler failure"):
        asyncio.run(
            process_video_file(
                db,
                analyzer,
                failing_butler,
                file_path=video_path,
                started_at=started_at,
                ended_at=ended_at,
            )
        )

    retry_butler = StubContextButler(
        [
            VideoContextButlerResult(
                changed=True,
                context="Przed graczem są zamknięte drzwi.",
                sources=["video://segment/1"],
            )
        ]
    )
    outcome = asyncio.run(
        process_video_file(
            db,
            analyzer,
            retry_butler,
            file_path=video_path,
            started_at=started_at,
            ended_at=ended_at,
        )
    )

    assert outcome.analysis.reused_existing_report is True
    assert outcome.context_update.changed is True
    assert analyzer.calls == [video_path]
    assert len(get_video_reports(db, outcome.analysis.segment.id)) == 1


def test_repeated_pipeline_call_reuses_both_results(db, tmp_path):
    """A completed file does not spend either analyzer again."""
    video_path, started_at, ended_at = _video(tmp_path)
    analyzer = StubVideoAnalyzer(VideoOverview(summary="Spokojna eksploracja."))
    butler = StubContextButler(
        [VideoContextButlerResult(changed=False, uncertainty="Brak zmian.")]
    )

    first = asyncio.run(
        process_video_file(
            db,
            analyzer,
            butler,
            file_path=video_path,
            started_at=started_at,
            ended_at=ended_at,
        )
    )
    second = asyncio.run(
        process_video_file(
            db,
            analyzer,
            butler,
            file_path=video_path,
            started_at=started_at,
            ended_at=ended_at,
        )
    )

    assert second.analysis.reused_existing_report is True
    assert second.context_update == first.context_update
    assert analyzer.calls == [video_path]
    assert len(butler.prompts) == 1
