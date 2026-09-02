import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from buddy.database import create_conversation
from buddy.video.detail_service import (
    analyze_video_detail_source,
    parse_video_source,
)
from buddy.video.models import (
    VideoDetailAnalysis,
    VideoDetailObservation,
    VideoOverview,
    VideoReportMode,
)
from buddy.video.storage import (
    claim_video_segment_for_analysis,
    create_video_segment,
    get_video_reports,
    save_video_report,
)


class StubDetailAnalyzer:
    provider = "stub"
    model_name = "stub-detail"

    def __init__(self, results: list[VideoDetailAnalysis]) -> None:
        self.results = results
        self.calls: list[tuple[Path, str, float, float]] = []

    async def analyze_detail(
        self,
        video_path: Path,
        *,
        question: str,
        start_seconds: float,
        end_seconds: float,
    ) -> VideoDetailAnalysis:
        self.calls.append((video_path, question, start_seconds, end_seconds))
        return self.results.pop(0)


def _completed_segment(db, tmp_path):
    conversation_id = create_conversation(db)
    video_path = tmp_path / "segment.mp4"
    video_path.write_bytes(b"video")
    started_at = datetime(2026, 8, 29, 18, 0, tzinfo=timezone.utc)
    segment = create_video_segment(
        db,
        file_path=video_path,
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=30),
        conversation_id=conversation_id,
    )
    assert claim_video_segment_for_analysis(db, segment.id) is not None
    save_video_report(
        db,
        segment_id=segment.id,
        mode=VideoReportMode.OVERVIEW,
        provider="stub",
        model_name="stub-overview",
        content=VideoOverview(summary="Ogólny raport.").model_dump(mode="json"),
    )
    return segment


def test_detail_analysis_is_source_bounded_persisted_and_reused(db, tmp_path):
    segment = _completed_segment(db, tmp_path)
    detail = VideoDetailAnalysis(
        answer="Na ścianie był symbol lisa.",
        observations=[
            VideoDetailObservation(
                start_seconds=12,
                end_seconds=14,
                description="Symbol lisa na ścianie.",
            )
        ],
    )
    analyzer = StubDetailAnalyzer([detail])
    source = f"{segment.source_uri}?t=10-20"

    first = asyncio.run(
        analyze_video_detail_source(
            db,
            analyzer,
            source_uri=source,
            question="Co było na ścianie?",
        )
    )
    second = asyncio.run(
        analyze_video_detail_source(
            db,
            analyzer,
            source_uri=source,
            question="Co było na ścianie?",
        )
    )

    assert first.answer == "Na ścianie był symbol lisa."
    assert first.sources == [f"{segment.source_uri}?t=12-14"]
    assert first.reused_existing_report is False
    assert second.reused_existing_report is True
    assert analyzer.calls == [
        (segment.file_path, "Co było na ścianie?", 10.0, 20.0)
    ]
    assert len(
        get_video_reports(db, segment.id, mode=VideoReportMode.DETAIL)
    ) == 1


def test_video_source_parser_rejects_external_unknown_and_oversized_ranges(
    db,
    tmp_path,
):
    segment = _completed_segment(db, tmp_path)

    with pytest.raises(ValueError, match="source_uri"):
        parse_video_source(db, "C:/private/video.mp4")
    with pytest.raises(ValueError, match="Unknown video segment"):
        parse_video_source(db, "video://segment/999")
    with pytest.raises(ValueError, match="exceeds"):
        parse_video_source(db, f"{segment.source_uri}?t=10-40")


def test_out_of_range_detail_evidence_is_removed_and_reported(db, tmp_path):
    segment = _completed_segment(db, tmp_path)
    analyzer = StubDetailAnalyzer(
        [
            VideoDetailAnalysis(
                answer="Nieprawidłowy zakres.",
                observations=[
                    VideoDetailObservation(
                        start_seconds=5,
                        end_seconds=14,
                        description="Dowód spoza zakresu.",
                    )
                ],
            )
        ]
    )

    result = asyncio.run(
        analyze_video_detail_source(
            db,
            analyzer,
            source_uri=f"{segment.source_uri}?t=10-20",
            question="Co widać?",
        )
    )

    assert result.answer == (
        "Nie udało się potwierdzić odpowiedzi w żądanym zakresie nagrania."
    )
    assert result.sources == [f"{segment.source_uri}?t=10-20"]
    assert "Odrzucono 1 obserwację" in result.uncertainties[-1]
    reports = get_video_reports(
        db,
        segment.id,
        mode=VideoReportMode.DETAIL,
    )
    assert len(reports) == 1
    assert reports[0].content["observations"] == []


def test_valid_detail_evidence_survives_invalid_provider_timestamp(db, tmp_path):
    segment = _completed_segment(db, tmp_path)
    analyzer = StubDetailAnalyzer(
        [
            VideoDetailAnalysis(
                answer="W zakresie był czerwony balon.",
                observations=[
                    VideoDetailObservation(
                        start_seconds=5,
                        end_seconds=9,
                        description="Dowód spoza zakresu.",
                    ),
                    VideoDetailObservation(
                        start_seconds=12,
                        end_seconds=13,
                        description="Czerwony balon.",
                    ),
                ],
            )
        ]
    )

    result = asyncio.run(
        analyze_video_detail_source(
            db,
            analyzer,
            source_uri=f"{segment.source_uri}?t=10-20",
            question="Gdzie był czerwony balon?",
        )
    )

    assert result.answer == "W zakresie był czerwony balon."
    assert result.sources == [f"{segment.source_uri}?t=12-13"]
    assert "Odrzucono 1 obserwację" in result.uncertainties[-1]


def test_concurrent_identical_detail_requests_call_provider_once(db, tmp_path):
    segment = _completed_segment(db, tmp_path)
    analyzer = StubDetailAnalyzer(
        [
            VideoDetailAnalysis(
                answer="Na ścianie był czerwony symbol.",
                observations=[
                    VideoDetailObservation(
                        start_seconds=10,
                        end_seconds=11,
                        description="Czerwony symbol.",
                    )
                ],
            )
        ]
    )

    async def scenario():
        return await asyncio.gather(
            analyze_video_detail_source(
                db,
                analyzer,
                source_uri=f"{segment.source_uri}?t=10-20",
                question="Co było na ścianie?",
            ),
            analyze_video_detail_source(
                db,
                analyzer,
                source_uri=f"{segment.source_uri}?t=10-20",
                question="Co było na ścianie?",
            ),
        )

    first, second = asyncio.run(scenario())

    assert len(analyzer.calls) == 1
    assert first.reused_existing_report is False
    assert second.reused_existing_report is True
    assert first.answer == second.answer
    detail_count = db.execute(
        "SELECT COUNT(*) FROM video_reports WHERE mode = 'detail'"
    ).fetchone()[0]
    assert detail_count == 1
