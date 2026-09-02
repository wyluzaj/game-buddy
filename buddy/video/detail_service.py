"""Secure, idempotent detailed analysis of application-owned video sources."""

import asyncio
import re
import sqlite3
from dataclasses import dataclass
from hashlib import sha256

from pydantic import ValidationError

from buddy.config import VIDEO
from buddy.video.analyzer import VideoDetailAnalyzer
from buddy.video.models import (
    VideoDetailAnalysis,
    VideoDetailAnswer,
    VideoReport,
    VideoReportMode,
    VideoSegmentStatus,
)
from buddy.video.storage import (
    get_video_report_by_request_key,
    get_video_segment,
    save_video_report,
)

_VIDEO_SOURCE_PATTERN = re.compile(
    r"^video://segment/(?P<segment_id>[1-9]\d*)"
    r"(?:\?t=(?P<start>\d+(?:\.\d+)?)-(?P<end>\d+(?:\.\d+)?))?$"
)
_DETAIL_REQUEST_LOCKS: dict[str, asyncio.Lock] = {}


class VideoDetailStateError(RuntimeError):
    """A stored detailed report or requested range is inconsistent."""


@dataclass(frozen=True, slots=True)
class ParsedVideoSource:
    segment_id: int
    start_seconds: float
    end_seconds: float
    canonical_uri: str


async def analyze_video_detail_source(
    conn: sqlite3.Connection,
    analyzer: VideoDetailAnalyzer,
    *,
    source_uri: str,
    question: str,
) -> VideoDetailAnswer:
    """Resolve one internal URI, re-read its clip, and persist the answer."""
    normalized_question = " ".join(question.split())
    if not normalized_question:
        raise ValueError("A detailed video question cannot be empty.")
    if len(normalized_question) > VIDEO.detail_question_chars:
        raise ValueError(
            f"A detailed video question cannot exceed "
            f"{VIDEO.detail_question_chars} characters."
        )

    parsed = parse_video_source(conn, source_uri)
    segment = get_video_segment(conn, parsed.segment_id)
    if segment is None:
        raise ValueError(f"Unknown video segment: {parsed.segment_id}.")
    if segment.status is not VideoSegmentStatus.COMPLETED:
        raise VideoDetailStateError(
            "A video segment needs a completed overview before detailed analysis."
        )

    request_key = _detail_request_key(
        segment.id,
        parsed.canonical_uri,
        normalized_question,
    )
    lock = _DETAIL_REQUEST_LOCKS.setdefault(request_key, asyncio.Lock())
    async with lock:
        existing = get_video_report_by_request_key(conn, request_key)
        if existing is not None:
            return _answer_from_report(existing, reused=True)

        analysis = await analyzer.analyze_detail(
            segment.file_path,
            question=normalized_question,
            start_seconds=parsed.start_seconds,
            end_seconds=parsed.end_seconds,
        )
        analysis, sources = _sanitize_and_build_sources(analysis, parsed)
        content = analysis.model_dump(mode="json")
        content["requested_source"] = parsed.canonical_uri
        content["sources"] = sources
        try:
            report = save_video_report(
                conn,
                segment_id=segment.id,
                mode=VideoReportMode.DETAIL,
                question=normalized_question,
                request_key=request_key,
                provider=analyzer.provider,
                model_name=analyzer.model_name,
                content=content,
            )
        except sqlite3.IntegrityError:
            report = get_video_report_by_request_key(conn, request_key)
            if report is None:
                raise
            return _answer_from_report(report, reused=True)
        return _answer_from_report(report, reused=False)


def parse_video_source(
    conn: sqlite3.Connection,
    source_uri: str,
) -> ParsedVideoSource:
    """Accept only internal segment URIs and ranges within the stored duration."""
    match = _VIDEO_SOURCE_PATTERN.fullmatch(source_uri.strip())
    if match is None:
        raise ValueError(
            "source_uri must look like video://segment/12 or "
            "video://segment/12?t=10-20."
        )
    segment_id = int(match.group("segment_id"))
    segment = get_video_segment(conn, segment_id)
    if segment is None:
        raise ValueError(f"Unknown video segment: {segment_id}.")
    start = float(match.group("start") or 0)
    end = float(match.group("end") or segment.duration_seconds)
    if end < start:
        raise ValueError("The video source range cannot end before it starts.")
    if end > segment.duration_seconds + 0.001:
        raise ValueError("The video source range exceeds the segment duration.")
    canonical = segment.source_uri
    if match.group("start") is not None:
        canonical += f"?t={_format_seconds(start)}-{_format_seconds(end)}"
    return ParsedVideoSource(
        segment_id=segment_id,
        start_seconds=start,
        end_seconds=end,
        canonical_uri=canonical,
    )


def _detail_request_key(
    segment_id: int,
    source_uri: str,
    question: str,
) -> str:
    raw = f"{segment_id}\0{source_uri}\0{question.casefold()}"
    return sha256(raw.encode()).hexdigest()


def _sanitize_and_build_sources(
    analysis: VideoDetailAnalysis,
    requested: ParsedVideoSource,
) -> tuple[VideoDetailAnalysis, list[str]]:
    """Drop provider evidence outside the requested clip without expanding it."""
    sources: list[str] = []
    valid_observations = []
    dropped_count = 0
    base = f"video://segment/{requested.segment_id}"
    for observation in analysis.observations:
        if (
            observation.start_seconds < requested.start_seconds - 0.001
            or observation.end_seconds > requested.end_seconds + 0.001
        ):
            dropped_count += 1
            continue
        valid_observations.append(observation)
        sources.append(
            f"{base}?t={_format_seconds(observation.start_seconds)}-"
            f"{_format_seconds(observation.end_seconds)}"
        )

    if dropped_count == 0:
        return analysis, list(dict.fromkeys(sources)) or [requested.canonical_uri]

    range_label = (
        f"{_format_seconds(requested.start_seconds)}-"
        f"{_format_seconds(requested.end_seconds)}s"
    )
    uncertainty = (
        f"Odrzucono {dropped_count} obserwację(-e) spoza żądanego zakresu "
        f"{range_label}."
    )
    answer = analysis.answer
    if not valid_observations:
        answer = "Nie udało się potwierdzić odpowiedzi w żądanym zakresie nagrania."
    sanitized = analysis.model_copy(
        update={
            "answer": answer,
            "observations": valid_observations,
            "uncertainties": [*analysis.uncertainties, uncertainty],
        }
    )
    return sanitized, list(dict.fromkeys(sources)) or [requested.canonical_uri]


def _answer_from_report(
    report: VideoReport,
    *,
    reused: bool,
) -> VideoDetailAnswer:
    try:
        analysis = VideoDetailAnalysis.model_validate(report.content)
        raw_sources = report.content["sources"]
        if not isinstance(raw_sources, list) or not all(
            isinstance(source, str) and source.startswith("video://segment/")
            for source in raw_sources
        ):
            raise ValueError("Detailed report sources are invalid.")
        sources = list(dict.fromkeys(raw_sources))
    except (KeyError, TypeError, ValidationError, ValueError) as exc:
        raise VideoDetailStateError(
            f"Detailed video report {report.id} is invalid."
        ) from exc
    return VideoDetailAnswer(
        answer=analysis.answer,
        sources=sources,
        uncertainties=analysis.uncertainties,
        reused_existing_report=reused,
    )


def _format_seconds(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")
