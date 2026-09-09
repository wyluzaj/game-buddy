"""Domain types shared by video capture, analysis, and persistence."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from buddy.config import VIDEO

BoundedObservationText = Annotated[str, Field(min_length=1, max_length=500)]


class VideoSegmentStatus(str, Enum):
    """Lifecycle of a recorded segment's overview analysis."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoReportMode(str, Enum):
    """Kinds of persisted analysis produced for a video segment."""

    OVERVIEW = "overview"
    EPISODE = "episode"
    DETAIL = "detail"


class VideoEventImportance(str, Enum):
    """How useful an observed event may be to the game companion."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EpisodeObservation(BaseModel):
    """One continuous, searchable part of a gameplay episode timeline."""

    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    activity: str = Field(min_length=1, max_length=500)
    location: str | None = Field(default=None, max_length=500)
    description: str = Field(min_length=1, max_length=2_000)
    visible_objects: list[BoundedObservationText] = Field(
        default_factory=list,
        max_length=VIDEO.max_observation_objects,
    )
    visible_text: list[BoundedObservationText] = Field(
        default_factory=list,
        max_length=VIDEO.max_observation_texts,
    )
    uncertainties: list[BoundedObservationText] = Field(
        default_factory=list,
        max_length=VIDEO.max_observation_uncertainties,
    )

    @model_validator(mode="after")
    def validate_time_range(self) -> "EpisodeObservation":
        """Reject observations that point backwards inside the episode."""
        if self.end_seconds < self.start_seconds:
            raise ValueError("An episode observation cannot end before it starts.")
        return self


class EpisodeAnalysis(BaseModel):
    """Detailed first-pass timeline produced for one complete episode."""

    observations: list[EpisodeObservation] = Field(
        default_factory=list,
        max_length=VIDEO.max_episode_observations,
    )
    uncertainties: list[BoundedObservationText] = Field(
        default_factory=list,
        max_length=VIDEO.max_uncertainties,
    )
    possible_game: str | None = Field(default=None, max_length=500)
    confidence: float | None = Field(default=None, ge=0, le=1)


class VideoEvent(BaseModel):
    """One timestamped event detected inside a video segment."""

    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    description: str = Field(min_length=1)
    importance: VideoEventImportance = VideoEventImportance.MEDIUM

    @model_validator(mode="after")
    def validate_time_range(self) -> "VideoEvent":
        """Reject inverted event ranges before they reach persistence."""
        if self.end_seconds < self.start_seconds:
            raise ValueError("A video event cannot end before it starts.")
        return self


class VisibleTextObservation(BaseModel):
    """Text read from the game screen at an approximate timestamp."""

    timestamp_seconds: float = Field(ge=0)
    text: str = Field(min_length=1)


class VideoOverview(BaseModel):
    """Structured fast analysis of one completed video segment."""

    summary: str = Field(min_length=1)
    events: list[VideoEvent] = Field(
        default_factory=list,
        max_length=VIDEO.max_overview_events,
    )
    visible_text: list[VisibleTextObservation] = Field(
        default_factory=list,
        max_length=VIDEO.max_visible_text_observations,
    )
    uncertainties: list[str] = Field(
        default_factory=list,
        max_length=VIDEO.max_uncertainties,
    )
    possible_game: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class VideoDetailObservation(BaseModel):
    """Question-relevant evidence at an absolute offset in one segment."""

    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    description: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_time_range(self) -> "VideoDetailObservation":
        if self.end_seconds < self.start_seconds:
            raise ValueError("A detailed observation cannot end before it starts.")
        return self


class VideoDetailAnalysis(BaseModel):
    """Structured answer produced by re-reading one requested video range."""

    answer: str = Field(min_length=1)
    observations: list[VideoDetailObservation] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


class VideoDetailAnswer(BaseModel):
    """Bounded persisted result exposed by the detail-analysis tool."""

    answer: str
    sources: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    reused_existing_report: bool = False


@dataclass(frozen=True, slots=True)
class VideoSegment:
    """Metadata for a locally stored, completed video fragment."""

    id: int
    conversation_id: int | None
    game_document_id: int | None
    file_path: Path
    recording_key: str
    started_at: datetime
    ended_at: datetime
    duration_seconds: float
    status: VideoSegmentStatus
    error: str | None
    created_at: datetime
    updated_at: datetime

    @property
    def source_uri(self) -> str:
        """Stable application URI that does not depend on an API provider."""
        return f"video://segment/{self.id}"


@dataclass(frozen=True, slots=True)
class StoredEpisodeObservation:
    """One normalized, searchable observation persisted from an episode."""

    id: int
    report_id: int
    segment_id: int
    start_seconds: float
    end_seconds: float
    activity: str
    location: str | None
    description: str
    visible_objects: list[str]
    visible_text: list[str]
    uncertainties: list[str]

    @property
    def source_uri(self) -> str:
        """Return the exact application-owned range supporting this record."""
        start = _format_seconds(self.start_seconds)
        end = _format_seconds(self.end_seconds)
        return f"video://segment/{self.segment_id}?t={start}-{end}"


@dataclass(frozen=True, slots=True)
class StoredEpisodeSummary:
    """One durable brief description used for chronological video recall."""

    segment_id: int
    conversation_id: int | None
    game_document_id: int | None
    game_title: str | None
    started_at: datetime
    ended_at: datetime
    summary: str

    @property
    def source_uri(self) -> str:
        return f"video://segment/{self.segment_id}"


@dataclass(frozen=True, slots=True)
class VideoReport:
    """One persisted structured analysis of a video segment."""

    id: int
    segment_id: int
    mode: VideoReportMode
    question: str | None
    request_key: str | None
    provider: str
    model_name: str
    content: dict[str, object]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class VideoContextUpdate:
    """Persisted decision made from one overview by the video context butler."""

    id: int
    report_id: int
    changed: bool
    context: str
    sources: list[str]
    uncertainty: str | None
    created_at: datetime
    injected_at: datetime | None
    episode_summary: str = ""


def episode_from_legacy_overview(
    overview: VideoOverview,
    *,
    duration_seconds: float,
) -> EpisodeAnalysis:
    """Project a stored legacy overview into the detailed episode contract."""
    observations = [
        EpisodeObservation(
            start_seconds=event.start_seconds,
            end_seconds=event.end_seconds,
            activity=event.description,
            description=event.description,
        )
        for event in overview.events
    ]
    observations.extend(
        EpisodeObservation(
            start_seconds=text.timestamp_seconds,
            end_seconds=text.timestamp_seconds,
            activity="Reading on-screen text.",
            description=f"Visible on-screen text: {text.text}",
            visible_text=[text.text],
        )
        for text in overview.visible_text
    )
    if not observations:
        observations.append(
            EpisodeObservation(
                start_seconds=0,
                end_seconds=duration_seconds,
                activity="General episode activity.",
                description=overview.summary,
            )
        )
    observations.sort(key=lambda item: (item.start_seconds, item.end_seconds))
    return EpisodeAnalysis(
        observations=observations[: VIDEO.max_episode_observations],
        uncertainties=overview.uncertainties,
        possible_game=overview.possible_game,
        confidence=overview.confidence,
    )


def _format_seconds(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")
