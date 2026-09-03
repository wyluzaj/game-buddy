"""Delegate that maintains a bounded replacement snapshot of observed gameplay."""

import json
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunUsage
from pydantic_ai.models import Model as PydanticModel

from buddy.config import MODELS, VIDEO, build_model
from buddy.prompts import (
    VIDEO_CONTEXT_BUTLER_INSTRUCTIONS,
    VIDEO_CONTEXT_BUTLER_PROMPT,
)
from buddy.video.models import EpisodeAnalysis, VideoContextUpdate, VideoSegment


class VideoContextButlerResult(BaseModel):
    """Durable episode summary plus an optional replacement gameplay state."""

    episode_summary: str = Field(min_length=1, max_length=VIDEO.episode_summary_chars)
    changed: bool
    context: str = Field(default="", max_length=VIDEO.context_chars)
    sources: list[str] = Field(
        default_factory=list,
        max_length=VIDEO.context_sources,
    )
    uncertainty: str | None = Field(
        default=None,
        max_length=VIDEO.context_uncertainty_chars,
    )


class VideoContextButlerRunResult(Protocol):
    @property
    def output(self) -> VideoContextButlerResult: ...


class VideoContextButler(Protocol):
    async def run(
        self,
        prompt: str,
        *,
        usage: RunUsage | None = None,
    ) -> VideoContextButlerRunResult: ...


@dataclass(frozen=True, slots=True)
class BuiltVideoContextPrompt:
    """Prompt plus the exact source allow-list used to validate its answer."""

    prompt: str
    allowed_sources: set[str]


def build_video_context_butler(
    model: str | PydanticModel | None = None,
) -> Agent[None, VideoContextButlerResult]:
    """Build a tool-less delegate using its explicit configured model."""
    butler_model = (
        model if model is not None else build_model(MODELS.video_context_butler)
    )
    return Agent[None, VideoContextButlerResult](
        model=butler_model,
        deps_type=type(None),
        name="video_context_butler",
        instructions=VIDEO_CONTEXT_BUTLER_INSTRUCTIONS,
        output_type=VideoContextButlerResult,
    )


def build_video_context_prompt(
    segment: VideoSegment,
    episode: EpisodeAnalysis,
    previous: VideoContextUpdate | None,
) -> BuiltVideoContextPrompt:
    """Attach stable source URIs to every selectable piece of evidence."""
    base_source = segment.source_uri
    observations = [
        {
            "source": _timestamp_source(
                base_source,
                observation.start_seconds,
                observation.end_seconds,
            ),
            **observation.model_dump(mode="json"),
        }
        for observation in episode.observations
    ]
    new_evidence = {
        "segment_source": base_source,
        "possible_game": episode.possible_game,
        "confidence": episode.confidence,
        "observations": observations,
        "uncertainties": episode.uncertainties,
    }
    previous_state = (
        {"context": previous.context, "sources": previous.sources}
        if previous is not None
        else {"context": "", "sources": []}
    )
    allowed_sources = {base_source}
    allowed_sources.update(
        str(observation["source"]) for observation in observations
    )
    if previous is not None:
        allowed_sources.update(previous.sources)
    return BuiltVideoContextPrompt(
        prompt=VIDEO_CONTEXT_BUTLER_PROMPT.format(
            previous_context=json.dumps(
                previous_state,
                ensure_ascii=False,
                indent=2,
            ),
            episode_analysis=json.dumps(
                new_evidence,
                ensure_ascii=False,
                indent=2,
            ),
        ),
        allowed_sources=allowed_sources,
    )


def validate_video_context_result(
    result: VideoContextButlerResult,
    *,
    allowed_sources: set[str],
) -> VideoContextButlerResult:
    """Prevent invented video links and normalize no-change decisions."""
    summary = result.episode_summary.strip()
    if not result.changed:
        return result.model_copy(
            update={
                "episode_summary": summary,
                "context": "",
                "sources": [],
            }
        )

    context = result.context.strip()
    sources = list(dict.fromkeys(result.sources))
    if not context or not sources:
        return VideoContextButlerResult(
            episode_summary=summary,
            changed=False,
            uncertainty="Video Context Butler nie uzasadnił zmiany źródłami.",
        )
    if any(source not in allowed_sources for source in sources):
        return VideoContextButlerResult(
            episode_summary=summary,
            changed=False,
            uncertainty="Video Context Butler wskazał nieznane źródło wideo.",
        )
    return result.model_copy(
        update={
            "episode_summary": summary,
            "context": context,
            "sources": sources,
        }
    )


def _timestamp_source(base_source: str, start: float, end: float) -> str:
    return f"{base_source}?t={_format_seconds(start)}-{_format_seconds(end)}"


def _format_seconds(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")
