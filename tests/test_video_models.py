import pytest
from pydantic import ValidationError

from buddy.config import VIDEO
from buddy.video.models import (
    EpisodeAnalysis,
    EpisodeObservation,
    VideoEvent,
    VideoOverview,
)


def test_video_event_rejects_an_inverted_time_range():
    """An event cannot point backwards inside its source video."""
    with pytest.raises(ValidationError, match="cannot end before it starts"):
        VideoEvent(
            start_seconds=20,
            end_seconds=10,
            description="Znaleziono drzwi.",
        )


def test_video_overview_rejects_unbounded_event_lists():
    event = VideoEvent(
        start_seconds=0,
        end_seconds=1,
        description="Repeated event",
    )

    with pytest.raises(ValidationError):
        VideoOverview(
            summary="Too many events",
            events=[event] * (VIDEO.max_overview_events + 1),
        )


def test_episode_observation_rejects_an_inverted_time_range():
    with pytest.raises(ValidationError, match="cannot end before it starts"):
        EpisodeObservation(
            start_seconds=20,
            end_seconds=10,
            activity="Walking through a forest.",
            description="The player follows a path between trees.",
        )


def test_episode_analysis_bounds_searchable_observations():
    observation = EpisodeObservation(
        start_seconds=0,
        end_seconds=1,
        activity="Walking.",
        description="The player moves along a path.",
    )

    with pytest.raises(ValidationError):
        EpisodeAnalysis(
            observations=[observation] * (VIDEO.max_episode_observations + 1)
        )
