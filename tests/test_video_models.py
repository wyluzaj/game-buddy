import pytest
from pydantic import ValidationError

from buddy.config import VIDEO
from buddy.video.models import VideoEvent, VideoOverview


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
