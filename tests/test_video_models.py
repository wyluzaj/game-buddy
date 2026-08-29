import pytest
from pydantic import ValidationError

from buddy.video.models import VideoEvent


def test_video_event_rejects_an_inverted_time_range():
    """An event cannot point backwards inside its source video."""
    with pytest.raises(ValidationError, match="cannot end before it starts"):
        VideoEvent(
            start_seconds=20,
            end_seconds=10,
            description="Znaleziono drzwi.",
        )
