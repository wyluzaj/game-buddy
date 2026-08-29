"""Provider-neutral contract for turning video segments into text reports."""

from pathlib import Path
from typing import Protocol, runtime_checkable

from buddy.video_models import VideoOverview


class VideoAnalysisError(RuntimeError):
    """Base error raised by a video analysis provider."""


class InvalidVideoFileError(VideoAnalysisError):
    """The local video cannot be submitted for analysis."""


class VideoProviderError(VideoAnalysisError):
    """The external video provider could not produce a valid report."""


@runtime_checkable
class VideoAnalyzer(Protocol):
    """Minimal contract implemented by Gemini and future providers."""

    @property
    def provider(self) -> str:
        """Stable provider identifier stored with generated reports."""
        ...

    @property
    def model_name(self) -> str:
        """Provider-native model identifier stored with generated reports."""
        ...

    async def analyze_overview(self, video_path: Path) -> VideoOverview:
        """Create one structured fast report for a completed local video."""
        ...
