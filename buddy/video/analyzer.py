"""Provider-neutral contract for turning video segments into text reports."""

from pathlib import Path
from typing import Protocol, runtime_checkable

from buddy.video.models import EpisodeAnalysis, VideoDetailAnalysis


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

    async def analyze_episode(self, video_path: Path) -> EpisodeAnalysis:
        """Create a detailed searchable timeline for a completed local video."""
        ...

    async def analyze_detail(
        self,
        video_path: Path,
        *,
        question: str,
        start_seconds: float,
        end_seconds: float,
    ) -> VideoDetailAnalysis:
        """Answer one question by closely inspecting a bounded video range."""
        ...


@runtime_checkable
class VideoDetailAnalyzer(Protocol):
    """Narrow provider contract required by the on-demand detail service."""

    @property
    def provider(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    async def analyze_detail(
        self,
        video_path: Path,
        *,
        question: str,
        start_seconds: float,
        end_seconds: float,
    ) -> VideoDetailAnalysis: ...
