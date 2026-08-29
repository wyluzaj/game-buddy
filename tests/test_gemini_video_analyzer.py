import asyncio
from pathlib import Path

import pytest
from google.genai import types

from buddy.video.analyzer import (
    InvalidVideoFileError,
    VideoAnalyzer,
    VideoProviderError,
)
from buddy.video.gemini import GeminiVideoAnalyzer
from buddy.video.models import (
    VideoDetailAnalysis,
    VideoDetailObservation,
    VideoEvent,
    VideoOverview,
)


class FakeFiles:
    def __init__(self, states: list[types.FileState]) -> None:
        self.states = states
        self.uploaded_path: Path | None = None
        self.upload_config: types.UploadFileConfig | None = None
        self.get_calls = 0
        self.deleted_names: list[str] = []

    async def upload(
        self,
        *,
        file: str | Path,
        config: types.UploadFileConfig,
    ) -> types.File:
        self.uploaded_path = Path(file)
        self.upload_config = config
        return self._next_file()

    async def get(self, *, name: str) -> types.File:
        assert name == "files/test-video"
        self.get_calls += 1
        return self._next_file()

    async def delete(self, *, name: str) -> object:
        self.deleted_names.append(name)
        return object()

    def _next_file(self) -> types.File:
        state = self.states.pop(0)
        return types.File(
            name="files/test-video",
            uri="https://example.invalid/test-video",
            mime_type="video/mp4",
            state=state,
        )


class FakeModels:
    def __init__(
        self,
        response: types.GenerateContentResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.model_name: str | None = None
        self.contents: list[object] | None = None
        self.config: types.GenerateContentConfig | None = None

    async def generate_content(
        self,
        *,
        model: str,
        contents: list[object],
        config: types.GenerateContentConfig,
    ) -> types.GenerateContentResponse:
        self.model_name = model
        self.contents = contents
        self.config = config
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


class FakeAsyncClient:
    def __init__(self, files: FakeFiles, models: FakeModels) -> None:
        self.files = files
        self.models = models
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class FakeClient:
    def __init__(self, files: FakeFiles, models: FakeModels) -> None:
        self.aio = FakeAsyncClient(files, models)


def test_gemini_analyzer_uploads_waits_parses_and_cleans_up(tmp_path):
    """A processing upload becomes one validated overview and is then deleted."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake video content")
    overview = VideoOverview(
        summary="Użytkowniczka znalazła zamknięte drzwi.",
        events=[
            VideoEvent(
                start_seconds=12,
                end_seconds=18,
                description="Znaleziono zamknięte drzwi.",
            )
        ],
        confidence=0.8,
    )
    files = FakeFiles([types.FileState.PROCESSING, types.FileState.ACTIVE])
    models = FakeModels(types.GenerateContentResponse(parsed=overview))
    analyzer = GeminiVideoAnalyzer(
        client=FakeClient(files, models),
        poll_interval_seconds=0,
    )

    result = asyncio.run(analyzer.analyze_overview(clip))

    assert isinstance(analyzer, VideoAnalyzer)
    assert analyzer.provider == "google"
    assert analyzer.model_name == "gemini-3.1-flash-lite"
    assert result == overview
    assert files.uploaded_path == clip
    assert files.upload_config is not None
    assert files.upload_config.mime_type == "video/mp4"
    assert files.get_calls == 1
    assert files.deleted_names == ["files/test-video"]
    assert models.model_name == "gemini-3.1-flash-lite"
    assert models.config is not None
    assert models.config.response_mime_type == "application/json"
    assert models.config.response_schema is VideoOverview


def test_gemini_analyzer_normalizes_provider_failure_and_still_cleans_up(tmp_path):
    """SDK failures become domain errors without leaking the uploaded file."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake video content")
    files = FakeFiles([types.FileState.ACTIVE])
    models = FakeModels(error=RuntimeError("temporary provider failure"))
    analyzer = GeminiVideoAnalyzer(client=FakeClient(files, models))

    with pytest.raises(VideoProviderError, match="temporary provider failure"):
        asyncio.run(analyzer.analyze_overview(clip))

    assert files.deleted_names == ["files/test-video"]


def test_gemini_analyzer_creates_question_specific_detail(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake video content")
    detail = VideoDetailAnalysis(
        answer="Na tabliczce widnieje napis: Magazyn.",
        observations=[
            VideoDetailObservation(
                start_seconds=12,
                end_seconds=14,
                description="Widoczna tabliczka z napisem Magazyn.",
            )
        ],
    )
    files = FakeFiles([types.FileState.ACTIVE])
    models = FakeModels(types.GenerateContentResponse(parsed=detail))
    analyzer = GeminiVideoAnalyzer(client=FakeClient(files, models))

    result = asyncio.run(
        analyzer.analyze_detail(
            clip,
            question="Co było napisane na tabliczce?",
            start_seconds=10,
            end_seconds=20,
        )
    )

    assert result == detail
    assert models.config is not None
    assert models.config.response_schema is VideoDetailAnalysis
    assert models.contents is not None
    assert "Co było napisane" in str(models.contents[0])
    assert "10.000 do 20.000" in str(models.contents[0])
    assert files.deleted_names == ["files/test-video"]


def test_gemini_analyzer_rejects_invalid_local_file_before_upload(tmp_path):
    """Missing, empty, and unsupported files never reach the external provider."""
    files = FakeFiles([types.FileState.ACTIVE])
    models = FakeModels()
    analyzer = GeminiVideoAnalyzer(client=FakeClient(files, models))
    empty_video = tmp_path / "empty.mp4"
    empty_video.write_bytes(b"")
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("not a video", encoding="utf-8")

    for invalid_path in (tmp_path / "missing.mp4", empty_video, unsupported):
        with pytest.raises(InvalidVideoFileError):
            asyncio.run(analyzer.analyze_overview(invalid_path))

    assert files.uploaded_path is None
