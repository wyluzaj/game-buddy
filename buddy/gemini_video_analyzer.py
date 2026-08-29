"""Gemini Files API implementation of the provider-neutral video analyzer."""

import asyncio
from pathlib import Path
from typing import Protocol, cast

from google import genai
from google.genai import types
from pydantic import ValidationError

from buddy.config import Model, VIDEO_OVERVIEW_MODEL
from buddy.prompts import (
    VIDEO_OVERVIEW_ANALYZER_INSTRUCTIONS,
    VIDEO_OVERVIEW_REQUEST,
)
from buddy.video_analyzer import (
    InvalidVideoFileError,
    VideoProviderError,
)
from buddy.video_models import VideoOverview


DEFAULT_FILE_PROCESSING_TIMEOUT_SECONDS = 180.0
DEFAULT_FILE_POLL_INTERVAL_SECONDS = 2.0

SUPPORTED_VIDEO_MIME_TYPES: dict[str, str] = {
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".mov": "video/mov",
    ".avi": "video/avi",
    ".flv": "video/x-flv",
    ".webm": "video/webm",
    ".wmv": "video/wmv",
    ".3gp": "video/3gpp",
}


class _AsyncFiles(Protocol):
    async def upload(
        self,
        *,
        file: str | Path,
        config: types.UploadFileConfig,
    ) -> types.File: ...

    async def get(self, *, name: str) -> types.File: ...

    async def delete(self, *, name: str) -> object: ...


class _AsyncModels(Protocol):
    async def generate_content(
        self,
        *,
        model: str,
        contents: list[object],
        config: types.GenerateContentConfig,
    ) -> types.GenerateContentResponse: ...


class _AsyncClient(Protocol):
    files: _AsyncFiles
    models: _AsyncModels

    async def aclose(self) -> None: ...


class _GenAIClient(Protocol):
    aio: _AsyncClient


class GeminiVideoAnalyzer:
    """Upload completed clips to Gemini and return validated overview data."""

    def __init__(
        self,
        *,
        model: Model | str = VIDEO_OVERVIEW_MODEL,
        client: _GenAIClient | None = None,
        processing_timeout_seconds: float = DEFAULT_FILE_PROCESSING_TIMEOUT_SECONDS,
        poll_interval_seconds: float = DEFAULT_FILE_POLL_INTERVAL_SECONDS,
    ) -> None:
        if processing_timeout_seconds <= 0:
            raise ValueError("Video processing timeout must be positive.")
        if poll_interval_seconds < 0:
            raise ValueError("Video polling interval cannot be negative.")

        self._model_name = _google_model_name(model)
        self._client = client or cast(_GenAIClient, genai.Client())
        self._owns_client = client is None
        self._processing_timeout_seconds = processing_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds

    @property
    def provider(self) -> str:
        return "google"

    @property
    def model_name(self) -> str:
        return self._model_name

    async def analyze_overview(self, video_path: Path) -> VideoOverview:
        """Upload one local video, analyze it, and remove the temporary upload."""
        path, mime_type = _validate_video_file(video_path)
        uploaded: types.File | None = None
        try:
            uploaded = await self._client.aio.files.upload(
                file=path,
                config=types.UploadFileConfig(mime_type=mime_type),
            )
            uploaded = await self._wait_until_active(uploaded)
            response = await self._client.aio.models.generate_content(
                model=self._model_name,
                contents=[VIDEO_OVERVIEW_REQUEST, uploaded],
                config=types.GenerateContentConfig(
                    system_instruction=VIDEO_OVERVIEW_ANALYZER_INSTRUCTIONS,
                    response_mime_type="application/json",
                    response_schema=VideoOverview,
                    temperature=0.1,
                ),
            )
            return _parse_overview(response)
        except VideoProviderError:
            raise
        except Exception as exc:
            # This is the provider boundary: transport, auth, quota, and SDK
            # errors are intentionally normalized for the application layer.
            raise VideoProviderError(f"Gemini video analysis failed: {exc}") from exc
        finally:
            if uploaded is not None and uploaded.name:
                await self._delete_uploaded_file(uploaded.name)

    async def close(self) -> None:
        """Close the internally created HTTP client."""
        if self._owns_client:
            await self._client.aio.aclose()

    async def __aenter__(self) -> "GeminiVideoAnalyzer":
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.close()

    async def _wait_until_active(self, uploaded: types.File) -> types.File:
        if not uploaded.name:
            raise VideoProviderError("Gemini upload did not return a file name.")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._processing_timeout_seconds
        current = uploaded
        while True:
            state = _file_state_name(current)
            if state == "ACTIVE":
                return current
            if state == "FAILED":
                raise VideoProviderError("Gemini failed to process the uploaded video.")
            if loop.time() >= deadline:
                raise VideoProviderError(
                    "Gemini did not finish processing the video before timeout."
                )
            await asyncio.sleep(self._poll_interval_seconds)
            current = await self._client.aio.files.get(name=uploaded.name)

    async def _delete_uploaded_file(self, name: str) -> None:
        try:
            await self._client.aio.files.delete(name=name)
        except Exception:
            # The remote file expires automatically; cleanup failure must not
            # replace a successful analysis or hide the original provider error.
            return


def _google_model_name(model: Model | str) -> str:
    value = model.value if isinstance(model, Model) else model
    if value.startswith("google:"):
        return value.removeprefix("google:")
    if ":" in value:
        raise ValueError("GeminiVideoAnalyzer requires a Google model.")
    if not value.strip():
        raise ValueError("Gemini video model name cannot be empty.")
    return value.strip()


def _validate_video_file(video_path: Path) -> tuple[Path, str]:
    path = Path(video_path)
    if not path.is_file():
        raise InvalidVideoFileError(f"Video file does not exist: {path}")
    if path.stat().st_size == 0:
        raise InvalidVideoFileError(f"Video file is empty: {path}")
    mime_type = SUPPORTED_VIDEO_MIME_TYPES.get(path.suffix.casefold())
    if mime_type is None:
        raise InvalidVideoFileError(
            f"Unsupported video file extension: {path.suffix or '(none)'}"
        )
    return path, mime_type


def _file_state_name(uploaded: types.File) -> str:
    state = uploaded.state
    if state is None:
        return "PROCESSING"
    name = getattr(state, "name", state)
    return str(name).rsplit(".", 1)[-1].upper()


def _parse_overview(response: types.GenerateContentResponse) -> VideoOverview:
    parsed = response.parsed
    try:
        if isinstance(parsed, VideoOverview):
            return parsed
        if parsed is not None:
            return VideoOverview.model_validate(parsed)
        if response.text:
            return VideoOverview.model_validate_json(response.text)
    except (ValidationError, ValueError) as exc:
        raise VideoProviderError(
            "Gemini returned an invalid structured video overview."
        ) from exc
    raise VideoProviderError("Gemini returned an empty video overview.")
