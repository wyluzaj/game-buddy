"""Gemini Files API implementation of the provider-neutral video analyzer."""

import asyncio
import json
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from buddy.config import MODELS, SUPPORTED_VIDEO_MIME_TYPES, VIDEO, Model
from buddy.prompts import (
    VIDEO_DETAIL_ANALYZER_INSTRUCTIONS,
    VIDEO_DETAIL_REQUEST,
    VIDEO_EPISODE_ANALYZER_INSTRUCTIONS,
    VIDEO_EPISODE_REQUEST,
    VIDEO_OVERVIEW_ANALYZER_INSTRUCTIONS,
    VIDEO_OVERVIEW_REQUEST,
)
from buddy.video.analyzer import (
    InvalidVideoFileError,
    VideoProviderError,
)
from buddy.video.models import EpisodeAnalysis, VideoDetailAnalysis, VideoOverview

StructuredVideoResult = TypeVar("StructuredVideoResult", bound=BaseModel)


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
        model: Model | str = MODELS.video_episode,
        client: _GenAIClient | None = None,
        processing_timeout_seconds: float = VIDEO.processing_timeout_seconds,
        poll_interval_seconds: float = VIDEO.poll_interval_seconds,
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

    async def analyze_episode(self, video_path: Path) -> EpisodeAnalysis:
        """Create a detailed searchable timeline for one complete episode."""
        response = await self._generate_structured(
            video_path,
            request=VIDEO_EPISODE_REQUEST,
            instructions=VIDEO_EPISODE_ANALYZER_INSTRUCTIONS,
            response_schema=EpisodeAnalysis,
        )
        return _parse_structured(response, EpisodeAnalysis, "episode analysis")

    async def analyze_overview(self, video_path: Path) -> VideoOverview:
        """Upload one local video, analyze it, and remove the temporary upload."""
        response = await self._generate_structured(
            video_path,
            request=VIDEO_OVERVIEW_REQUEST,
            instructions=VIDEO_OVERVIEW_ANALYZER_INSTRUCTIONS,
            response_schema=VideoOverview,
        )
        return _parse_structured(response, VideoOverview, "video overview")

    async def analyze_detail(
        self,
        video_path: Path,
        *,
        question: str,
        start_seconds: float,
        end_seconds: float,
    ) -> VideoDetailAnalysis:
        """Closely inspect one caller-selected range of a local segment."""
        request = VIDEO_DETAIL_REQUEST.format(
            question_json=json.dumps(question, ensure_ascii=False),
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
        response = await self._generate_structured(
            video_path,
            request=request,
            instructions=VIDEO_DETAIL_ANALYZER_INSTRUCTIONS,
            response_schema=VideoDetailAnalysis,
        )
        return _parse_structured(response, VideoDetailAnalysis, "video detail")

    async def _generate_structured(
        self,
        video_path: Path,
        *,
        request: str,
        instructions: str,
        response_schema: type[StructuredVideoResult],
    ) -> types.GenerateContentResponse:
        """Own the shared upload, processing, generation, and cleanup lifecycle."""
        path, mime_type = _validate_video_file(video_path)
        uploaded: types.File | None = None
        stage = "upload"
        try:
            uploaded = await self._client.aio.files.upload(
                file=path,
                config=types.UploadFileConfig(mime_type=mime_type),
            )
            stage = "processing"
            uploaded = await self._wait_until_active(uploaded)
            stage = "generation"
            response = await self._client.aio.models.generate_content(
                model=self._model_name,
                contents=[request, uploaded],
                config=types.GenerateContentConfig(
                    system_instruction=instructions,
                    response_mime_type="application/json",
                    response_json_schema=_gemini_json_schema(response_schema),
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True,
                    ),
                    temperature=0.1,
                ),
            )
            return response
        except VideoProviderError:
            raise
        except Exception as exc:
            # This is the provider boundary: transport, auth, quota, and SDK
            # errors are intentionally normalized for the application layer.
            raise VideoProviderError(
                f"Gemini video {stage} failed: {exc}"
            ) from exc
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


_GEMINI_SCHEMA_KEYS = frozenset(
    {"type", "properties", "items", "required", "enum"}
)


def _gemini_json_schema(
    response_type: type[BaseModel],
) -> dict[str, Any]:
    """Build a minimal Gemini-compatible schema from a Pydantic model.

    Gemini accepts only a subset of JSON Schema and may reject otherwise valid
    Pydantic schemas with a generic ``400 INVALID_ARGUMENT``. Generation only
    needs the structural keys below; Pydantic validates bounds and constraints
    after the response is received.
    """
    schema = response_type.model_json_schema()
    definitions = schema.get("$defs", {})

    def simplify(value: Any) -> Any:
        if isinstance(value, list):
            return [simplify(item) for item in value]

        if not isinstance(value, dict):
            return value

        alternatives = value.get("anyOf")
        if isinstance(alternatives, list):
            non_null = [
                item
                for item in alternatives
                if not (isinstance(item, dict) and item.get("type") == "null")
            ]
            if len(non_null) == 1:
                return simplify(non_null[0])

        reference = value.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            definition_name = reference.removeprefix("#/$defs/")
            definition = definitions.get(definition_name)
            if definition is None:
                raise ValueError(
                    f"Unknown schema reference: {reference}"
                )

            resolved = simplify(definition)

            # Resolve nested Pydantic models while retaining structural siblings.
            siblings = {
                key: simplify(item)
                for key, item in value.items()
                if key != "$ref" and key in _GEMINI_SCHEMA_KEYS
            }
            if isinstance(resolved, dict):
                return {**resolved, **siblings}
            return resolved

        result: dict[str, Any] = {}
        for key, item in value.items():
            if key not in _GEMINI_SCHEMA_KEYS:
                continue
            if key == "properties" and isinstance(item, dict):
                result[key] = {
                    property_name: simplify(property_schema)
                    for property_name, property_schema in item.items()
                }
            else:
                result[key] = simplify(item)

        return result

    simplified = simplify(schema)
    if not isinstance(simplified, dict):
        raise TypeError("The simplified response schema must be an object.")
    return simplified


def _parse_structured(
    response: types.GenerateContentResponse,
    result_type: type[StructuredVideoResult],
    label: str,
) -> StructuredVideoResult:
    parsed = response.parsed
    try:
        if isinstance(parsed, result_type):
            return parsed
        if parsed is not None:
            return result_type.model_validate(parsed)
        if response.text:
            return result_type.model_validate_json(response.text)
    except (ValidationError, ValueError) as exc:
        raise VideoProviderError(
            f"Gemini returned an invalid structured {label}."
        ) from exc
    raise VideoProviderError(f"Gemini returned an empty {label}.")
