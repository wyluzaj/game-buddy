"""Central application configuration grouped by functional area."""

import os
from dataclasses import dataclass
from enum import Enum

from pydantic_ai.models.fallback import FallbackModel


class Model(str, Enum):
    """AI models available to the application."""

    NVIDIA = "openrouter:nvidia/nemotron-3-ultra-550b-a55b:free"
    POOLSIDE = "openrouter:poolside/laguna-s-2.1:free"
    MINIMAX = "openrouter:minimax/minimax-m3:free"
    NVIDIA_LITE = "openrouter:nvidia/nemotron-3.5-lightning:free"
    LING = "openrouter:inclusionai/ling-3.0-flash-fin:free"
    GEMINI_LITE = "google:gemini-3.1-flash-lite"
    GEMINI_LITE_3_5 = "google:gemini-3.5-flash-lite"


MODEL_TIERS: dict[Model, str] = {
    Model.NVIDIA: "high",
    Model.POOLSIDE: "high",
    Model.MINIMAX: "low",
    Model.NVIDIA_LITE: "low",
    Model.GEMINI_LITE: "medium",
    Model.GEMINI_LITE_3_5: "medium",
    Model.LING: "medium",
}


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """Models assigned to individual application roles."""

    active: Model = Model.NVIDIA
    short_term_memory: Model = Model.GEMINI_LITE
    long_term_memory: Model = Model.GEMINI_LITE
    memory_context_butler: Model = Model.GEMINI_LITE
    video_episode: Model = Model.GEMINI_LITE
    video_overview: Model = Model.GEMINI_LITE
    video_detail: Model = Model.GEMINI_LITE
    video_context_butler: Model = Model.GEMINI_LITE


@dataclass(frozen=True, slots=True)
class AgentSettings:
    """Main-agent history and tool-result limits."""

    history_minutes: int = 30
    min_recent_messages: int = 15
    clear_tool_results_max_tokens: int = 50_000
    clear_tool_results_keep_pairs: int = 3


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Terminal diagnostics controlled through environment variables."""

    show_tracebacks: bool = os.getenv("BUDDY_DEBUG", "").casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True, slots=True)
class StorageSettings:
    """Physical and logical storage locations."""

    chat_database_path: str = "data/chatbot.db"
    memory_database_path: str = "data/memory.db"
    memory_agent_name: str = "main"
    session_memory_prefix: str = "main/sessions"
    game_memory_prefix: str = "main/games"


@dataclass(frozen=True, slots=True)
class MemorySettings:
    """Bounds for durable memory, search, and Context Butler output."""

    global_document_chars: int = 100_000
    global_facts: int = 100
    global_fact_chars: int = 500
    session_document_chars: int = 1_000_000
    session_index_summary_chars: int = 500
    game_name_chars: int = 100
    game_document_chars: int = 100_000
    game_profile_updates: int = 8
    game_index_summary_chars: int = 500
    search_results: int = 20
    recall_results: int = 5
    recall_document_chars: int = 4_000
    recall_source_chars: int = 1_000_000
    context_chars: int = 1_500
    context_uncertainty_chars: int = 400
    context_sources: int = 5


@dataclass(frozen=True, slots=True)
class VideoSettings:
    """Video-provider, persistence, and context limits."""

    processing_timeout_seconds: float = 180.0
    poll_interval_seconds: float = 2.0
    stored_error_chars: int = 2_000
    detail_question_chars: int = 500
    context_chars: int = 1_000
    context_sources: int = 5
    context_uncertainty_chars: int = 400
    session_observations: int = 50
    session_observation_chars: int = 30_000
    manual_duration_seconds: float = 120.0
    max_episode_observations: int = 120
    max_observation_objects: int = 30
    max_observation_texts: int = 20
    max_observation_uncertainties: int = 10
    max_overview_events: int = 100
    max_visible_text_observations: int = 100
    max_uncertainties: int = 50
    hash_chunk_bytes: int = 1024 * 1024


MODELS = ModelSettings()
AGENT = AgentSettings()
RUNTIME = RuntimeSettings()
STORAGE = StorageSettings()
MEMORY = MemorySettings()
VIDEO = VideoSettings()


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


def choose_model() -> Model:
    """Prompt for a model and return the configured default on empty input."""
    models = list(Model)

    print("Available models:")
    for number, model in enumerate(models):
        print(f"  {number} - {model.value} [{MODEL_TIERS[model]}]")

    while True:
        user_input = input(f"Choose model [Enter = {MODELS.active.value}]: ").strip()
        if not user_input:
            return MODELS.active
        if user_input.isdigit():
            index = int(user_input)
            if 0 <= index < len(models):
                return models[index]
        print(f"Invalid selection. Enter a number from 0 to {len(models) - 1} or press Enter.")


def build_model(chosen: Model) -> str | FallbackModel:
    """Build a same-tier fallback model, or return the model string if alone."""
    tier = MODEL_TIERS[chosen]
    fallbacks = [
        model.value
        for model in MODEL_TIERS
        if MODEL_TIERS[model] == tier and model != chosen
    ]
    if not fallbacks:
        return chosen.value
    return FallbackModel(chosen.value, *fallbacks)
