"""Model catalog, tier fallback, and history filter settings."""
from enum import Enum
from pydantic_ai.models.fallback import FallbackModel


class Model(str, Enum):
    """Enum representing different AI models."""

    NVIDIA = "openrouter:nvidia/nemotron-3-ultra-550b-a55b:free"
    POOLSIDE = "openrouter:poolside/laguna-s-2.1:free"
    MINIMAX = "openrouter:minimax/minimax-m3:free"
    NVIDIA_LITE = "openrouter:nvidia/nemotron-3.5-lightning:free"
    LING = "openrouter:inclusionai/ling-3.0-flash-fin:free"
    # GEMINI = "google:gemini-3.5-flash"
    GEMINI_LITE = "google:gemini-3.1-flash-lite"
    GEMINI_LITE_3_5 = "google:gemini-3.5-flash-lite"


MODEL_TIERS: dict[Model, str] = {
    Model.NVIDIA: "high",
    Model.POOLSIDE: "high",
    Model.MINIMAX: "low",
    Model.NVIDIA_LITE: "low",
    # Model.GEMINI: "high",
    Model.GEMINI_LITE: "medium",
    Model.GEMINI_LITE_3_5: "medium",
    Model.LING: "medium",
}

ACTIVE_MODEL = Model.NVIDIA

MIN_RECENT_MESSAGES = 15
HISTORY_MINUTES = 30

SHORT_TERM_MEMORY_MODEL = Model.GEMINI_LITE
LONG_TERM_MEMORY_MODEL = Model.GEMINI_LITE
CONTEXT_BUTLER_MODEL = Model.GEMINI_LITE
VIDEO_OVERVIEW_MODEL = Model.GEMINI_LITE
VIDEO_DETAIL_MODEL = Model.GEMINI_LITE

MEMORY_DATABASE_PATH = "data/memory.db"


def choose_model() -> Model:
    """
    Prompts the user to choose a model from the available options.
    Returns the selected model or the default ACTIVE_MODEL if no selection is made.
    """
    models = list(Model)

    print("Available models:")
    for numer, model in enumerate(models):
        print(f"  {numer} - {model.value} [{MODEL_TIERS[model]}]")

    while True:
        user_input = input(f"Choose model [Enter = {ACTIVE_MODEL.value}]: ").strip()

        if not user_input:
            return ACTIVE_MODEL

        if user_input.isdigit():
            index = int(user_input)
            if 0 <= index < len(models):
                return models[index]

        print(f"Invalid selection. Enter a number from 0 to {len(models) - 1} or press Enter.")


def build_model(chosen: Model) -> str | FallbackModel:
    """ Builds a FallbackModel based on the chosen model and its tier.
    If the chosen model has no fallbacks, it returns the chosen model directly."""
    tier = MODEL_TIERS[chosen]
    fallbacks = [m.value for m in MODEL_TIERS if MODEL_TIERS[m] == tier and m != chosen]

    if not fallbacks:
        return chosen.value

    return FallbackModel(chosen.value, *fallbacks)
