"""Model catalog, tier fallback, and history filter settings."""
from enum import Enum
from pydantic_ai.models.fallback import FallbackModel


class Model(str, Enum):
    """Enum representing different AI models."""

    GEMINI = "google:gemini-3.5-flash"
    GEMINI_LITE = "google:gemini-3.1-flash-lite"
    OPENAI = "openrouter:openai/gpt-oss-120b:free"
    OPENAI_MINI = "openrouter:openai/gpt-oss-20b:free"


MODEL_TIERS: dict[Model, str] = {
    Model.GEMINI: "high",
    Model.GEMINI_LITE: "medium",
    Model.OPENAI: "medium",
    Model.OPENAI_MINI: "low"
}

ACTIVE_MODEL = Model.GEMINI

MIN_RECENT_MESSAGES = 15
HISTORY_MINUTES = 30


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
