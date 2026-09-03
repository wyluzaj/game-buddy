from pydantic_ai.models.fallback import FallbackModel

from buddy.config import (
    LANGUAGE,
    MODEL_TIERS,
    MODELS,
    Model,
    build_model,
)
from buddy.prompts import build_response_language_instruction


def test_build_model_returns_plain_string_when_alone_in_tier(monkeypatch):
    """Verify build_model skips the FallbackModel wrapper when no peers exist in the tier."""
    monkeypatch.setitem(MODEL_TIERS, Model.NVIDIA_LITE, "isolated-test-tier")

    result = build_model(Model.NVIDIA_LITE)

    assert isinstance(result, str)
    assert result == Model.NVIDIA_LITE.value


def test_fallback_list_contains_only_same_tier_peers(monkeypatch):
    """Verify the fallback ordering and membership: same tier, chosen excluded."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")

    result = build_model(Model.MINIMAX)
    assert isinstance(result, FallbackModel)
    used_names = [m.model_name if hasattr(m, "model_name") else str(m) for m in result.models]
    expected = [Model.MINIMAX.value.split(":", 1)[1]] + [
        model.value.split(":", 1)[1]
        for model, tier in MODEL_TIERS.items()
        if tier == MODEL_TIERS[Model.MINIMAX] and model is not Model.MINIMAX
    ]
    assert used_names == expected


def test_video_analysis_models_are_configured_explicitly():
    """Overview and detail analysis can evolve independently in config."""
    assert MODELS.video_episode is Model.GEMINI_LITE
    assert MODELS.video_overview is Model.GEMINI_LITE
    assert MODELS.video_detail is Model.GEMINI_LITE
    assert MODELS.video_context_butler is Model.GEMINI_LITE


def test_response_language_is_configured_separately_from_agent_behavior():
    instruction = build_response_language_instruction(
        LANGUAGE.default_response_language
    )

    assert LANGUAGE.default_response_language == "Polish"
    assert "latest explicit request" in instruction
    assert "internal memory" in instruction


def test_response_language_rejects_an_empty_default():
    try:
        build_response_language_instruction("  ")
    except ValueError as error:
        assert str(error) == "The default response language cannot be empty."
    else:
        raise AssertionError("An empty response language should be rejected.")
