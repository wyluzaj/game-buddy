from pydantic_ai.models.fallback import FallbackModel

from buddy.config import Model, MODEL_TIERS, build_model


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
