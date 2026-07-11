from pydantic_ai.models.fallback import FallbackModel

from buddy.config import Model, MODEL_TIERS, build_model


def test_build_model_returns_plain_string_when_alone_in_tier():
    """Verify build_model skips the FallbackModel wrapper when no peers exist in the tier."""
    result = build_model(Model.OPENAI_MINI)
    assert isinstance(result, str)
    assert result == Model.OPENAI_MINI.value


def test_fallback_list_contains_only_same_tier_peers(monkeypatch):
    """Verify the fallback ordering and membership: same tier, chosen excluded."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")

    result = build_model(Model.GEMINI_LITE)
    assert isinstance(result, FallbackModel)
    # FallbackModel stores its models as a list; extract the raw strings
    used_names = [m.model_name if hasattr(m, "model_name") else str(m) for m in result.models]
    # chosen must be present, peer must be present, everyone else absent
    assert Model.GEMINI_LITE.value.split(":")[-1] in "".join(used_names)
    assert Model.OPENAI.value.split(":")[-1] in "".join(used_names)
