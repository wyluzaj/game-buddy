from datetime import datetime, timedelta, timezone

from pydantic_ai.messages import ModelRequest, UserPromptPart

from buddy import agent


def make_msg(minutes_ago: int) -> ModelRequest:
    """Build a fake message stamped N minutes before now."""
    ts = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return ModelRequest(parts=[UserPromptPart(content="x", timestamp=ts)], timestamp=ts)


def _defaults(monkeypatch, minutes=30, limit=4):
    monkeypatch.setattr(agent, "USE_LAST_MINUTES", True)
    monkeypatch.setattr(agent, "USE_LAST_N", True)
    monkeypatch.setattr(agent, "HISTORY_MINUTES", minutes)
    monkeypatch.setattr(agent, "HISTORY_LIMIT", limit)


def test_n_acts_as_floor_not_ceiling(monkeypatch):
    """Verify HISTORY_LIMIT is a minimum, not a maximum: many recent messages are all kept."""
    _defaults(monkeypatch, minutes=30, limit=4)
    messages = [make_msg(m) for m in [25, 20, 15, 10, 5, 3, 1]]
    result = agent.keep_recent(messages)
    assert len(result) == 7


def test_floor_extends_when_time_filter_returns_too_few(monkeypatch):
    """Verify floor pulls in older messages to reach HISTORY_LIMIT when window returns fewer."""
    _defaults(monkeypatch, minutes=30, limit=4)
    messages = [make_msg(m) for m in [60, 50, 40, 35, 1]]  # only 1 within 30 min
    result = agent.keep_recent(messages)
    assert len(result) == 4
    assert result == messages[-4:]


def test_flags_off_returns_full_list_unchanged(monkeypatch):
    """Verify both feature flags disabled means no filtering at all — passthrough."""
    monkeypatch.setattr(agent, "USE_LAST_MINUTES", False)
    monkeypatch.setattr(agent, "USE_LAST_N", False)
    messages = [make_msg(600), make_msg(300), make_msg(1)]
    result = agent.keep_recent(messages)
    assert result == messages
