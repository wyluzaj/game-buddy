import asyncio
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    SystemPromptPart,
    TextContent,
    UserPromptPart,
)

from buddy.timed_compaction import (
    _SUMMARY_PREFIX,
    SummarizingCompactionTimed,
    _find_time_cutoff,
    format_messages,
)


def make_msg(minutes_ago: int | None, content: str = "x") -> ModelRequest:
    """Build a fake message stamped N minutes before now."""
    ts = (
        datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
        if minutes_ago is not None
        else None
    )
    return ModelRequest(
        parts=[UserPromptPart(content=content, timestamp=ts)],
        timestamp=ts,
    )


def make_summary(minutes_ago: int = 60) -> ModelRequest:
    """Build a synthetic summary message stamped as old."""
    ts = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return ModelRequest(
        parts=[SystemPromptPart(content=f"{_SUMMARY_PREFIX}old summary")],
        timestamp=ts,
    )


def compact_with_fake_summary(
    monkeypatch: pytest.MonkeyPatch,
    capability: SummarizingCompactionTimed,
    messages: Sequence[ModelMessage],
) -> tuple[list[ModelMessage], list[tuple[list[ModelMessage], str | None]]]:
    """Run compaction without an LLM and capture summarizer inputs."""
    calls: list[tuple[list[ModelMessage], str | None]] = []

    async def fake_summarize(
        summarized: list[ModelMessage],
        ctx: Any,
        *,
        previous_summary: str | None = None,
    ) -> str:
        calls.append((summarized, previous_summary))
        return "updated summary"

    monkeypatch.setattr(capability, "_summarize", fake_summarize)
    result = asyncio.run(capability.compact(list(messages), cast(Any, None)))
    return result, calls


def test_min_keep_is_floor_not_ceiling():
    """When many messages fit the window, all are kept — floor doesn't shrink them."""
    messages = [make_msg(m) for m in [25, 20, 15, 10, 5, 3, 1]]
    cutoff = _find_time_cutoff(
        messages,
        history_minutes=30,
        min_keep_messages=4,
    )
    # All 7 messages are within the 30-min window → nothing to summarize.
    assert cutoff == 0


def test_floor_extends_when_time_filter_returns_too_few():
    """Floor keeps enough tail messages even when the window alone would trim too many."""
    messages = [make_msg(m) for m in [60, 50, 40, 35, 1]]  # only msg[4] within 30 min
    cutoff = _find_time_cutoff(
        messages,
        history_minutes=30,
        min_keep_messages=4,
    )
    # Time-only cutoff would be 4 (only msg[4] fits window), but floor of 4 forces
    # cutoff=1 so that messages[1:] (4 messages) are preserved.
    assert cutoff == 1


def test_missing_timestamps_are_treated_as_old():
    """Timestamp-less messages are eligible while the count floor remains intact."""
    messages = [make_msg(None) for _ in range(20)]
    cutoff = _find_time_cutoff(
        messages,
        history_minutes=30,
        min_keep_messages=15,
    )
    assert cutoff == 5


def test_class_rejects_negative_min_keep():
    """SummarizingCompactionTimed refuses negative floor."""
    with pytest.raises(ValueError, match="min_keep_messages"):
        SummarizingCompactionTimed(min_keep_messages=-1)


def test_class_defaults_from_config():
    """Default construction matches project's config values."""
    cap = SummarizingCompactionTimed()
    assert cap.history_minutes == 30
    assert cap.min_keep_messages == 15
    assert cap.incremental is True
    assert cap.preserve_first_user_message is False


def test_compact_summarizes_messages_without_timestamps(monkeypatch):
    """Missing timestamps reach the summarizer instead of being stopped by a trigger."""
    cap = SummarizingCompactionTimed()
    messages = [make_msg(None, str(index)) for index in range(20)]

    result, calls = compact_with_fake_summary(monkeypatch, cap, messages)

    assert len(calls) == 1
    assert len(calls[0][0]) == 5
    assert calls[0][1] is None
    assert len(result) == 16  # summary + 15-message tail


def test_existing_summary_alone_does_not_trigger_another_summary(monkeypatch):
    """A synthetic summary alone does not cause repeated LLM calls."""
    cap = SummarizingCompactionTimed()
    messages = [make_summary(), *[make_msg(1) for _ in range(15)]]

    result, calls = compact_with_fake_summary(monkeypatch, cap, messages)

    assert calls == []
    assert result == messages


def test_incremental_summary_receives_only_newly_expired_messages(monkeypatch):
    """An existing summary is used as an anchor, not summarized as raw input again."""
    cap = SummarizingCompactionTimed()
    summary = make_summary()
    newly_expired = make_msg(45, "newly expired")
    messages = [
        summary,
        newly_expired,
        *[make_msg(1) for _ in range(15)],
    ]

    result, calls = compact_with_fake_summary(monkeypatch, cap, messages)

    assert len(calls) == 1
    assert calls[0][0] == [newly_expired]
    assert calls[0][1] == "old summary"
    summary_parts = [
        part
        for part in result[0].parts
        if isinstance(part, SystemPromptPart)
        and part.content.startswith(_SUMMARY_PREFIX)
    ]
    assert len(summary_parts) == 1
    assert summary_parts[0].content == f"{_SUMMARY_PREFIX}updated summary"


def test_format_messages_excludes_harness_memory_injection():
    """Long-term memory injected into a request is not copied into session context."""
    request = ModelRequest(
        parts=[
            UserPromptPart(content="Nowa wiadomość użytkownika"),
            UserPromptPart(
                content=[
                    TextContent(
                        content="<memory>stara pamięć</memory>",
                        metadata="pydantic-ai-harness.memory.v1:test",
                    )
                ]
            ),
        ]
    )

    formatted = format_messages([request])

    assert formatted == "User: Nowa wiadomość użytkownika"
    assert "stara pamięć" not in formatted
