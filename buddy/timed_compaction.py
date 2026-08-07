"""SummarizingCompactionTimed: time-based variant of Harness SummarizingCompaction.

Adapts the LLM-powered summarization compaction to trigger on a time window with a
message-count floor.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    SystemPromptPart,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.compaction._shared import (
    _is_safe_cutoff,
    compact_with_span,
    context_for_request,
    find_first_user_message,
)

if TYPE_CHECKING:
    from pydantic_ai.models import Model, ModelRequestContext


_GAMING_SUMMARY_PROMPT = """\
Jesteś asystentem streszczającym rozmowy podczas grania. Dotychczasowa rozmowa zostanie \
zastąpiona twoim streszczeniem, więc musi zawierać wszystko potrzebne żeby \
kontynuować rozmowę w naturalny sposób.

Napisz streszczenie pod tymi dokładnymi nagłówkami, pomijając sekcję tylko \
jeśli nie ma treści:

## O czym rozmawialiśmy
Główny temat, ton rozmowy, nastrój usera.

## Co się działo w grze
Wydarzenia, walki, odkrycia, momenty które user wspominał albo które \
były kontekstem rozmowy.

## Emocje i reakcje
Frustracje, radości, obserwacje, komentarze — nastrój i stan emocjonalny.

## Styl gry i preferencje
Co user powiedział o swoim stylu, preferencjach, decyzjach w grze.

## Otwarte tematy
Rzeczy o których planowaliśmy pogadać dalej, niedokończone tematy.

Skup się na kontekście i tym co ważne, nie na dokładnym powtarzaniu tego \
co zostało powiedziane. Odpowiedz TYLKO streszczeniem — bez wstępu, bez \
markdown fences.

<messages>
{messages}
</messages>\
"""

_SUMMARY_PREFIX = 'Summary of previous conversation:\n\n'


def _format_messages(messages: Sequence[ModelMessage]) -> str:
    """Render messages into a human-readable string for summarization."""
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    lines.append(f'User: {_user_prompt_text(part)}')
                elif isinstance(part, SystemPromptPart):
                    lines.append(f'System: {part.content}')
                elif isinstance(part, ToolReturnPart):
                    content_str = str(part.content)[:500]
                    if len(str(part.content)) > 500:
                        content_str += '...'
                    lines.append(f'Tool [{part.tool_name}]: {content_str}')
        else:
            for part in msg.parts:
                if isinstance(part, TextPart):
                    lines.append(f'Assistant: {part.content}')
                elif isinstance(part, ToolCallPart):
                    lines.append(f'Tool Call [{part.tool_name}]: {part.args}')
    return '\n'.join(lines)


def _user_prompt_text(part: UserPromptPart) -> str:
    """Extract text content from a user prompt part."""
    if isinstance(part.content, str):
        return part.content
    texts: list[str] = []
    for item in part.content:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, TextContent):
            texts.append(item.content)
    return ' '.join(texts) if texts else ''


def _extract_system_prompts(messages: list[ModelMessage]) -> list[SystemPromptPart]:
    """Extract leading system-prompt parts from the conversation."""
    parts: list[SystemPromptPart] = []
    for msg in messages:
        if not isinstance(msg, ModelRequest):
            break
        for part in msg.parts:
            if isinstance(part, SystemPromptPart):
                if not part.content.startswith(_SUMMARY_PREFIX):
                    parts.append(part)
            else:
                return parts
    return parts


def _extract_previous_summary(messages: list[ModelMessage]) -> str | None:
    """Extract the most recent compaction summary from the message history.

    Looks for a ``SystemPromptPart`` whose content starts with the summary prefix,
    which indicates it was produced by a prior compaction pass.
    """
    for msg in reversed(messages):
        if not isinstance(msg, ModelRequest):
            continue
        for part in reversed(msg.parts):
            if isinstance(part, SystemPromptPart) and part.content.startswith(_SUMMARY_PREFIX):
                return part.content[len(_SUMMARY_PREFIX):]
    return None


def _is_summary_message(message: ModelMessage) -> bool:
    """Return whether a message contains a generated compaction summary."""
    if not isinstance(message, ModelRequest):
        return False
    return any(
        isinstance(part, SystemPromptPart) and part.content.startswith(_SUMMARY_PREFIX)
        for part in message.parts
    )


def _find_time_cutoff(
    messages: list[ModelMessage],
    history_minutes: int,
    min_keep_messages: int,
) -> int:
    """Find cutoff index for summarization based on time window and count floor.

    Messages ``[:cutoff]`` will be summarized, messages ``[cutoff:]`` preserved as-is.

    Rules:
    - Messages older than ``history_minutes`` become the candidate cutoff
      (the first message within the window sets the boundary).
    - The cutoff is clamped so at least ``min_keep_messages`` remain in the
      tail (the floor only raises the kept count, never lowers it).
    - Walks the cutoff backward through ``_is_safe_cutoff`` to avoid orphaning
      any tool_call / tool_return pair.
    """
    if not messages:
        return 0

    # Time-based cutoff: index of first message inside the window.
    # None-timestamped messages are treated as old (defensive, keeps them in summary).
    cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=history_minutes)
    time_cutoff = len(messages)  # default: all messages are old
    for idx, msg in enumerate(messages):
        if msg.timestamp is not None and msg.timestamp >= cutoff_time:
            time_cutoff = idx
            break

    # Count floor: never trim more than ``len - min_keep_messages`` from the head.
    floor_cutoff = max(0, len(messages) - min_keep_messages)
    cutoff = min(time_cutoff, floor_cutoff)

    # Walk backward to preserve tool_call/tool_return pairing.
    for idx in range(cutoff, -1, -1):
        if _is_safe_cutoff(messages, idx):
            return idx
    return 0


@dataclass
class SummarizingCompactionTimed(AbstractCapability[AgentDepsT]):
    """Time-based LLM-powered conversation compaction.

    Same behavior as Harness ``SummarizingCompaction`` but the trigger and cutoff
    are driven by a time window with a message-count floor. Older messages outside the window are
    summarized and replaced with a single structured summary message; recent
    messages are preserved as-is.

    Example:
        ```python
        SummarizingCompactionTimed(
            model="google:gemini-3.1-flash-lite",
            history_minutes=30,
            min_keep_messages=15,
        )
        ```
    """

    model: str | Model | None = None
    """Model used to generate summaries. When ``None``, inherits the running agent's model."""

    history_minutes: int = 30
    """Time window in minutes; messages older than this are candidates for summarization."""

    min_keep_messages: int = 15
    """Floor on tail size; at least this many recent messages are always preserved."""

    summary_prompt: str = _GAMING_SUMMARY_PROMPT
    """Prompt template for generating summaries. Must contain a ``{messages}`` placeholder."""

    preserve_first_user_message: bool = False
    """When ``True``, the first user prompt is always kept after compaction."""

    incremental: bool = True
    """When ``True``, existing summary from a prior compaction is extended rather than regenerated."""

    def __post_init__(self) -> None:
        if self.history_minutes < 1:
            raise ValueError('history_minutes must be positive.')
        if self.min_keep_messages < 0:
            raise ValueError('min_keep_messages must be non-negative.')

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Compact newly expired messages before a model request."""
        messages: list[ModelMessage] = list(request_context.messages)
        request_ctx = context_for_request(ctx, request_context)

        request_context.messages = await compact_with_span(
            request_ctx,
            strategy='SummarizingCompactionTimed',
            messages=messages,
            compact=lambda: self.compact(messages, request_ctx),
        )
        return request_context

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
    ) -> list[ModelMessage]:
        """Summarize older messages, replacing them with a single summary message."""
        cutoff = _find_time_cutoff(
            messages,
            self.history_minutes,
            self.min_keep_messages,
        )

        if cutoff <= 0:
            return messages

        system_parts = _extract_system_prompts(messages)
        to_summarize = messages[:cutoff]
        preserved = messages[cutoff:]

        first_user_msg = (
            find_first_user_message(messages)
            if self.preserve_first_user_message
            else None
        )
        new_old_messages = [
            message
            for message in to_summarize
            if not _is_summary_message(message) and message is not first_user_msg
        ]
        if not new_old_messages:
            return messages

        previous_summary = _extract_previous_summary(messages) if self.incremental else None
        prompt_messages = to_summarize
        if previous_summary is not None:
            prompt_messages = [
                message
                for message in to_summarize
                if not _is_summary_message(message) and message is not first_user_msg
            ]
        summary = await self._summarize(
            prompt_messages,
            ctx,
            previous_summary=previous_summary,
        )

        # Timestamp the summary as the oldest message it covers, so future compaction
        # passes see it as "old" and can iterate past it when looking for the time cutoff.
        oldest_ts = min(
            (m.timestamp for m in to_summarize if m.timestamp is not None),
            default=datetime.now(timezone.utc),
        )
        summary_part = SystemPromptPart(content=f'{_SUMMARY_PREFIX}{summary}')
        summary_message = ModelRequest(parts=[*system_parts, summary_part], timestamp=oldest_ts)

        first_user: list[ModelMessage] = []
        if first_user_msg is not None:
            idx = messages.index(first_user_msg)
            if idx < cutoff and first_user_msg not in preserved:
                first_user = [first_user_msg]

        return [summary_message,
                *first_user,
                *preserved]

    async def _summarize(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
        *,
        previous_summary: str | None = None,
    ) -> str:
        """Generate a summary for the given messages using the configured model."""
        from pydantic_ai import Agent

        formatted = _format_messages(messages)
        prompt = self.summary_prompt.format(messages=formatted)

        if previous_summary is not None:
            prompt = f'{prompt}\n\n<previous_summary>\n{previous_summary}\n</previous_summary>'

        model = self.model if self.model is not None else ctx.model
        agent: Agent[None, str] = Agent(
            model,
            instructions='You are a context summarization assistant. Extract the most important information from conversations.',
        )
        result = await agent.run(prompt, usage=ctx.usage)
        return result.output.strip()
