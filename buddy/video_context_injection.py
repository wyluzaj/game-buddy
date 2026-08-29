"""Inject one replaceable observed-gameplay state into chat history."""

from collections.abc import Sequence
from dataclasses import replace

from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart

from buddy.prompts import VIDEO_CONTEXT_INJECTION_PROMPT
from buddy.video_models import VideoContextUpdate


VIDEO_CONTEXT_MARKER = "[game-buddy.video-context.v1]"


def inject_video_context(
    history: Sequence[ModelMessage],
    update: VideoContextUpdate,
) -> list[ModelMessage]:
    """Replace every prior video injection with the newest bounded snapshot."""
    cleaned = _without_video_context(history)
    sources = "\n".join(f"- {source}" for source in update.sources)
    content = VIDEO_CONTEXT_INJECTION_PROMPT.format(
        context=update.context,
        sources=sources,
    )
    cleaned.append(ModelRequest(parts=[SystemPromptPart(content=content)]))
    return cleaned


def _without_video_context(
    history: Sequence[ModelMessage],
) -> list[ModelMessage]:
    cleaned: list[ModelMessage] = []
    for message in history:
        if not isinstance(message, ModelRequest):
            cleaned.append(message)
            continue
        remaining_parts = [
            part
            for part in message.parts
            if not (
                isinstance(part, SystemPromptPart)
                and part.content.startswith(VIDEO_CONTEXT_MARKER)
            )
        ]
        if remaining_parts:
            cleaned.append(replace(message, parts=remaining_parts))
    return cleaned
