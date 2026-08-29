"""Delegate agent that reduces retrieved evidence before it reaches chat history."""

from typing import Protocol

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunUsage
from pydantic_ai.models import Model as PydanticModel

from buddy.config import MEMORY, MODELS, build_model
from buddy.prompts import (
    MEMORY_CONTEXT_BUTLER_INSTRUCTIONS,
    MEMORY_CONTEXT_BUTLER_PROMPT,
)


class ContextButlerResult(BaseModel):
    """Small, source-aware context returned to the main chat agent."""

    enough_evidence: bool
    context: str = Field(max_length=MEMORY.context_chars)
    sources: list[str] = Field(default_factory=list, max_length=MEMORY.context_sources)
    uncertainty: str | None = Field(
        default=None,
        max_length=MEMORY.context_uncertainty_chars,
    )


class ContextButlerRunResult(Protocol):
    """Minimal result interface required from the delegate agent."""

    @property
    def output(self) -> ContextButlerResult: ...


class ContextButler(Protocol):
    """Minimal async delegate interface used by memory recall."""

    async def run(
        self,
        prompt: str,
        *,
        usage: RunUsage | None = None,
    ) -> ContextButlerRunResult: ...


def build_context_butler(
    model: str | PydanticModel | None = None,
) -> Agent[None, ContextButlerResult]:
    """Build the tool-less delegate with an explicitly configured model."""
    butler_model = model if model is not None else build_model(MODELS.memory_context_butler)
    return Agent[None, ContextButlerResult](
        model=butler_model,
        deps_type=type(None),
        name="memory_context_butler",
        instructions=MEMORY_CONTEXT_BUTLER_INSTRUCTIONS,
        output_type=ContextButlerResult,
    )


def build_memory_context_prompt(question: str, recall_result_json: str) -> str:
    """Build a clearly delimited evidence-reduction prompt."""
    return MEMORY_CONTEXT_BUTLER_PROMPT.format(
        question=question,
        recall_result=recall_result_json,
    )


def validate_butler_sources(
    result: ContextButlerResult,
    *,
    allowed_sources: set[str],
) -> ContextButlerResult:
    """Reject source paths invented by the delegate instead of trusting them."""
    if any(source not in allowed_sources for source in result.sources):
        return ContextButlerResult(
            enough_evidence=False,
            context="",
            sources=[],
            uncertainty="Context Butler wskazał źródło spoza wyników wyszukiwania.",
        )
    sources = list(dict.fromkeys(result.sources))
    if result.enough_evidence and not sources:
        return ContextButlerResult(
            enough_evidence=False,
            context="",
            sources=[],
            uncertainty=(
                "Context Butler nie wskazał prawidłowego źródła dla znalezionych "
                "informacji."
            ),
        )
    return result.model_copy(update={"sources": sources})
