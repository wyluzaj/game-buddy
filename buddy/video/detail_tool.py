"""Pydantic AI tool for question-specific re-analysis of stored gameplay."""

from collections.abc import Awaitable, Callable
import sqlite3

from pydantic_ai import RunContext

from buddy.config import MODELS
from buddy.video.gemini import GeminiVideoAnalyzer
from buddy.video.detail_service import analyze_video_detail_source
from buddy.video.models import VideoDetailAnswer


AnalyzeVideoDetailTool = Callable[..., Awaitable[VideoDetailAnswer]]


def create_analyze_video_detail_tool(
    conn: sqlite3.Connection,
) -> AnalyzeVideoDetailTool:
    """Bind the detailed analyzer to application-owned video metadata."""

    async def analyze_video_detail(
        ctx: RunContext,
        source_uri: str,
        question: str,
    ) -> VideoDetailAnswer:
        """Inspect a stored gameplay clip more closely.

        Use a `video://segment/...` source already present in the observed
        gameplay context. `question` must state exactly what visual detail the
        user needs. Never invent or modify a segment ID.
        """
        del ctx
        async with GeminiVideoAnalyzer(model=MODELS.video_detail) as analyzer:
            return await analyze_video_detail_source(
                conn,
                analyzer,
                source_uri=source_uri,
                question=question,
            )

    return analyze_video_detail
