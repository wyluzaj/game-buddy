from datetime import datetime, timedelta, timezone

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)

from buddy.database import create_conversation
from buddy.video_context_injection import VIDEO_CONTEXT_MARKER, inject_video_context
from buddy.video_models import VideoContextUpdate, VideoOverview, VideoReportMode
from buddy.video_storage import (
    create_video_segment,
    get_latest_pending_video_context,
    get_video_context_update_for_report,
    mark_video_context_injected_through,
    save_video_context_update,
    save_video_report,
)


def _update(db, tmp_path, conversation_id, *, minute, context):
    started_at = datetime(2026, 8, 29, 18, minute, tzinfo=timezone.utc)
    path = tmp_path / f"segment-{minute}.mp4"
    path.write_bytes(b"video")
    segment = create_video_segment(
        db,
        file_path=path,
        started_at=started_at,
        ended_at=started_at + timedelta(minutes=2),
        conversation_id=conversation_id,
    )
    report = save_video_report(
        db,
        segment_id=segment.id,
        mode=VideoReportMode.OVERVIEW,
        provider="stub",
        model_name="stub",
        content=VideoOverview(summary=context).model_dump(mode="json"),
    )
    return save_video_context_update(
        db,
        report_id=report.id,
        changed=True,
        context=context,
        sources=[segment.source_uri],
    )


def test_injection_replaces_old_video_state_without_touching_chat():
    old = VideoContextUpdate(
        id=1,
        report_id=1,
        changed=True,
        context="Stary stan.",
        sources=["video://segment/1"],
        uncertainty=None,
        created_at=datetime.now(timezone.utc),
        injected_at=None,
    )
    new = VideoContextUpdate(
        id=2,
        report_id=2,
        changed=True,
        context="Gracz dotarł do zamkniętej bramy.",
        sources=["video://segment/2?t=10-20"],
        uncertainty=None,
        created_at=datetime.now(timezone.utc),
        injected_at=None,
    )
    original = [
        ModelRequest(parts=[UserPromptPart(content="Co teraz?")]),
        ModelResponse(parts=[TextPart(content="Rozejrzyjmy się.")]),
    ]
    with_old = inject_video_context(original, old)

    result = inject_video_context(with_old, new)

    rendered = "\n".join(
        part.content
        for message in result
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, (SystemPromptPart, UserPromptPart))
        and isinstance(part.content, str)
    )
    assert rendered.count(VIDEO_CONTEXT_MARKER) == 1
    assert "Stary stan" not in rendered
    assert "zamkniętej bramy" in rendered
    assert "video://segment/2?t=10-20" in rendered
    assert result[:2] == original


def test_acknowledging_latest_state_consumes_superseded_pending_updates(
    db,
    tmp_path,
):
    conversation_id = create_conversation(db)
    older = _update(
        db,
        tmp_path,
        conversation_id,
        minute=0,
        context="Gracz wszedł do zamku.",
    )
    latest = _update(
        db,
        tmp_path,
        conversation_id,
        minute=3,
        context="Gracz znalazł zamkniętą bramę.",
    )

    selected = get_latest_pending_video_context(
        db,
        conversation_id=conversation_id,
    )
    assert selected == latest

    mark_video_context_injected_through(
        db,
        conversation_id=conversation_id,
        update_id=latest.id,
    )

    assert get_latest_pending_video_context(
        db,
        conversation_id=conversation_id,
    ) is None
    assert get_video_context_update_for_report(db, older.report_id).injected_at
    assert get_video_context_update_for_report(db, latest.report_id).injected_at


def test_pending_video_context_is_isolated_by_conversation(db, tmp_path):
    first_conversation = create_conversation(db)
    second_conversation = create_conversation(db)
    first_update = _update(
        db,
        tmp_path,
        first_conversation,
        minute=0,
        context="Pierwsza gra.",
    )
    second_update = _update(
        db,
        tmp_path,
        second_conversation,
        minute=3,
        context="Druga gra.",
    )

    assert get_latest_pending_video_context(
        db,
        conversation_id=first_conversation,
    ) == first_update
    assert get_latest_pending_video_context(
        db,
        conversation_id=second_conversation,
    ) == second_update
