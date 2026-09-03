import asyncio
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from buddy.config import STORAGE
from buddy.video import cli


def test_parser_uses_central_configuration_defaults() -> None:
    args = cli.build_parser().parse_args(["clip.mp4"])

    assert args.video_path == Path("clip.mp4")
    assert args.duration_seconds is None
    assert args.database == STORAGE.chat_database_path
    assert args.memory_database == STORAGE.memory_database_path
    assert args.game is None


def test_main_rejects_non_positive_duration(tmp_path: Path, capsys) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video")

    exit_code = cli.main([str(clip), "--duration-seconds", "0"])

    assert exit_code == 2
    assert "must be positive" in capsys.readouterr().err


def test_main_rejects_missing_file(tmp_path: Path, capsys) -> None:
    exit_code = cli.main([str(tmp_path / "missing.mp4")])

    assert exit_code == 2
    assert "does not exist" in capsys.readouterr().err


def test_main_runs_analysis_and_prints_outcome(tmp_path: Path, monkeypatch) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video")
    database = tmp_path / "chatbot.db"
    sentinel = object()
    captured: dict[str, object] = {}

    async def fake_run_analysis(conn, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(cli, "_run_analysis", fake_run_analysis)
    monkeypatch.setattr(
        cli,
        "_print_outcome",
        lambda outcome: captured.update(outcome=outcome),
    )

    exit_code = cli.main(
        [
            str(clip),
            "--duration-seconds",
            "30",
            "--conversation-id",
            "7",
            "--database",
            str(database),
            "--memory-database",
            str(tmp_path / "memory.db"),
            "--game",
            "Big Walk",
        ]
    )

    assert exit_code == 0
    assert captured["video_path"] == clip.resolve()
    assert captured["conversation_id"] == 7
    assert captured["game"] == "Big Walk"
    assert captured["memory_database"] == str(tmp_path / "memory.db")
    assert captured["outcome"] is sentinel
    assert database.is_file()


def test_main_maps_expected_analysis_failure_to_exit_one(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video")

    async def fail_analysis(conn, **kwargs):
        raise ValueError("invalid provider response")

    monkeypatch.setattr(cli, "_run_analysis", fail_analysis)

    exit_code = cli.main(
        [str(clip), "--database", str(tmp_path / "chatbot.db")]
    )

    assert exit_code == 1
    error = capsys.readouterr().err
    assert "Video analysis failed" in error
    assert "invalid provider response" in error


def test_run_analysis_links_confirmed_game_profile(tmp_path, monkeypatch) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video")
    memory = object()
    context_butler = object()
    sentinel = object()
    captured: dict[str, object] = {}

    class FakeAnalyzer:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    async def fake_ensure_game_document(conn, actual_memory, game):
        captured.update(memory=actual_memory, game=game)
        return SimpleNamespace(id=42)

    async def fake_process_video_file(conn, analyzer, butler, **kwargs):
        captured.update(
            analyzer=analyzer,
            context_butler=butler,
            process_kwargs=kwargs,
        )
        return sentinel

    monkeypatch.setattr(cli, "create_long_term_memory", lambda path: memory)
    monkeypatch.setattr(cli, "ensure_game_document", fake_ensure_game_document)
    monkeypatch.setattr(
        cli,
        "build_video_context_butler",
        lambda: context_butler,
    )
    monkeypatch.setattr(cli, "GeminiVideoAnalyzer", FakeAnalyzer)
    monkeypatch.setattr(cli, "process_video_file", fake_process_video_file)
    started_at = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)

    result = asyncio.run(
        cli._run_analysis(
            object(),
            video_path=clip,
            started_at=started_at,
            ended_at=started_at + timedelta(minutes=2),
            conversation_id=7,
            game="Big Walk",
            memory_database=tmp_path / "memory.db",
        )
    )

    assert result is sentinel
    assert captured["memory"] is memory
    assert captured["game"] == "Big Walk"
    assert captured["context_butler"] is context_butler
    assert captured["process_kwargs"] == {
        "file_path": clip,
        "started_at": started_at,
        "ended_at": started_at + timedelta(minutes=2),
        "conversation_id": 7,
        "game_document_id": 42,
    }


def test_main_detects_mp4_duration(tmp_path: Path, monkeypatch) -> None:
    clip = tmp_path / "clip.mp4"
    mvhd_payload = b"\0\0\0\0" + b"\0" * 8 + struct.pack(">II", 1_000, 117_000)
    mvhd = struct.pack(">I4s", len(mvhd_payload) + 8, b"mvhd") + mvhd_payload
    moov = struct.pack(">I4s", len(mvhd) + 8, b"moov") + mvhd
    clip.write_bytes(moov)
    captured = {}

    async def fake_run_analysis(conn, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(cli, "_run_analysis", fake_run_analysis)
    monkeypatch.setattr(cli, "_print_outcome", lambda outcome: None)

    assert cli.main([str(clip), "--database", str(tmp_path / "db.sqlite")]) == 0
    assert (
        captured["ended_at"] - captured["started_at"]
    ).total_seconds() == 117
