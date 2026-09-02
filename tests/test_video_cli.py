from pathlib import Path

from buddy.config import STORAGE, VIDEO
from buddy.video import cli


def test_parser_uses_central_configuration_defaults() -> None:
    args = cli.build_parser().parse_args(["clip.mp4"])

    assert args.video_path == Path("clip.mp4")
    assert args.duration_seconds == VIDEO.manual_duration_seconds
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
