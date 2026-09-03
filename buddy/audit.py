"""Read-only UTF-8 export of chat metadata and logical memory documents."""

import argparse
import json
import sqlite3
from pathlib import Path

from buddy.config import STORAGE

CHAT_AUDIT_TABLES = (
    "conversations",
    "messages",
    "memory_documents",
    "memory_relations",
    "memory_processing_jobs",
    "memory_search_en",
    "video_segments",
    "video_reports",
    "video_context_updates",
    "video_observations",
    "video_observation_search_en",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Buddy databases to human-readable UTF-8 text files.",
    )
    parser.add_argument(
        "--database",
        default=STORAGE.chat_database_path,
        help="Chat SQLite database path (default: data/chatbot.db).",
    )
    parser.add_argument(
        "--memory-database",
        default=STORAGE.memory_database_path,
        help="Harness Memory SQLite path (default: data/memory.db).",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("audit"),
        help="Directory for UTF-8 exports (default: audit).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        chat_path, memory_path = export_audit(
            database_path=Path(args.database),
            memory_database_path=Path(args.memory_database),
            output_directory=args.output_directory,
        )
    except (OSError, sqlite3.Error, ValueError) as error:
        print(f"Audit export failed: {type(error).__name__}: {error}")
        return 1
    print(f"Chat database audit: {chat_path}")
    print(f"Memory documents audit: {memory_path}")
    return 0


def export_audit(
    *,
    database_path: Path,
    memory_database_path: Path,
    output_directory: Path,
) -> tuple[Path, Path]:
    """Export application state without routing Unicode through the shell."""
    chat_source = database_path.expanduser().resolve(strict=True)
    memory_source = memory_database_path.expanduser().resolve(strict=True)
    destination = output_directory.expanduser().resolve(strict=False)
    destination.mkdir(parents=True, exist_ok=True)

    chat_output = destination / "chat-database-audit.txt"
    memory_output = destination / "memory-documents-audit.txt"
    chat_output.write_text(
        _render_chat_database(chat_source),
        encoding="utf-8",
        newline="\n",
    )
    memory_output.write_text(
        _render_memory_documents(memory_source),
        encoding="utf-8",
        newline="\n",
    )
    return chat_output, memory_output


def _render_chat_database(database_path: Path) -> str:
    connection = sqlite3.connect(_read_only_uri(database_path), uri=True)
    connection.row_factory = sqlite3.Row
    try:
        available = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }
        sections = []
        for table in CHAT_AUDIT_TABLES:
            if table not in available:
                continue
            rows = connection.execute(f'SELECT * FROM "{table}"').fetchall()
            payload = "\n".join(
                json.dumps(dict(row), ensure_ascii=False, default=str)
                for row in rows
            ) or "(empty)"
            sections.append(f"=== {table} ===\n{payload}")
        return "\n\n".join(sections) + "\n"
    finally:
        connection.close()


def _render_memory_documents(database_path: Path) -> str:
    connection = sqlite3.connect(_read_only_uri(database_path), uri=True)
    connection.row_factory = sqlite3.Row
    try:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memory_files'"
        ).fetchone()
        if table is None:
            raise ValueError("The selected database is not a Harness Memory store.")
        rows = connection.execute(
            """
            SELECT path, content FROM memory_files
            WHERE path = 'main' OR path LIKE 'main/%'
            ORDER BY path
            """
        ).fetchall()
        sections = [
            f"=== {row['path']} ===\n{row['content']}"
            for row in rows
        ]
        return ("\n\n".join(sections) or "(no memory documents)") + "\n"
    finally:
        connection.close()


def _read_only_uri(path: Path) -> str:
    return f"{path.as_uri()}?mode=ro"


if __name__ == "__main__":
    raise SystemExit(main())
