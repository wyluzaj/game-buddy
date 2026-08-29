"""Long-term Harness Memory backed by SQLite."""

from pathlib import Path

from pydantic_ai_harness.memory import Memory, SqliteMemoryStore

from buddy.config import STORAGE
from buddy.prompts import LONG_TERM_MEMORY_GUIDANCE


MEMORY_STORE_PATH = f"{STORAGE.memory_agent_name}/MEMORY.md"


class InjectionOnlyMemory(Memory):
    """Inject MEMORY.md without exposing Harness's flat-file memory tools.

    Session and game documents use nested logical paths and are accessed through
    the application's indexed ``recall_memory`` tool. Durable writes are handled
    by the post-session analyzers directly through ``store``.
    """

    def get_toolset(self):
        return None


def create_long_term_memory(
    database_path: str | Path = STORAGE.memory_database_path,
) -> Memory:
    """Create long-term Harness Memory persisted in a SQLite database."""
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    store = SqliteMemoryStore(database=database_path)
    return InjectionOnlyMemory(
        store=store,
        agent_name=STORAGE.memory_agent_name,
        guidance=LONG_TERM_MEMORY_GUIDANCE,
    )
