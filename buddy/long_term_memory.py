"""Long-term Harness Memory backed by SQLite."""

from pathlib import Path

from pydantic_ai_harness.memory import Memory, SqliteMemoryStore

from buddy.config import MEMORY_DATABASE_PATH


LONG_TERM_MEMORY_GUIDANCE = """
Ta pamięć jest trwała i jest dostępna pomiędzy różnymi sesjami rozmowy.
Zapisuj w niej tylko informacje przydatne w przyszłych rozmowach: trwałe
preferencje użytkownika, długoterminowe cele, ważne decyzje i istotny kontekst.
Nie zapisuj sekretów ani chwilowych szczegółów bez znaczenia. Utrzymuj
MEMORY.md zwięzłe, aktualizuj istniejące informacje i unikaj duplikatów.
""".strip()

MEMORY_AGENT_NAME = "main"
MEMORY_STORE_PATH = f"{MEMORY_AGENT_NAME}/MEMORY.md"


class InjectionOnlyMemory(Memory):
    """Inject MEMORY.md without exposing Harness's flat-file memory tools.

    Session and game documents use nested logical paths and are accessed through
    the application's indexed ``recall_memory`` tool. Durable writes are handled
    by the post-session analyzers directly through ``store``.
    """

    def get_toolset(self):
        return None


def create_long_term_memory(
    database_path: str | Path = MEMORY_DATABASE_PATH,
) -> Memory:
    """Create long-term Harness Memory persisted in a SQLite database."""
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    store = SqliteMemoryStore(database=database_path)
    return InjectionOnlyMemory(
        store=store,
        agent_name=MEMORY_AGENT_NAME,
        guidance=LONG_TERM_MEMORY_GUIDANCE,
    )
