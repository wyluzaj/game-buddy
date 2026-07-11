import pytest
from buddy.database import init_db


@pytest.fixture
def db():
    """Fresh in-memory SQLite connection per test."""
    conn = init_db(":memory:")
    yield conn
    conn.close()
