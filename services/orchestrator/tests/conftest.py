import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from memory import get_db
from services.orchestrator.main import app


class _DummyDB:
    async def commit(self):
        return None


@pytest.fixture(autouse=True, scope="session")
def override_db():
    async def _override_get_db():
        yield _DummyDB()

    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.pop(get_db, None)
