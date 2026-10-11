"""Shared pytest fixtures.

M0 has no database-backed functionality yet, so the test suite never
opens a real connection to PostgreSQL — it only needs Settings to
construct successfully, which it does from the non-secret defaults in
app.core.config, or from a real .env if one is present.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
