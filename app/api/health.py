"""Health check endpoint.

Deliberately independent of the database: this endpoint answers "is the
web process up and serving requests?", not "is every dependency healthy?".
Conflating the two would make the health check flaky in exactly the
situations (e.g. a slow database) where you most want it to stay green.
Database connectivity is verified separately — see scripts/check_db.py.
"""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def get_health() -> dict[str, str]:
    """Report that the service is running. Exposes no internal details."""
    return {"status": "ok"}
