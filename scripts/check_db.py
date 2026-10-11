#!/usr/bin/env python3
"""Standalone PostgreSQL connectivity check.

Usage:
    python scripts/check_db.py

Exits 0 and prints a success message if a connection + trivial query
succeeds; exits 1 otherwise. Intended for manual verification during
local development and Docker Compose setup — this is deliberately not
wired into the /health endpoint (see app/api/health.py for why).
"""

import sys

from app.core.database import check_database_connection


def main() -> int:
    if check_database_connection():
        print("OK: connected to PostgreSQL successfully.")
        return 0
    print("FAILED: could not connect to PostgreSQL. Check DATABASE_URL.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
