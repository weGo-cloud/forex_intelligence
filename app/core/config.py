"""Typed, environment-based application configuration.

Settings are read from environment variables (and, for local development,
from a `.env` file that is never committed to version control). No secret
values are hard-coded here — the defaults below are non-sensitive local
development placeholders only, intended to let the app and its test suite
start without requiring a `.env` file to exist.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings sourced from environment variables.

    Precedence (highest to lowest): real environment variables, then
    values in a local `.env` file, then the defaults declared below.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: Literal["development", "test", "staging", "production"] = "development"
    log_level: str = "INFO"

    # Non-secret local-dev default. Override via the DATABASE_URL env var
    # (or .env) in every real deployment.
    database_url: str = "postgresql+psycopg://forex:forex@localhost:5432/forex_intelligence"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Cached because Settings parses environment variables on construction;
    re-parsing on every call would be wasted work for values that do not
    change during the process lifetime.
    """
    return Settings()
