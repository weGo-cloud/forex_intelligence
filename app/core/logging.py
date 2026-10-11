"""Application logging setup.

Provides a single `configure_logging` entry point used once at application
startup, and a `get_logger` helper for modules that need a logger.
Log records never include secrets: callers are responsible for not passing
sensitive values into log messages, and this module does not log
configuration values (which may include connection strings) directly.
"""

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(log_level: str = "INFO") -> None:
    """Configure root logging for the application process.

    Safe to call once at startup. Idempotent in the sense that calling it
    again simply reconfigures the same handler rather than stacking
    duplicate handlers.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.upper())

    # Remove any pre-existing handlers so repeated calls (e.g. in tests)
    # don't produce duplicate log lines.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with the given name."""
    return logging.getLogger(name)
