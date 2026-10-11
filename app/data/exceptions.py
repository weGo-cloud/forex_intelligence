"""Exceptions specific to the market-data domain."""

from __future__ import annotations


class NormalizationError(Exception):
    """A raw provider record could not be converted to canonical form.

    This represents a *data* problem (unparseable timestamp, missing
    price field, wrong type), not a programming error. Callers such as
    the ingestion pipeline are expected to catch this, record the
    record as rejected, and continue processing the rest of the batch.
    """
