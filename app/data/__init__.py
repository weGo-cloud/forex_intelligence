"""Market-data domain: instruments, candles, provider abstraction,
normalization, validation, ingestion, and query/repository access.

Importing this package ensures `app.data.models` is imported, which
registers the market-data ORM models on `app.core.database.Base`'s
metadata. Anything that needs `Base.metadata.create_all(...)` to see
these tables (tests, `scripts/check_db.py`-style utilities) should
import `app.data` (or `app.data.models` directly) first.
"""

from app.data import models as models  # noqa: F401  (registers ORM models on Base.metadata)
