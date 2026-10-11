"""Application entry point.

Run locally with:
    uvicorn app.main:app --reload
"""

from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.market_data import router as market_data_router
from app.api.research import router as research_router
from app.api.setups import router as setups_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

# Importing a strategy module registers it in app.strategies.registry - the
# API layer needs every strategy available by name, so they're imported
# here for their registration side effect (same pattern scripts/lab.py
# uses), rather than app.research being coupled to specific strategies.
import app.strategies.orb  # noqa: F401
import app.strategies.rsi_divergence  # noqa: F401
import app.strategies.mean_reversion  # noqa: F401
import app.strategies.liquidity_sweep_fvg  # noqa: F401

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)

app = FastAPI(
    title="Forex Intelligence",
    description=(
        "Quantitative forex research and analysis platform. "
        "Research and paper-trading only — see README for current scope."
    ),
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(market_data_router)
app.include_router(research_router)
app.include_router(setups_router)


@app.on_event("startup")
def on_startup() -> None:
    logger.info("Forex Intelligence starting up (env=%s)", settings.app_env)

