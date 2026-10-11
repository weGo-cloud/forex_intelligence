"""M3: Research Experiment & Robustness Foundation.

Turns a single backtest (M2's question: "how did this perform once?")
into a structured, reproducible research experiment (M3's question:
"how reproducible, robust and trustworthy is that performance?").

Nothing in this package is coupled to a specific strategy, indicator,
broker, or the future dashboard - see docs/M3_RESEARCH_EXPERIMENTS.md
for the architecture this package exists to support.

Importing this package registers `app.research.registry.ExperimentRecord`
on `app.core.database.Base`'s metadata, same convention as `app.data`.
"""

from app.research import registry as registry  # noqa: F401  (registers ORM models on Base.metadata)
