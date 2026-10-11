"""Market-data provider implementations.

`base` defines the vendor-agnostic interface; concrete providers (e.g.
`mock`) implement it. No commercial vendor is wired in at M1 — the
rest of the application depends only on `MarketDataProvider`.
"""
