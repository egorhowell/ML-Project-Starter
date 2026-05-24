"""Configuration for the pipeline.

Anything that you might want to tweak without changing pipeline logic lives here.
Treat this file as your control panel.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# WHAT YOU ARE PREDICTING
# -----------------------------------------------------------------------------
# Each entry is one thing the pipeline will train a model for and produce a
# prediction about. For the default weather demo, each entry is a city.
#
# REPLACE THIS with whatever your project predicts:
#   - Stocks: [("AAPL", {}), ("GOOGL", {}), ...]
#   - Football teams: [("Arsenal", {"league": "EPL"}), ...]
#   - Cryptocurrencies: [("BTC", {}), ("ETH", {}), ...]
#
# The dict in each entry is free-form metadata passed through to the extractor.
ITEMS_TO_PREDICT: list[tuple[str, dict]] = [
    ("London", {"latitude": 51.5074, "longitude": -0.1278}),
    ("New York", {"latitude": 40.7128, "longitude": -74.0060}),
    ("Tokyo", {"latitude": 35.6762, "longitude": 139.6503}),
]

# How much historical data the model trains on. More history = more signal but
# slower and potentially less relevant. 90 days is a sensible default for daily data.
LOOKBACK_DAYS: int = 90

# -----------------------------------------------------------------------------
# STORAGE
# -----------------------------------------------------------------------------
# Name of the Supabase table where predictions are written. You'll create this
# table in the Supabase UI — see README "Supabase setup" for the schema.
SUPABASE_TABLE_NAME: str = "predictions"

# If Supabase isn't configured (env vars missing), the pipeline falls back to
# writing predictions to this local JSON file. Useful for first-run / local dev.
# This file is gitignored.
LOCAL_FALLBACK_PATH: str = "data/predictions.json"

# -----------------------------------------------------------------------------
# DASHBOARD
# -----------------------------------------------------------------------------
# Title shown at the top of the Streamlit dashboard. Make it yours.
DASHBOARD_TITLE: str = "ML Project Starter — Daily Weather Predictions"
DASHBOARD_SUBTITLE: str = (
    "Placeholder demo: predicting tomorrow's mean temperature for a few cities. "
    "Replace the model, data, and copy with your own project."
)
