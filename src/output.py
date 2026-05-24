"""Post-processing of predictions.

Optional step that runs after the model and before saving. Use it for things like:
  - Aggregating per-item predictions into a portfolio-level allocation
    (this is what the original Egor repo did: forecasts → optimal weights)
  - Applying business rules (clip predictions to a valid range, round to nearest cent)
  - Reconciling predictions across multiple models (ensembles)
  - Generating recommendations from raw predictions

The default implementation is a passthrough: it returns the predictions exactly
as they came in. If your project doesn't need post-processing, leave it alone.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def postprocess(
    predictions: dict[str, float],
    raw_data: dict[str, pd.DataFrame],
) -> dict[str, dict]:
    """Convert raw model predictions into rows ready for storage.

    Args:
        predictions: dict from item name -> predicted next-step value.
        raw_data:    the cleaned historical data, in case you need it for
                     post-processing (e.g. to compute predicted % change).

    Returns:
        A dict from item name -> dict of fields to persist. The default fields
        are designed to be useful on a dashboard; add or remove as you like.
    """
    rows: dict[str, dict] = {}
    for name, prediction in predictions.items():
        history = raw_data[name]
        last_value = float(history["value"].iloc[-1]) if len(history) else None
        change_pct = (
            ((prediction - last_value) / last_value * 100.0) if last_value else None
        )
        rows[name] = {
            "item": name,
            "prediction": float(prediction),
            "last_value": last_value,
            "predicted_change_pct": change_pct,
        }

    return rows
