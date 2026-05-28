"""Output formatting.

Converts raw model predictions into a list of dicts ready for storage.
Extend this file with any post-prediction logic your project needs:
  - Thresholding probabilities → binary labels
  - Ranking / top-k selection
  - Inverse-transforming a log-scaled target
  - Attaching metadata (run ID, model version, timestamp)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def format_predictions(
    source_df: pd.DataFrame,
    predictions: np.ndarray,
) -> list[dict]:
    """Attach predictions to their source rows and return a list ready for storage.

    Args:
        source_df:   the DataFrame that was passed to model.predict() — used to
                     carry source-row context (IDs, feature values, …) into storage.
        predictions: 1-D array from model.predict(), same length as source_df.

    Returns:
        List of dicts, one per row, with a "prediction" key added and a
        "predicted_at" UTC timestamp.
    """
    if len(source_df) != len(predictions):
        raise ValueError(
            f"Length mismatch: source_df has {len(source_df)} rows "
            f"but predictions has {len(predictions)} values."
        )

    now = datetime.now(tz=timezone.utc).isoformat()
    rows = source_df.to_dict(orient="records")
    for row, pred in zip(rows, predictions):
        row["prediction"] = _coerce(pred)
        row["predicted_at"] = now
    return rows


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _coerce(value) -> float | str:
    """Normalise a single prediction to a JSON-serialisable scalar."""
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)
