"""Data preprocessing.

Cleans up raw data before it hits the model. This is where you'd handle things like:
  - Dropping or imputing missing values
  - Removing obvious outliers
  - Resampling irregular timestamps onto a fixed grid
  - Aligning multiple items so they all share the same date range
  - Feature engineering (deriving new columns from existing ones)

The default implementation is intentionally minimal — it just drops null rows and
sorts by date. As you build out your project, extend this file with whatever
domain-specific cleaning your data needs.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def preprocess(raw_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Clean each item's DataFrame independently.

    Args:
        raw_data: dict from item name -> raw DataFrame (as returned by extractor).

    Returns:
        Same shape, but cleaned. Drops rows with null values and sorts by date.

    REPLACE / EXTEND THIS with your own preprocessing. Common additions:
      - Imputation of missing values (forward-fill, mean, model-based)
      - Outlier detection and handling
      - Feature engineering: rolling means, lag features, etc.
      - Time-zone normalisation
    """
    cleaned: dict[str, pd.DataFrame] = {}
    for name, df in raw_data.items():
        before = len(df)
        df = df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)
        after = len(df)
        if before != after:
            logger.info("Dropped %d null rows for %s", before - after, name)
        cleaned[name] = df

    return cleaned
