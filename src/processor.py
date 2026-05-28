"""Data preprocessing.

Cleans raw data and splits it into features (X) and target (y) before the
model sees it.

    preprocess(df)          → (X, y)  — use during training
    preprocess_features(df) → X       — use during inference (no target column)

Extend _clean() and the feature-engineering helpers below with whatever
domain-specific transformations your project needs.

Common additions:
  - Imputation       → df.fillna(df.median()) or sklearn SimpleImputer
  - Outlier clipping → df[col].clip(lower=q01, upper=q99)
  - Date parsing     → pd.to_datetime(); extract year/month/dayofweek
  - Binning          → pd.cut() / pd.qcut()
  - Polynomial terms → sklearn PolynomialFeatures (add inside the model Pipeline)
  - Text features    → sklearn TfidfVectorizer or sentence-transformers
"""

from __future__ import annotations

import logging

import pandas as pd

from src import settings

logger = logging.getLogger(__name__)


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean df and split into (X, y) for model training.

    Args:
        df: raw DataFrame from the extractor, must contain settings.TARGET_COLUMN.

    Returns:
        X: feature DataFrame
        y: target Series
    """
    df = _clean(df, require_target=True)
    y = df[settings.TARGET_COLUMN]
    X = _select_features(df.drop(columns=[settings.TARGET_COLUMN]))
    return X, y


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean df and return the feature matrix X for inference (no target).

    Args:
        df: raw DataFrame from the extractor. Target column need not be present.

    Returns:
        X: feature DataFrame ready for model.predict().
    """
    df = _clean(df, require_target=False)
    return _select_features(df)


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _clean(df: pd.DataFrame, *, require_target: bool) -> pd.DataFrame:
    """Basic cleaning applied before every split."""
    before = len(df)

    cols_to_check = [settings.TARGET_COLUMN] if require_target else []
    if settings.FEATURE_COLUMNS:
        cols_to_check += settings.FEATURE_COLUMNS

    if cols_to_check:
        present = [c for c in cols_to_check if c in df.columns]
        if present:
            df = df.dropna(subset=present)

    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d rows with nulls in required columns", dropped)

    return df.reset_index(drop=True)


def _select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the configured feature columns (all non-target if unset)."""
    if settings.FEATURE_COLUMNS:
        missing = set(settings.FEATURE_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"FEATURE_COLUMNS references columns not in data: {missing}")
        return df[settings.FEATURE_COLUMNS]
    return df


# ------------------------------------------------------------------
# Optional feature-engineering helpers
# ------------------------------------------------------------------

def _add_interaction_terms(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    """Append a multiplicative interaction feature between two numeric columns."""
    df[f"{col_a}_x_{col_b}"] = df[col_a] * df[col_b]
    return df


def _add_date_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Expand a datetime column into calendar features.

    Useful when date_col carries seasonality signal (day of week, month, …).
    """
    dt = pd.to_datetime(df[date_col])
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    return df
