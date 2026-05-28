"""Shared pytest fixtures.

Fixtures defined here are auto-discovered by pytest and available to every test
file without an explicit import.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import settings


# ------------------------------------------------------------------
# Storage fixtures
# ------------------------------------------------------------------

@pytest.fixture
def temp_fallback_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the local-JSON fallback to a per-test temp file.

    Also clears SUPABASE_* env vars so code under test always uses the
    local-fallback path rather than trying to reach a real database.
    """
    target = tmp_path / "predictions.json"
    monkeypatch.setattr(settings, "LOCAL_FALLBACK_PATH", str(target))
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    return target


# ------------------------------------------------------------------
# Generic ML data fixtures
# ------------------------------------------------------------------

@pytest.fixture
def regression_df() -> pd.DataFrame:
    """100-row DataFrame with two numeric features and a numeric target.

    Seeded for determinism. Use this as the default dataset for regression tasks.
    """
    rng = np.random.default_rng(42)
    n = 100
    feature_a = rng.normal(0, 1, n)
    feature_b = rng.normal(5, 2, n)
    target = 3.0 * feature_a - 1.5 * feature_b + rng.normal(0, 0.5, n)
    return pd.DataFrame({"feature_a": feature_a, "feature_b": feature_b, "target": target})


@pytest.fixture
def classification_df() -> pd.DataFrame:
    """100-row DataFrame with two numeric features and a binary string target.

    Use this as the default dataset for classification tasks.
    """
    rng = np.random.default_rng(0)
    n = 100
    feature_a = rng.normal(0, 1, n)
    feature_b = rng.normal(0, 1, n)
    label = np.where(feature_a + feature_b > 0, "pos", "neg")
    return pd.DataFrame({"feature_a": feature_a, "feature_b": feature_b, "target": label})


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    """100-row DataFrame mixing numeric and categorical features with a numeric target.

    Use to test that the ColumnTransformer preprocessing handles mixed dtypes.
    """
    rng = np.random.default_rng(7)
    n = 100
    numeric = rng.normal(0, 1, n)
    categorical = rng.choice(["A", "B", "C"], n)
    target = numeric + rng.normal(0, 0.1, n)
    return pd.DataFrame({"numeric": numeric, "category": categorical, "target": target})
