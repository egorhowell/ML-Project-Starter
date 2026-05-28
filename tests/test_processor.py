"""Tests for src.processor."""

from __future__ import annotations

import pandas as pd
import pytest

from src import processor, settings
from src.processor import _add_date_features, _add_interaction_terms


# ------------------------------------------------------------------
# preprocess() — (X, y) split
# ------------------------------------------------------------------

def test_preprocess_splits_X_and_y(regression_df, monkeypatch) -> None:
    monkeypatch.setattr(settings, "TARGET_COLUMN", "target")
    monkeypatch.setattr(settings, "FEATURE_COLUMNS", [])

    X, y = processor.preprocess(regression_df)

    assert "target" not in X.columns
    assert y.name == "target"
    assert len(X) == len(y)


def test_preprocess_respects_feature_columns(regression_df, monkeypatch) -> None:
    monkeypatch.setattr(settings, "TARGET_COLUMN", "target")
    monkeypatch.setattr(settings, "FEATURE_COLUMNS", ["feature_a"])

    X, _ = processor.preprocess(regression_df)

    assert list(X.columns) == ["feature_a"]


def test_preprocess_drops_null_target_rows(monkeypatch) -> None:
    monkeypatch.setattr(settings, "TARGET_COLUMN", "target")
    monkeypatch.setattr(settings, "FEATURE_COLUMNS", [])

    df = pd.DataFrame({"feature_a": [1.0, 2.0, 3.0], "target": [10.0, None, 30.0]})
    X, y = processor.preprocess(df)

    assert len(X) == 2
    assert y.isna().sum() == 0


def test_preprocess_resets_index(regression_df, monkeypatch) -> None:
    monkeypatch.setattr(settings, "TARGET_COLUMN", "target")
    monkeypatch.setattr(settings, "FEATURE_COLUMNS", [])

    X, y = processor.preprocess(regression_df.iloc[10:].copy())

    assert list(X.index) == list(range(len(X)))


# ------------------------------------------------------------------
# preprocess_features() — inference (no target)
# ------------------------------------------------------------------

def test_preprocess_features_excludes_target(regression_df, monkeypatch) -> None:
    monkeypatch.setattr(settings, "TARGET_COLUMN", "target")
    monkeypatch.setattr(settings, "FEATURE_COLUMNS", [])

    df = regression_df.drop(columns=["target"])
    X = processor.preprocess_features(df)

    assert "target" not in X.columns


def test_preprocess_features_raises_on_missing_feature_column(monkeypatch) -> None:
    monkeypatch.setattr(settings, "TARGET_COLUMN", "target")
    monkeypatch.setattr(settings, "FEATURE_COLUMNS", ["nonexistent"])

    df = pd.DataFrame({"feature_a": [1.0], "feature_b": [2.0]})
    with pytest.raises(ValueError, match="FEATURE_COLUMNS"):
        processor.preprocess_features(df)


# ------------------------------------------------------------------
# Feature-engineering helpers
# ------------------------------------------------------------------

def test_add_interaction_terms(regression_df) -> None:
    result = _add_interaction_terms(regression_df.copy(), "feature_a", "feature_b")
    assert "feature_a_x_feature_b" in result.columns
    expected = regression_df["feature_a"] * regression_df["feature_b"]
    pd.testing.assert_series_equal(result["feature_a_x_feature_b"], expected, check_names=False)


def test_add_date_features() -> None:
    df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=7, freq="D"), "value": range(7)})
    result = _add_date_features(df.copy(), "date")
    for col in ("day_of_week", "month", "quarter", "is_weekend"):
        assert col in result.columns
    assert set(result["is_weekend"].unique()).issubset({0, 1})
