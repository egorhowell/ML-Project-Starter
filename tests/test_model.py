"""Tests for src.model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.model import Model, _evaluate


# ------------------------------------------------------------------
# fit() and predict()
# ------------------------------------------------------------------

def test_fit_returns_regression_metrics(regression_df) -> None:
    X = regression_df.drop(columns=["target"])
    y = regression_df["target"]
    metrics = Model().fit(X, y)
    assert set(metrics) == {"mae", "rmse", "r2"}
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0


def test_fit_returns_classification_metrics(classification_df) -> None:
    X = classification_df.drop(columns=["target"])
    y = classification_df["target"]
    from sklearn.linear_model import LogisticRegression
    from unittest.mock import patch
    with patch("src.model._build_estimator", return_value=LogisticRegression()):
        metrics = Model().fit(X, y)
    assert "accuracy" in metrics
    assert "f1_weighted" in metrics


def test_predict_returns_array(regression_df) -> None:
    X = regression_df.drop(columns=["target"])
    y = regression_df["target"]
    model = Model()
    model.fit(X, y)
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X),)


def test_predict_before_fit_raises() -> None:
    model = Model()
    X = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(X)


def test_model_handles_mixed_feature_types(mixed_df) -> None:
    X = mixed_df.drop(columns=["target"])
    y = mixed_df["target"]
    model = Model()
    metrics = model.fit(X, y)
    assert "mae" in metrics
    preds = model.predict(X)
    assert len(preds) == len(X)


# ------------------------------------------------------------------
# save() / load()
# ------------------------------------------------------------------

def test_save_and_load_round_trip(regression_df, tmp_path) -> None:
    X = regression_df.drop(columns=["target"])
    y = regression_df["target"]

    model = Model()
    model.fit(X, y)
    preds_before = model.predict(X)

    path = tmp_path / "model.joblib"
    model.save(path)

    loaded = Model.load(path)
    preds_after = loaded.predict(X)

    np.testing.assert_array_almost_equal(preds_before, preds_after)


# ------------------------------------------------------------------
# _evaluate()
# ------------------------------------------------------------------

def test_evaluate_regression_returns_correct_keys() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    metrics = _evaluate(y_true, y_pred)
    assert set(metrics) == {"mae", "rmse", "r2"}


def test_evaluate_classification_returns_correct_keys() -> None:
    y_true = np.array(["cat", "dog", "cat"])
    y_pred = np.array(["cat", "cat", "cat"])
    metrics = _evaluate(y_true, y_pred)
    assert set(metrics) == {"accuracy", "f1_weighted"}
