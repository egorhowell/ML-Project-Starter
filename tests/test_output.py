"""Tests for src.output."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import output


def test_format_predictions_attaches_prediction_column() -> None:
    df = pd.DataFrame({"feature_a": [1.0, 2.0], "feature_b": [3.0, 4.0]})
    preds = np.array([10.0, 20.0])

    rows = output.format_predictions(df, preds)

    assert len(rows) == 2
    assert rows[0]["prediction"] == pytest.approx(10.0)
    assert rows[1]["prediction"] == pytest.approx(20.0)


def test_format_predictions_preserves_source_columns() -> None:
    df = pd.DataFrame({"id": ["a", "b"], "value": [1.0, 2.0]})
    preds = np.array([0.5, 0.8])

    rows = output.format_predictions(df, preds)

    assert rows[0]["id"] == "a"
    assert rows[0]["value"] == pytest.approx(1.0)


def test_format_predictions_adds_timestamp() -> None:
    df = pd.DataFrame({"x": [1.0]})
    rows = output.format_predictions(df, np.array([42.0]))
    assert "predicted_at" in rows[0]


def test_format_predictions_raises_on_length_mismatch() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0]})
    preds = np.array([1.0])  # wrong length
    with pytest.raises(ValueError, match="Length mismatch"):
        output.format_predictions(df, preds)


def test_format_predictions_coerces_numpy_types() -> None:
    df = pd.DataFrame({"x": [1.0]})
    preds = np.array([np.float32(3.14)])
    rows = output.format_predictions(df, preds)
    assert isinstance(rows[0]["prediction"], float)


def test_format_predictions_handles_string_predictions() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0]})
    preds = np.array(["cat", "dog"])
    rows = output.format_predictions(df, preds)
    assert rows[0]["prediction"] == "cat"
    assert rows[1]["prediction"] == "dog"
