"""Tests for src.model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.model import Model


def test_model_predicts_after_fitting() -> None:
    """A fitted model returns a float close to the last observed value."""
    np.random.seed(0)  # Random walk is stochastic; seed for determinism.
    series = pd.Series([10.0, 10.5, 11.0, 10.8, 11.2])

    model = Model()
    model.fit(series)
    prediction = model.predict_next()

    assert isinstance(prediction, float)
    # The placeholder is anchored to the last value; check it's in a sane range.
    assert 9.0 < prediction < 13.0


def test_model_raises_before_fitting() -> None:
    """Calling predict_next() before fit() must error clearly."""
    model = Model()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_next()


def test_fit_rejects_empty_series() -> None:
    model = Model()
    with pytest.raises(ValueError):
        model.fit(pd.Series(dtype=float))
