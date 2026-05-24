"""The model.

THIS IS THE ONE FILE YOU ARE DEFINITELY GOING TO REPLACE.

The default `Model` here is a deliberately-trivial placeholder: it predicts
tomorrow's value as today's value plus a small amount of random noise. It's
useful for one purpose only — proving the rest of the pipeline plumbs through.
Once you see the dashboard show predictions, you'll want to swap this out for
something that actually learns from the data.

A few starting points for your real model:
  - Time-series forecasting → Facebook Prophet, statsmodels, sktime, NeuralProphet
  - Regression → scikit-learn (linear, random forest, gradient boosting)
  - Classification → scikit-learn, XGBoost, LightGBM
  - Deep learning → PyTorch or TensorFlow
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class Model:
    """Placeholder random-walk model.

    REPLACE ME. The interface this class exposes (`fit`, `predict_next`) is what
    the rest of the pipeline expects — if you keep those two method signatures
    the same, the pipeline will work with any model you swap in.
    """

    def __init__(self) -> None:
        # After `fit`, these hold the values needed to make a prediction.
        # In a real model, these would be the trained parameters (weights, etc).
        self._last_value: float | None = None
        self._noise_scale: float = 0.0

    def fit(self, series: pd.Series) -> None:
        """Train the model on a single time-series.

        For this placeholder, "training" just means remembering the most recent
        value and how volatile the series has been recently. A real model would
        do gradient descent, tree splitting, or whatever its algorithm requires.

        Args:
            series: a chronologically-ordered pd.Series of historical values.
        """
        if len(series) == 0:
            raise ValueError("Cannot fit on an empty series.")

        self._last_value = float(series.iloc[-1])

        # Use the standard deviation of day-over-day changes as our noise scale.
        # Anchoring noise to the data keeps the random walk in a believable range.
        daily_changes = series.diff().dropna()
        self._noise_scale = float(daily_changes.std()) if len(daily_changes) > 1 else 0.0

    def predict_next(self) -> float:
        """Predict the next value in the series.

        Returns:
            The predicted next-step value, as a float.
        """
        if self._last_value is None:
            raise RuntimeError("Model not fitted. Call .fit(series) before .predict_next().")

        # Random walk: last value plus a small random kick.
        # `np.random.normal(0, sigma)` draws from a Gaussian centred on zero.
        # We use half the daily-change stdev so predictions feel plausible.
        noise = float(np.random.normal(0, self._noise_scale * 0.5))
        return self._last_value + noise
