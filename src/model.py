"""The model.

Replace or extend this file with your own estimator. The interface the pipeline
expects:

    metrics = model.fit(X, y)   → dict of evaluation metrics
    preds   = model.predict(X)  → np.ndarray of predictions

The default is a scikit-learn Pipeline combining automatic preprocessing
(StandardScaler for numeric columns, OneHotEncoder for categoricals) with a
LinearRegression estimator. Swap the estimator in _build_estimator() to change
the algorithm without touching anything else.

Common swaps:
  Regression     → Ridge, Lasso, RandomForestRegressor, XGBRegressor, LGBMRegressor
  Classification → LogisticRegression, RandomForestClassifier, XGBClassifier, LGBMClassifier
  Unsupervised   → KMeans, DBSCAN (skip fit(X, y) — call fit(X) directly on the estimator)
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

MODEL_DIR = Path("data")
MODEL_PATH = MODEL_DIR / "model.joblib"


class Model:
    """sklearn Pipeline: auto-preprocessing + configurable estimator.

    Works out of the box for numeric-only, categorical-only, or mixed-type
    feature matrices without any manual column configuration.
    """

    def __init__(self) -> None:
        self._pipeline = _build_pipeline()
        self._fitted = False

    # ------------------------------------------------------------------
    # Pipeline interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Fit on (X, y) and return held-out validation metrics.

        Args:
            X: feature DataFrame (any mix of numeric and string columns).
            y: target Series (numeric for regression, string/int for classification).

        Returns:
            dict of metric name → value evaluated on the held-out split.
        """
        from src import settings  # local import to keep Model decoupled from settings

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=settings.TEST_SIZE,
            random_state=settings.RANDOM_STATE,
        )

        self._pipeline.fit(X_train, y_train)
        self._fitted = True

        y_pred = self._pipeline.predict(X_val)
        metrics = _evaluate(y_val.to_numpy(), y_pred)
        logger.info("Val metrics: %s", metrics)
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predictions for every row in X.

        Args:
            X: feature DataFrame with the same columns seen during fit().

        Returns:
            1-D numpy array of predictions.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict().")
        return self._pipeline.predict(X)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> None:
        """Serialise the fitted pipeline to disk with joblib."""
        dest = path or MODEL_PATH
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, dest)
        logger.info("Saved model → %s", dest)

    @classmethod
    def load(cls, path: Path | None = None) -> "Model":
        """Load a previously saved pipeline from disk."""
        src_path = path or MODEL_PATH
        m = cls()
        m._pipeline = joblib.load(src_path)
        m._fitted = True
        return m


# ------------------------------------------------------------------
# MLflow helper — opt-in experiment tracking
# ------------------------------------------------------------------

def train_with_tracking(
    X: pd.DataFrame, y: pd.Series
) -> tuple[Model, dict[str, float]]:
    """Train a Model and log params + metrics to MLflow.

    Call this from main.py instead of constructing Model() directly when you
    want experiment tracking. Writes runs to the URI in settings.MLFLOW_TRACKING_URI
    (default: local mlruns/ directory).

    Returns:
        (fitted model, metrics dict)
    """
    import mlflow  # lazy import — keeps mlflow optional for basic usage

    from src import settings

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_params({
            "estimator": type(_build_estimator()).__name__,
            "test_size": settings.TEST_SIZE,
            "n_features": X.shape[1],
            "n_samples": len(X),
        })
        model = Model()
        metrics = model.fit(X, y)
        mlflow.log_metrics(metrics)

    return model, metrics


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _build_pipeline() -> Pipeline:
    """Assemble the full preprocessing + estimator Pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                make_column_selector(dtype_include=np.number),
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                make_column_selector(dtype_include=object),
            ),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("estimator", _build_estimator()),
    ])


def _build_estimator():
    """Return the sklearn estimator. Swap this line to change the algorithm."""
    return LinearRegression()
    # ── Regression alternatives ───────────────────────────────────────
    # from sklearn.linear_model import Ridge, Lasso
    # return Ridge(alpha=1.0)
    # from sklearn.ensemble import RandomForestRegressor
    # return RandomForestRegressor(n_estimators=100, random_state=42)
    # from xgboost import XGBRegressor
    # return XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    # from lightgbm import LGBMRegressor
    # return LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    # ── Classification alternatives ───────────────────────────────────
    # from sklearn.linear_model import LogisticRegression
    # return LogisticRegression(max_iter=1000)
    # from sklearn.ensemble import RandomForestClassifier
    # return RandomForestClassifier(n_estimators=100, random_state=42)
    # from xgboost import XGBClassifier
    # return XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute metrics appropriate for the target dtype."""
    if np.issubdtype(y_true.dtype, np.number):
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }
