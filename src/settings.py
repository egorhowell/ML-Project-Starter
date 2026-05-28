"""Configuration — the project's control panel.

Anything you'd tweak without touching pipeline logic lives here.
"""

from __future__ import annotations

# ── Data ──────────────────────────────────────────────────────────────────────
# Path to the labeled CSV used for training.
DATA_PATH: str = "data/train.csv"
# Path to the unlabeled CSV used for inference (no target column required).
INFERENCE_DATA_PATH: str = "data/inference.csv"

# ── Features & target ─────────────────────────────────────────────────────────
# Name of the column the model should learn to predict.
TARGET_COLUMN: str = "target"
# Columns to use as model inputs. Leave empty to use every column except TARGET_COLUMN.
FEATURE_COLUMNS: list[str] = []

# ── Train / validation split ──────────────────────────────────────────────────
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42

# ── MLflow experiment tracking ────────────────────────────────────────────────
# Set MLFLOW_TRACKING_URI to a remote URI (e.g. "http://mlflow:5000") for a
# shared server. The default writes runs locally to mlruns/.
MLFLOW_EXPERIMENT_NAME: str = "ml-project-starter"
MLFLOW_TRACKING_URI: str = "mlruns"

# ── Storage ───────────────────────────────────────────────────────────────────
SUPABASE_TABLE_NAME: str = "predictions"
LOCAL_FALLBACK_PATH: str = "data/predictions.json"

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_TITLE: str = "ML Project Starter"
DASHBOARD_SUBTITLE: str = (
    "Replace this with a one-liner describing what your model predicts."
)
