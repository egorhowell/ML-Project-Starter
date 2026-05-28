"""Pipeline entry point.

Two modes:

    python -m src.main train    → load data, train model, evaluate, save to disk
    python -m src.main predict  → load inference data, load saved model, predict, store results

Run locally:    make train   /   make predict
Run in CI/CD:   .github/workflows/daily-run.yml (predict mode on a schedule)

You usually won't need to edit this file. To change what each stage does, edit
the file for that stage instead (extractor.py, processor.py, model.py, …).
To add MLflow experiment tracking, swap Model() / model.fit() for
train_with_tracking() in the train() function below.
"""

from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

from src import database, extractor, output, processor
from src.model import Model

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def train() -> None:
    """Load labeled data, train a model, evaluate it, and save it to disk."""
    logger.info("=== Training run starting ===")

    raw = extractor.load_training_data()
    logger.info("Loaded %d rows for training", len(raw))

    X, y = processor.preprocess(raw)
    logger.info("Features: %s | Target: %s", list(X.columns), y.name)

    model = Model()
    # To log params/metrics to MLflow, replace the two lines above with:
    #   from src.model import train_with_tracking
    #   model, metrics = train_with_tracking(X, y)
    metrics = model.fit(X, y)
    logger.info("Metrics: %s", metrics)

    model.save()
    logger.info("=== Training complete ===")


def predict() -> None:
    """Load inference data, run the saved model, and store predictions."""
    logger.info("=== Prediction run starting ===")

    raw = extractor.load_inference_data()
    logger.info("Loaded %d rows for inference", len(raw))

    X = processor.preprocess_features(raw)

    model = Model.load()
    preds = model.predict(X)

    rows = output.format_predictions(raw, preds)
    database.save_predictions(rows)

    logger.info("Stored %d predictions", len(rows))
    logger.info("=== Prediction run complete ===")


_MODES = {"train": train, "predict": predict}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "predict"
    if mode not in _MODES:
        sys.exit(f"Unknown mode {mode!r}. Choose from: {list(_MODES)}")
    _MODES[mode]()
