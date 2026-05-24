"""Pipeline entry point.

Orchestrates the full flow:

    extract → preprocess → fit model & predict → postprocess → save

This is what gets run:
  - locally via `make run`
  - daily in production via .github/workflows/daily-run.yml

You usually won't need to edit this file unless you're adding a new stage to the
pipeline. To change WHAT each stage does, edit the file for that stage instead.
"""

from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

from src import database, extractor, output, processor, settings
from src.model import Model

# Load any environment variables defined in a .env file at the repo root.
# Important for the Supabase keys when running locally.
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def run() -> None:
    """Execute one full pipeline run."""
    logger.info("=== Pipeline run starting ===")

    # 1. Extract: pull raw historical data for every item we want to predict.
    raw_data = extractor.extract_data()
    logger.info("Extracted data for %d items", len(raw_data))

    # 2. Preprocess: clean it up.
    cleaned_data = processor.preprocess(raw_data)

    # 3. Model: fit a separate model per item and produce one prediction each.
    #    For a small number of items, training one model per item is the simplest
    #    pattern. If you ever have hundreds or thousands of items, consider
    #    training one shared model with the item identity as a feature.
    predictions: dict[str, float] = {}
    for name, df in cleaned_data.items():
        model = Model()
        model.fit(df["value"])
        predictions[name] = model.predict_next()
        logger.info("Predicted %s → %.4f", name, predictions[name])

    # 4. Postprocess: turn raw predictions into rows ready for storage.
    rows = output.postprocess(predictions, cleaned_data)

    # 5. Persist.
    database.save_predictions(rows)

    logger.info("=== Pipeline run complete ===")


if __name__ == "__main__":
    run()
