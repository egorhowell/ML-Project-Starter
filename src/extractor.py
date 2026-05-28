"""Data extraction.

Loads data from its source and returns a plain DataFrame. Two entry points:

    load_training_data()  → labeled DataFrame (features + target column)
    load_inference_data() → unlabeled DataFrame (features only)

Default: reads from local CSV files configured in settings.

REPLACE the _load_csv() function (or the two public functions) with your own
data source:
  - Database  → psycopg2, SQLAlchemy, BigQuery, Snowflake
  - REST API  → requests (see _fetch_from_api stub below)
  - Cloud     → boto3 (S3), google-cloud-storage, Azure Blob
  - Streaming → Kafka, Kinesis

Keep the return type (pd.DataFrame) and the rest of the pipeline works unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src import settings

logger = logging.getLogger(__name__)


def load_training_data() -> pd.DataFrame:
    """Load the labeled dataset used for training.

    Returns:
        DataFrame that includes settings.TARGET_COLUMN alongside feature columns.
    """
    logger.info("Loading training data from %s", settings.DATA_PATH)
    return _load_csv(settings.DATA_PATH)


def load_inference_data() -> pd.DataFrame:
    """Load the unlabeled dataset used for inference.

    Returns:
        DataFrame with feature columns only (no target column required).
    """
    logger.info("Loading inference data from %s", settings.INFERENCE_DATA_PATH)
    return _load_csv(settings.INFERENCE_DATA_PATH)


# ------------------------------------------------------------------
# Default backend — CSV file
# ------------------------------------------------------------------

def _load_csv(path: str) -> pd.DataFrame:
    """Read a CSV file and return it as a DataFrame.

    REPLACE this with your real data-fetch logic when you move beyond CSV files.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Data file not found: {p}. "
            "Create the file or replace _load_csv() with your own fetch logic."
        )
    return pd.read_csv(p)


# ------------------------------------------------------------------
# Alternative backend stub — HTTP / REST API
# ------------------------------------------------------------------
# Uncomment and adapt when fetching from an external API.
#
# import requests
#
# def _fetch_from_api(endpoint: str, params: dict) -> pd.DataFrame:
#     response = requests.get(endpoint, params=params, timeout=30)
#     response.raise_for_status()
#     return pd.DataFrame(response.json())
