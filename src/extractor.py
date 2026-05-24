"""Data extraction.

Pulls historical data for each item you want to predict.

Default implementation: fetches daily mean temperatures for the cities listed in
settings.ITEMS_TO_PREDICT from the Open-Meteo API. Open-Meteo is free and
requires no API key, which makes it perfect for a starter template.

REPLACE THIS FILE with your own data source:
  - Stock prices → use yfinance, Alpha Vantage, or Polygon
  - Sports stats → use TheSportsDB, ESPN APIs, or scrape
  - Crypto → use CoinGecko or Binance
  - Your own data → load from a CSV, database, or whatever you have

Whatever you do, keep the function signature and return shape the same and the
rest of the pipeline keeps working.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import requests

from src import settings

logger = logging.getLogger(__name__)

# The Open-Meteo forecast endpoint can return up to ~92 days of past data
# without authentication. See: https://open-meteo.com/en/docs
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def extract_data() -> dict[str, pd.DataFrame]:
    """Fetch historical data for every item in settings.ITEMS_TO_PREDICT.

    Returns:
        A dict mapping item name -> DataFrame with two columns:
          - "date" (datetime): the observation date
          - "value" (float):  the observed value

        The rest of the pipeline (preprocessor, model, dashboard) assumes this
        shape. If you change it, update everything downstream too.
    """
    results: dict[str, pd.DataFrame] = {}

    for name, meta in settings.ITEMS_TO_PREDICT:
        logger.info("Extracting data for %s", name)
        df = _fetch_open_meteo(
            latitude=meta["latitude"],
            longitude=meta["longitude"],
            past_days=settings.LOOKBACK_DAYS,
        )
        results[name] = df

    return results


def _fetch_open_meteo(latitude: float, longitude: float, past_days: int) -> pd.DataFrame:
    """Pull the last `past_days` days of daily mean temperature for one location.

    This is the only place that knows about Open-Meteo specifically. When you
    swap data sources, this is the function you'll replace.
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=past_days)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_mean",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timezone": "UTC",
    }
    response = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    # Open-Meteo returns parallel arrays: daily.time[i] corresponds to
    # daily.temperature_2m_mean[i]. We reshape into a tidy DataFrame.
    daily = payload["daily"]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]),
            "value": daily["temperature_2m_mean"],
        }
    )
    return df
