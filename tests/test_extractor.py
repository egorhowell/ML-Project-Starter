"""Tests for src.extractor.

We mock the Open-Meteo HTTP call so the test suite stays fast and offline.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from src import extractor


def test_fetch_open_meteo_returns_expected_shape() -> None:
    """The fetcher should return a DataFrame with `date` and `value` columns."""
    fake_payload = {
        "daily": {
            "time": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "temperature_2m_mean": [5.1, 5.4, 4.9],
        }
    }

    with patch.object(extractor.requests, "get") as mock_get:
        mock_get.return_value.json.return_value = fake_payload
        mock_get.return_value.raise_for_status.return_value = None

        df = extractor._fetch_open_meteo(latitude=0.0, longitude=0.0, past_days=3)

    assert list(df.columns) == ["date", "value"]
    assert len(df) == 3
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["value"].iloc[0] == 5.1
