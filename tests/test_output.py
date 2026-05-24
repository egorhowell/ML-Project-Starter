"""Tests for src.output."""

from __future__ import annotations

import pandas as pd

from src import output


def test_postprocess_passes_through_predictions() -> None:
    predictions = {"thing": 12.0}
    raw_data = {
        "thing": pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                "value": [10.0, 11.0],
            }
        )
    }

    rows = output.postprocess(predictions, raw_data)

    assert rows["thing"]["item"] == "thing"
    assert rows["thing"]["prediction"] == 12.0
    assert rows["thing"]["last_value"] == 11.0
    # Predicted change from 11 → 12 is roughly +9.09%.
    assert abs(rows["thing"]["predicted_change_pct"] - 9.09) < 0.01
