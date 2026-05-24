"""Tests for src.processor.

This file demonstrates two common pytest patterns:
  1. Plain test functions — see `test_preprocess_drops_null_rows` and
     `test_preprocess_sorts_by_date` below.
  2. Parametrized tests — see `test_preprocess_handles_various_null_patterns`,
     which runs the same test body against several different inputs without
     duplicating the setup code.

When you find yourself copy-pasting a test and only changing the input values,
that's the signal to switch to parametrize.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import processor


def test_preprocess_drops_null_rows() -> None:
    raw = {
        "thing": pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
                "value": [1.0, None, 3.0],
            }
        )
    }

    cleaned = processor.preprocess(raw)

    assert len(cleaned["thing"]) == 2
    assert cleaned["thing"]["value"].isna().sum() == 0


def test_preprocess_sorts_by_date() -> None:
    raw = {
        "thing": pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-02"]),
                "value": [3.0, 1.0, 2.0],
            }
        )
    }

    cleaned = processor.preprocess(raw)

    assert list(cleaned["thing"]["value"]) == [1.0, 2.0, 3.0]


# Each tuple in the list below becomes one separate test case. pytest names each
# one using the `id=` slug at the end, so failures point you at the exact scenario.
@pytest.mark.parametrize(
    "input_values, expected_count",
    [
        pytest.param([1.0, 2.0, 3.0], 3, id="no-nulls"),
        pytest.param([1.0, None, 3.0], 2, id="single-null-in-middle"),
        pytest.param([None, None, None], 0, id="all-nulls"),
        pytest.param([1.0], 1, id="single-value"),
    ],
)
def test_preprocess_handles_various_null_patterns(
    input_values: list[float | None], expected_count: int
) -> None:
    """Same logic as test_preprocess_drops_null_rows, but across four scenarios.

    This is the value of parametrize: the test body is written once and pytest
    runs it once per parameter set, reporting them as four separate results.
    """
    n_rows = len(input_values)
    raw = {
        "thing": pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=n_rows, freq="D"),
                "value": input_values,
            }
        )
    }

    cleaned = processor.preprocess(raw)

    assert len(cleaned["thing"]) == expected_count
