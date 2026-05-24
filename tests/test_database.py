"""Tests for src.database.

These exercise only the local-JSON fallback path so the test suite has no
external dependencies. The Supabase path is exercised in production.

The `temp_fallback_path` fixture used by these tests lives in conftest.py and is
auto-discovered by pytest — see that file for what it does.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from src import database


def test_save_then_load_round_trip(temp_fallback_path: Path) -> None:
    rows = {
        "thing": {
            "item": "thing",
            "prediction": 12.0,
            "last_value": 11.0,
            "predicted_change_pct": 9.09,
        }
    }
    database.save_predictions(rows, as_of=date(2026, 5, 14))

    loaded = database.load_recent_predictions(limit=10)

    assert len(loaded) == 1
    assert loaded[0]["item"] == "thing"
    assert loaded[0]["as_of_date"] == "2026-05-14"


def test_save_appends_to_existing_file(temp_fallback_path: Path) -> None:
    """Two saves on different days should leave two rows in the file."""
    database.save_predictions(
        {"thing": {"item": "thing", "prediction": 1.0, "last_value": 1.0, "predicted_change_pct": 0.0}},
        as_of=date(2026, 5, 13),
    )
    database.save_predictions(
        {"thing": {"item": "thing", "prediction": 2.0, "last_value": 1.0, "predicted_change_pct": 100.0}},
        as_of=date(2026, 5, 14),
    )

    raw = json.loads(temp_fallback_path.read_text())
    assert len(raw) == 2

    loaded = database.load_recent_predictions(limit=10)
    # load_recent_predictions returns newest first.
    assert loaded[0]["as_of_date"] == "2026-05-14"
