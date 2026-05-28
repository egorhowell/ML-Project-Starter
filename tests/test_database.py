"""Tests for src.database.

Exercises only the local-JSON fallback path so the test suite has no
external dependencies. The `temp_fallback_path` fixture (conftest.py) redirects
writes to a temp file and clears Supabase env vars.
"""

from __future__ import annotations

import json
from pathlib import Path

from src import database


def test_save_then_load_round_trip(temp_fallback_path: Path) -> None:
    rows = [{"item": "thing", "prediction": 12.0, "predicted_at": "2026-05-14T00:00:00+00:00"}]

    database.save_predictions(rows)
    loaded = database.load_recent_predictions(limit=10)

    assert len(loaded) == 1
    assert loaded[0]["item"] == "thing"
    assert loaded[0]["prediction"] == 12.0


def test_save_appends_to_existing_file(temp_fallback_path: Path) -> None:
    database.save_predictions([{"prediction": 1.0}])
    database.save_predictions([{"prediction": 2.0}])

    raw = json.loads(temp_fallback_path.read_text())
    assert len(raw) == 2


def test_load_recent_respects_limit(temp_fallback_path: Path) -> None:
    rows = [{"prediction": float(i)} for i in range(10)]
    database.save_predictions(rows)

    loaded = database.load_recent_predictions(limit=3)

    assert len(loaded) == 3
    # Should return the last 3 rows (most recently written).
    assert [r["prediction"] for r in loaded] == [7.0, 8.0, 9.0]


def test_load_returns_empty_list_when_no_file(temp_fallback_path: Path) -> None:
    assert database.load_recent_predictions() == []
