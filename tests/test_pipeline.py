"""Integration test for the full pipeline.

Unit tests check that each module behaves correctly in isolation. Integration
tests check that the modules behave correctly *together* — wired up the way
production wires them.

The trick to a good integration test:
  - DO mock out external services (network APIs, real databases) so the test
    is fast and offline.
  - DON'T mock out anything you're trying to test. Here we want to test that
    main.run() correctly orchestrates extract → process → model → output → save,
    so we let all of those run with real code.

Pattern: mock at the *boundary* of the system (the extractor's data source, the
database's external store), not inside the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import main


def test_full_pipeline_writes_predictions_to_storage(
    temp_fallback_path: Path,
) -> None:
    """Run the full pipeline against a mocked extractor and verify the result
    lands in storage with the right shape.

    `temp_fallback_path` (from conftest.py) does two things:
      - Points the local-JSON fallback at a temp file
      - Clears Supabase env vars so the pipeline uses the local fallback

    `patch` replaces `src.extractor.extract_data` for the duration of the test,
    returning predictable fake data instead of hitting Open-Meteo over HTTP.
    """
    fake_extracted = {
        "TestItem": pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
                ),
                "value": [10.0, 10.5, 11.0, 10.8, 11.2],
            }
        ),
    }

    with patch("src.extractor.extract_data", return_value=fake_extracted):
        main.run()

    # The pipeline should have written exactly one row (one item) to the fallback file.
    rows = json.loads(temp_fallback_path.read_text())
    assert len(rows) == 1

    row = rows[0]
    # Schema check: every field the dashboard expects should be present.
    assert row["item"] == "TestItem"
    assert isinstance(row["prediction"], float)
    assert row["last_value"] == 11.2
    assert isinstance(row["predicted_change_pct"], float)
    assert "as_of_date" in row
