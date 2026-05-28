"""Integration tests for main.train() and main.predict().

Mocks at the system boundary (extractor + saved model) so tests are fast and
offline while still exercising the real preprocessing → model → output → storage
orchestration.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src import main, settings


def test_train_saves_model_to_disk(regression_df, tmp_path, monkeypatch) -> None:
    """train() should fit a model and persist it."""
    saved: list[Path] = []

    def fake_save(self, path=None):
        dest = path or (tmp_path / "model.joblib")
        import joblib
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, dest)
        saved.append(dest)

    monkeypatch.setattr(settings, "TARGET_COLUMN", "target")
    monkeypatch.setattr(settings, "FEATURE_COLUMNS", [])

    from src.model import Model
    with (
        patch("src.extractor.load_training_data", return_value=regression_df),
        patch.object(Model, "save", fake_save),
    ):
        main.train()

    assert len(saved) == 1 and saved[0].exists()


def test_predict_stores_one_row_per_input(
    regression_df, tmp_path, monkeypatch, temp_fallback_path: Path
) -> None:
    """predict() should store exactly one prediction per inference row."""
    model_path = tmp_path / "model.joblib"
    inference_df = regression_df.drop(columns=["target"]).head(5)

    monkeypatch.setattr(settings, "TARGET_COLUMN", "target")
    monkeypatch.setattr(settings, "FEATURE_COLUMNS", [])

    # First train a model so we have something to load.
    with (
        patch("src.extractor.load_training_data", return_value=regression_df),
        patch("src.model.MODEL_PATH", model_path),
        patch("src.model.MODEL_DIR", tmp_path),
    ):
        main.train()

    with (
        patch("src.extractor.load_inference_data", return_value=inference_df),
        patch("src.model.MODEL_PATH", model_path),
    ):
        main.predict()

    rows = json.loads(temp_fallback_path.read_text())
    assert len(rows) == 5
    assert all("prediction" in r for r in rows)
    assert all("predicted_at" in r for r in rows)
