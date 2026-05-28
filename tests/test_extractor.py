"""Tests for src.extractor."""

from __future__ import annotations

import pandas as pd
import pytest

from src import extractor, settings


def test_load_training_data_reads_csv(tmp_path, monkeypatch) -> None:
    csv = tmp_path / "train.csv"
    csv.write_text("feature_a,feature_b,target\n1.0,2.0,3.0\n4.0,5.0,6.0\n")
    monkeypatch.setattr(settings, "DATA_PATH", str(csv))

    df = extractor.load_training_data()

    assert list(df.columns) == ["feature_a", "feature_b", "target"]
    assert len(df) == 2


def test_load_inference_data_reads_csv(tmp_path, monkeypatch) -> None:
    csv = tmp_path / "inference.csv"
    csv.write_text("feature_a,feature_b\n1.0,2.0\n3.0,4.0\n")
    monkeypatch.setattr(settings, "INFERENCE_DATA_PATH", str(csv))

    df = extractor.load_inference_data()

    assert list(df.columns) == ["feature_a", "feature_b"]
    assert len(df) == 2


def test_load_csv_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        extractor._load_csv("nonexistent/path.csv")


def test_load_csv_returns_dataframe(tmp_path) -> None:
    csv = tmp_path / "data.csv"
    csv.write_text("a,b,c\n1,x,0.1\n2,y,0.2\n")

    df = extractor._load_csv(str(csv))

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["a", "b", "c"]
