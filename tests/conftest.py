"""Shared pytest fixtures.

A `conftest.py` is special: pytest auto-discovers it and any fixtures defined here
become available to every test file in this directory (and subdirectories) without
needing to be imported. It's the idiomatic place to put fixtures that two or more
test files need.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src import settings


@pytest.fixture
def temp_fallback_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the local-JSON fallback to a temporary file per test.

    Why this matters:
      - `tmp_path` is a built-in pytest fixture that gives each test its own
        clean temp directory. Files written here are auto-deleted after the test.
      - `monkeypatch` lets us safely override module-level values for the
        duration of a single test. Outside the test, the original value is
        restored — no leakage between tests.
      - We also clear the SUPABASE_* env vars so the code under test takes the
        local-fallback branch, not the Supabase branch.
    """
    target = tmp_path / "predictions.json"
    monkeypatch.setattr(settings, "LOCAL_FALLBACK_PATH", str(target))
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    return target
