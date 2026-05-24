"""Storage layer.

Writes predictions to a database so the dashboard can read them. The default
backend is Supabase (a hosted PostgreSQL + auth + storage service with a generous
free tier). If you haven't set up Supabase yet, the pipeline falls back to writing
to a local JSON file so you can still run the demo end-to-end.

ENV VARS THE SUPABASE BACKEND READS:
    SUPABASE_URL — your project URL (https://<project-ref>.supabase.co)
    SUPABASE_KEY — the service-role key (NOT the anon key; the service-role key
                   is required to write to the database)

EXPECTED SUPABASE TABLE SCHEMA (named via settings.SUPABASE_TABLE_NAME):
    Column                  Type           Notes
    ----------------------  -------------  --------------------------------
    id                      int8           primary key, auto-increment
    created_at              timestamptz    default now()
    as_of_date              date
    item                    text
    prediction              float8
    last_value              float8
    predicted_change_pct    float8

The README explains how to create this table from the Supabase dashboard.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from pathlib import Path

from src import settings

logger = logging.getLogger(__name__)


def _supabase_configured() -> bool:
    """True iff both Supabase env vars are present."""
    return bool(os.getenv("SUPABASE_URL")) and bool(os.getenv("SUPABASE_KEY"))


def save_predictions(rows: dict[str, dict], as_of: date | None = None) -> None:
    """Persist today's batch of predictions.

    Args:
        rows:   dict of item name -> prediction fields (as produced by output.postprocess).
        as_of:  the date the predictions are "as of". Defaults to today.

    If Supabase env vars are set, writes to the Supabase table.
    Otherwise writes to a local JSON file at settings.LOCAL_FALLBACK_PATH.
    """
    as_of = as_of or date.today()
    # Add the as_of_date stamp to every row.
    stamped_rows = [{**row, "as_of_date": as_of.isoformat()} for row in rows.values()]

    if _supabase_configured():
        _save_to_supabase(stamped_rows)
    else:
        logger.warning(
            "SUPABASE_URL/SUPABASE_KEY not set — falling back to local JSON at %s. "
            "Set up Supabase before deploying (see README).",
            settings.LOCAL_FALLBACK_PATH,
        )
        _save_to_local_file(stamped_rows)


def load_recent_predictions(limit: int = 100) -> list[dict]:
    """Load the most recent N predictions for the dashboard to display.

    Reads from whichever backend save_predictions wrote to.
    """
    if _supabase_configured():
        return _load_from_supabase(limit)
    return _load_from_local_file(limit)


# -----------------------------------------------------------------------------
# Supabase backend
# -----------------------------------------------------------------------------

def _get_supabase_client():
    """Lazy-import so users running the local-fallback path don't need the
    supabase library installed (it's still a hard dep in pyproject.toml, but
    this way an import error here doesn't break offline development)."""
    from supabase import create_client

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    return create_client(url, key)


def _save_to_supabase(rows: list[dict]) -> None:
    client = _get_supabase_client()
    response = client.table(settings.SUPABASE_TABLE_NAME).insert(rows).execute()
    logger.info("Inserted %d rows into Supabase table %r", len(rows), settings.SUPABASE_TABLE_NAME)
    return response


def _load_from_supabase(limit: int) -> list[dict]:
    client = _get_supabase_client()
    response = (
        client.table(settings.SUPABASE_TABLE_NAME)
        .select("*")
        .order("as_of_date", desc=True)
        .limit(limit)
        .execute()
    )
    return response.data or []


# -----------------------------------------------------------------------------
# Local JSON-file backend (fallback for offline development)
# -----------------------------------------------------------------------------

def _save_to_local_file(rows: list[dict]) -> None:
    path = Path(settings.LOCAL_FALLBACK_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    if path.exists():
        existing = json.loads(path.read_text())

    existing.extend(rows)
    path.write_text(json.dumps(existing, indent=2))
    logger.info("Wrote %d rows to local file %s", len(rows), path)


def _load_from_local_file(limit: int) -> list[dict]:
    path = Path(settings.LOCAL_FALLBACK_PATH)
    if not path.exists():
        return []
    rows = json.loads(path.read_text())
    # Newest first.
    rows.sort(key=lambda r: r.get("as_of_date", ""), reverse=True)
    return rows[:limit]
