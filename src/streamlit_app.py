"""Streamlit dashboard.

Displays recent predictions stored by the pipeline. Deploy to Streamlit
Community Cloud for a free public URL.

Run locally:  make dashboard
Opens at:     http://localhost:8501
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src import database, settings

load_dotenv()

# On Streamlit Community Cloud, credentials live in st.secrets. Copy them into
# env vars so database.py can reach them via os.getenv, same as the pipeline.
try:
    for _key, _val in st.secrets.items():
        os.environ.setdefault(_key, str(_val))
except Exception:
    pass

st.set_page_config(page_title=settings.DASHBOARD_TITLE, layout="wide")


def main() -> None:
    st.title(settings.DASHBOARD_TITLE)
    st.caption(settings.DASHBOARD_SUBTITLE)

    rows = database.load_recent_predictions(limit=500)
    if not rows:
        _render_empty_state()
        return

    df = pd.DataFrame(rows)

    st.subheader("Recent predictions")
    _render_summary_stats(df)

    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        _render_prediction_distribution(df)
    with col_right:
        _render_predictions_over_time(df)

    st.divider()

    st.subheader("Raw data")
    st.dataframe(df, use_container_width=True)


def _render_summary_stats(df: pd.DataFrame) -> None:
    """Show high-level stats about the prediction column."""
    preds = pd.to_numeric(df["prediction"], errors="coerce").dropna()
    if preds.empty:
        st.info("prediction column contains no numeric values.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Mean prediction", f"{preds.mean():.4g}")
    c3.metric("Min", f"{preds.min():.4g}")
    c4.metric("Max", f"{preds.max():.4g}")


def _render_prediction_distribution(df: pd.DataFrame) -> None:
    """Histogram of the prediction column."""
    preds = pd.to_numeric(df["prediction"], errors="coerce").dropna()
    if preds.empty:
        return
    fig = px.histogram(preds, nbins=30, title="Prediction distribution", labels={"value": "prediction"})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def _render_predictions_over_time(df: pd.DataFrame) -> None:
    """Line chart of predictions over time (requires a predicted_at column)."""
    if "predicted_at" not in df.columns:
        return
    df = df.copy()
    df["predicted_at"] = pd.to_datetime(df["predicted_at"], errors="coerce")
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df = df.dropna(subset=["predicted_at", "prediction"]).sort_values("predicted_at")
    if df.empty:
        return
    fig = px.line(df, x="predicted_at", y="prediction", title="Predictions over time")
    st.plotly_chart(fig, use_container_width=True)


def _render_empty_state() -> None:
    st.info(
        "No predictions stored yet. Train a model and run inference with:\n\n"
        "```\nmake train\nmake predict\n```\n\n"
        "Then refresh this page."
    )


if __name__ == "__main__":
    main()
