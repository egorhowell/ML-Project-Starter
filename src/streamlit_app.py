"""Streamlit dashboard.

Displays the latest predictions and lets the user explore historical data for
each item. This is what gets deployed to your VPS so other people can see your
work in a browser.

Run locally with:    make dashboard
Opens at:            http://localhost:8501
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src import database, extractor, settings

load_dotenv()

# Streamlit page setup. Wide layout gives the charts more room to breathe.
st.set_page_config(
    page_title=settings.DASHBOARD_TITLE,
    layout="wide",
)


def main() -> None:
    st.title(settings.DASHBOARD_TITLE)
    st.caption(settings.DASHBOARD_SUBTITLE)

    rows = database.load_recent_predictions(limit=500)
    if not rows:
        _render_empty_state()
        return

    df = pd.DataFrame(rows)

    # Latest "as of" date present in the data — what we'll show in the summary cards.
    latest_as_of = df["as_of_date"].max()
    latest = df[df["as_of_date"] == latest_as_of]

    st.subheader(f"Latest predictions ({latest_as_of})")
    _render_summary_cards(latest)

    st.divider()

    st.subheader("Historical view")
    item_names = [name for name, _ in settings.ITEMS_TO_PREDICT]
    selected = st.selectbox("Pick an item", item_names)
    _render_history_chart(selected, df)


def _render_summary_cards(latest: pd.DataFrame) -> None:
    """Show one card per item with prediction, last value, and predicted change."""
    cols = st.columns(len(latest))
    for col, (_, row) in zip(cols, latest.iterrows()):
        with col:
            delta_str = (
                f"{row['predicted_change_pct']:+.2f}%"
                if row.get("predicted_change_pct") is not None
                else "—"
            )
            st.metric(
                label=row["item"],
                value=f"{row['prediction']:.2f}",
                delta=delta_str,
            )
            st.caption(f"Last actual: {row['last_value']:.2f}")


def _render_history_chart(item_name: str, predictions_df: pd.DataFrame) -> None:
    """Plot actuals (from the data source) overlaid with the recent predictions."""
    # Pull fresh history so the chart isn't limited to predictions stored in the DB.
    # For a real-time dashboard you might cache this — Streamlit makes that easy
    # with @st.cache_data, but we're keeping it simple for the template.
    meta = dict(settings.ITEMS_TO_PREDICT).get(item_name)
    if not meta:
        st.error(f"No metadata found for {item_name!r} in settings.ITEMS_TO_PREDICT")
        return

    raw = extractor._fetch_open_meteo(  # type: ignore[attr-defined]
        latitude=meta["latitude"],
        longitude=meta["longitude"],
        past_days=settings.LOOKBACK_DAYS,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=raw["date"],
            y=raw["value"],
            mode="lines+markers",
            name="Actual",
        )
    )

    item_predictions = predictions_df[predictions_df["item"] == item_name].copy()
    if not item_predictions.empty:
        item_predictions["as_of_date"] = pd.to_datetime(item_predictions["as_of_date"])
        fig.add_trace(
            go.Scatter(
                x=item_predictions["as_of_date"],
                y=item_predictions["prediction"],
                mode="markers",
                name="Prediction",
                marker={"size": 10, "symbol": "diamond"},
            )
        )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_empty_state() -> None:
    """Friendly message when there are no predictions yet."""
    st.info(
        "No predictions stored yet. Run the pipeline once with:\n\n"
        "```\nmake run\n```\n\n"
        "Then refresh this page."
    )


if __name__ == "__main__":
    main()
