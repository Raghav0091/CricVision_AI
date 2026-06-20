import streamlit as st

from Backends.src.ui.ui_components import empty_state, metric_card, page_header, section_header


def show_results_page():
    page_header(
        "Results History",
        "Saved bowling sessions and delivery analytics will appear here as history builds up.",
    )

    has_history = False

    if not has_history:
        empty_state(
            title="No delivery history yet",
            message=(
                "Analyze a video or complete a live session to start building your delivery history. "
                "Future updates will store sessions in Supabase for trend tracking."
            ),
            icon="📊",
        )

    section_header("Coming Soon")
    future_cols = st.columns(4)

    with future_cols[0]:
        metric_card("Recent Deliveries", "—", "Latest analyzed clips and reports")

    with future_cols[1]:
        metric_card("Best Length", "—", "Most effective length zone")

    with future_cols[2]:
        metric_card("Line Consistency", "—", "Off / middle / leg distribution")

    with future_cols[3]:
        metric_card("Tracking Quality Trend", "—", "Detection quality over time")

    st.info("Database connection will be added later using Supabase.")
