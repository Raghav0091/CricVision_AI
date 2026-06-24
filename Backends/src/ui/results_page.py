import streamlit as st

from Backends.src.storage.session_store import get_session_summary, load_session_results
from Backends.src.ui.components import (
    render_results_filters,
    render_session_analytics_sections,
    render_session_result_card,
    render_session_summary,
)
from Backends.src.ui.theme import render_empty_state, render_page_header, render_section_title


def show_results_page():
    render_page_header(
        "Session Results",
        "Review saved delivery, impact, shot, outcome, and agent analysis.",
    )

    load_warning = False
    try:
        results = load_session_results()
    except Exception:
        results = []
        load_warning = True

    if load_warning:
        st.warning("Session history file could not be read. Showing an empty history for now.")

    if not results:
        render_empty_state(
            "No session results yet",
            "Analyze a video or live delivery to start building your CricVision history.",
            action_label="Go to Video Analysis or Live Session",
        )
        return

    summary = get_session_summary(results)
    render_session_summary(summary)
    render_session_analytics_sections(summary)

    render_section_title("Saved Delivery Details", "Filter and open any saved delivery report.")
    filtered_results = render_results_filters(results)
    st.caption(f"Showing {len(filtered_results)} of {len(results)} saved deliveries.")

    if not filtered_results:
        st.info("No saved deliveries match the current filters.")
        return

    for index, result in enumerate(filtered_results):
        render_session_result_card(result, expanded=index == 0)
