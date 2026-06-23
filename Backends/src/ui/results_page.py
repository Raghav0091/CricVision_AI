import streamlit as st

from Backends.src.ui.components import metric_grid, report_history_card
from Backends.src.ui.theme import render_empty_state, render_page_header, render_section_title


def show_results_page():
    render_page_header(
        "Reports",
        "Delivery history and match-style analysis summaries.",
    )

    reports = st.session_state.get("video_analysis_result")
    has_history = bool(reports and reports.get("success"))

    if not has_history:
        render_empty_state(
            "No reports yet",
            "Analyze a delivery or complete a live session to build your report history.",
            action_label="Go to Analyze to create your first report",
        )
    else:
        impact = reports.get("impact_info", {}) or {}
        report_history_card(
            {
                "title": "Latest Delivery Report",
                "timestamp": "Current session",
                "analysis_type": reports.get("analysis_mode", "Full Delivery Analysis"),
                "line": reports.get("estimated_line", "Unknown"),
                "length": reports.get("estimated_length", "Unknown"),
                "tracking_quality": reports.get("overall_tracking_quality", "Unknown"),
                "impact_confidence": impact.get("impact_confidence", "Unknown"),
            }
        )
        if st.button("View Latest Report Details", use_container_width=True):
            st.info("Open Analyze to review the full processed video and detailed report.")

    render_section_title("Insights", "Future history cards will populate here automatically.")
    metric_grid(
        [
            ("Recent Deliveries", "—", "Latest analyzed clips"),
            ("Best Length", "—", "Most effective zone"),
            ("Line Consistency", "—", "Off / middle / leg trend"),
            ("Tracking Quality", "—", "Quality over time"),
        ],
        columns=4,
    )

    st.caption("Persistent report storage will be added with Supabase in a future update.")
