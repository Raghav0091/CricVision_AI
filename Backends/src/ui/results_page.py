import streamlit as st

from Backends.src.ui.components import (
    metric_grid,
    render_delivery_report,
    render_impact_frame_preview,
    render_impact_report,
    render_save_status,
    render_shot_report,
    report_history_card,
)
from Backends.src.ui.theme import render_empty_state, render_page_header, render_section_title


def show_results_page():
    render_page_header(
        "Reports",
        "Delivery history and match-style analysis summaries.",
    )

    reports = [
        ("Video Analysis", st.session_state.get("video_analysis_result")),
        ("Live Session", st.session_state.get("live_last_result")),
    ]
    reports = [(label, report) for label, report in reports if report and report.get("success")]
    has_history = bool(reports)

    if not has_history:
        render_empty_state(
            "No reports yet",
            "Analyze a delivery or complete a live session to build your report history.",
            action_label="Go to Analyze to create your first report",
        )
    else:
        for index, (source_label, report) in enumerate(reports):
            impact = report.get("impact_info", {}) or {}
            report_history_card(
                {
                    "title": f"{source_label} Delivery Report",
                    "timestamp": "Current session",
                    "analysis_type": report.get("analysis_mode", source_label),
                    "line": report.get("estimated_line", "Unknown"),
                    "length": report.get("estimated_length", "Unknown"),
                    "tracking_quality": report.get("overall_tracking_quality", "Unknown"),
                    "impact_confidence": impact.get("impact_confidence", "Unknown"),
                }
            )
            with st.expander(f"View {source_label} Report Details", expanded=index == 0):
                render_delivery_report(report)
                render_impact_report(report)
                render_impact_frame_preview(report)
                render_shot_report(report)
                render_save_status(report, source_label)

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
