from pathlib import Path

import streamlit as st

from Backends.src.models.model_registry import validate_model_paths
from Backends.src.ui.components import metric_grid
from Backends.src.ui.theme import render_page_header


REPORTS_DIR = Path("outputs/reports")


def _count_reports():
    if not REPORTS_DIR.exists():
        return 0
    return sum(1 for path in REPORTS_DIR.iterdir() if path.is_file())


def _last_analysis_label():
    video_result = st.session_state.get("video_analysis_result") or {}
    live_result = st.session_state.get("live_last_result") or {}
    if video_result.get("success"):
        return video_result.get("analysis_mode", "Video analysis")
    if live_result.get("success"):
        return "Live session delivery"
    return "None yet"


def _active_model_label():
    settings = st.session_state.get("video_analysis_settings") or {}
    if settings.get("model_name"):
        return settings["model_name"]
    live_model = st.session_state.get("live_session_model")
    if live_model:
        return live_model
    return "Ball + Stump Detector"


def _system_ready_label():
    statuses = list(validate_model_paths().values())
    ready = sum(1 for item in statuses if item["found"] or item.get("remote_available"))
    if ready == len(statuses) and statuses:
        return "Ready"
    if ready > 0:
        return "Partial"
    return "Setup needed"


def show_dashboard():
    render_page_header(
        "CricVision AI",
        "AI-powered cricket performance analysis",
    )

    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button("Start Live Session", type="primary", use_container_width=True, key="dashboard_live"):
            st.session_state["nav_target"] = "Live Session"
            st.rerun()
    with action_cols[1]:
        if st.button("Analyze Video", type="primary", use_container_width=True, key="dashboard_analyze"):
            st.session_state["nav_target"] = "Video Analysis"
            st.rerun()

    metric_grid(
        [
            ("Reports Generated", str(_count_reports()), "Saved analysis reports"),
            ("Last Analysis", _last_analysis_label(), "Most recent completed run"),
            ("Active Model", _active_model_label(), "Current detection model"),
            ("System Ready", _system_ready_label(), "Model files on disk", _system_ready_label()),
        ],
        columns=4,
    )
