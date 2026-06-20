import streamlit as st

from Backends.src.ui.ui_components import (
    feature_card,
    hero_section,
    metric_card,
    section_header,
    workflow_step,
)


def show_dashboard():
    hero_section(
        title="🏏 CricVision AI",
        subtitle="Live Bowling Analytics from Umpire View Camera",
        description=(
            "Track ball path, estimate speed, detect pitch point, classify length, "
            "and save every delivery for cricket performance analysis. Built for coaches, "
            "players, and analysts who want computer vision insights from every delivery."
        ),
    )

    section_header("Platform Metrics")
    metric_cols = st.columns(4)

    with metric_cols[0]:
        metric_card("Models Available", "4+", "Ball, stump, ensemble options")

    with metric_cols[1]:
        metric_card("Analysis Modes", "3", "Fast, Balanced, High Precision")

    with metric_cols[2]:
        metric_card("Tracking Features", "6+", "Kalman, ROI, bounce, line & length")

    with metric_cols[3]:
        metric_card("Deployment Ready", "Yes", "Streamlit Cloud compatible")

    section_header("Core Features")
    feature_rows = [
        [
            ("Video Analysis", "Upload bowling clips and generate annotated delivery videos with bounce and pitch insights.", "🎥"),
            ("Live Delivery Capture", "Record one clean delivery from your camera and analyze it after the ball is bowled.", "📹"),
            ("Ball + Stump Detection", "YOLO-based detection pipeline for ball and wicket visibility in real clips.", "🎯"),
        ],
        [
            ("Trajectory Tracking", "Kalman filtering, interpolation, and recovery for continuous ball path tracking.", "📈"),
            ("Line & Length Analysis", "Estimate off/leg/middle line and yorker/full/good/short length from bounce point.", "📏"),
            ("Delivery Report Agent", "AI coaching feedback, quality scoring, and warnings for every analyzed delivery.", "🤖"),
        ],
    ]

    for row in feature_rows:
        cols = st.columns(3)
        for col, (title, description, icon) in zip(cols, row):
            with col:
                feature_card(title, description, icon)

    section_header("Recommended Workflow")
    workflow_cols = st.columns([1, 1])

    with workflow_cols[0]:
        workflow_step(1, "Upload or record a delivery clip from umpire or bowler view.")
        workflow_step(2, "Select the Ball + Stump model for best line and length estimates.")
        workflow_step(3, "Use the Balanced or Fast Bowling preset depending on clip quality.")

    with workflow_cols[1]:
        workflow_step(4, "Run analysis and review the processed video with trajectory overlays.")
        workflow_step(5, "Review the delivery report, pitch map, and export review frames if needed.")
