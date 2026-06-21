import streamlit as st

from Backends.src.models.model_registry import validate_model_paths
from Backends.src.ui.components import hero_section, metric_grid, primary_action_card, render_feature_card
from Backends.src.ui.theme import render_page_header, render_section_title


def _count_ready_models():
    statuses = validate_model_paths().values()
    ready = sum(1 for item in statuses if item["found"])
    return ready, len(list(statuses))


def show_dashboard():
    render_page_header(
        "CricVision AI",
        "AI-powered cricket performance analysis for players, coaches, academies, and analysts.",
        badge="System Ready",
    )

    hero_section(
        title="Analyze every delivery with confidence",
        subtitle="Professional cricket vision analytics",
        description=(
            "Upload a clip or record live from camera. CricVision automatically tracks the ball, "
            "estimates line and length, and produces a clean delivery report."
        ),
    )

    cta_cols = st.columns(2)
    with cta_cols[0]:
        primary_action_card(
            "Analyze a Delivery",
            "Upload a bowling or full-delivery clip and generate a processed video plus report.",
            "Open Analyze from the sidebar",
        )
    with cta_cols[1]:
        primary_action_card(
            "Start Live Session",
            "Record one clean delivery from your camera and review it after the ball is bowled.",
            "Open Live Session from the sidebar",
        )

    ready_count, total_count = _count_ready_models()
    metric_grid(
        [
            ("Deliveries Analyzed", "—", "History builds after analysis"),
            ("Reports Generated", "—", "Saved reports coming soon"),
            ("Active Models", f"{ready_count}/{total_count}", "Configured model files"),
            ("Tracking Quality", "Auto", "Smart defaults enabled", "Ready"),
        ],
        columns=4,
    )

    render_section_title("What CricVision analyzes")
    feature_cols = st.columns(5)
    features = [
        ("Line & Length", "Off, middle, leg and yorker/full/good/short estimates.", "📏"),
        ("Bounce Point", "Pitch contact and normalized pitch-map view.", "🎯"),
        ("Ball Trajectory", "Continuous tracking with recovery and smoothing.", "📈"),
        ("Bat Impact", "Possible ball-bat contact frame in full delivery mode.", "🏏"),
        ("Field Zones", "Wagon-wheel direction and nearest fielder context.", "🗺️"),
    ]
    for col, (title, description, icon) in zip(feature_cols, features):
        with col:
            render_feature_card(title, description, icon)
