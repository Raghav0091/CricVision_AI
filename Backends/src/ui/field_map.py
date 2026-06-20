"""Field map and wagon-wheel visualization page."""

import math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, Wedge
import streamlit as st

from Backends.src.analysis.field_zones import FIELD_ZONES, classify_shot_zone
from Backends.src.ui.ui_components import badge_row, card, page_header, section_header, status_badge


ZONE_ANGLES = {
    "Long Off": 0,
    "Cover": 45,
    "Point": 90,
    "Third Man / Gully": 135,
    "Fine Leg": 180,
    "Square Leg": 225,
    "Mid Wicket": 270,
    "Long On": 315,
}

DEPTH_RADIUS = {
    "close": 0.25,
    "inner ring": 0.52,
    "deep": 0.82,
}


def polar_to_xy(angle_degrees, radius=0.8):
    radians = math.radians(angle_degrees)
    x = math.sin(radians) * radius
    y = math.cos(radians) * radius
    return x, y


def draw_field_map(shot_angle=None, selected_zone="Unknown", fielders=None):
    fielders = fielders or []
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#07111f")
    ax.set_facecolor("#0c5f3b")
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)

    ground = Ellipse((0, 0), 2.0, 2.0, facecolor="#137a46", edgecolor="#d8f3dc", linewidth=2)
    ax.add_patch(ground)

    zone_sequence = [
        ("Long Off", -22.5, 22.5),
        ("Cover", 22.5, 67.5),
        ("Point", 67.5, 112.5),
        ("Third Man / Gully", 112.5, 157.5),
        ("Fine Leg", 157.5, 202.5),
        ("Square Leg", 202.5, 247.5),
        ("Mid Wicket", 247.5, 292.5),
        ("Long On", 292.5, 337.5),
    ]

    for zone_name, start_angle, end_angle in zone_sequence:
        color = "#2dd4bf" if zone_name == selected_zone else "#94a3b8"
        wedge = Wedge(
            (0, 0),
            1.0,
            90 - end_angle,
            90 - start_angle,
            width=0.98,
            alpha=0.14 if zone_name == selected_zone else 0.08,
            facecolor=color,
            edgecolor="#d8f3dc",
            linewidth=0.7,
        )
        ax.add_patch(wedge)
        label_x, label_y = polar_to_xy((start_angle + end_angle) / 2, 0.72)
        ax.text(label_x, label_y, zone_name, ha="center", va="center", fontsize=8, color="#f8fafc")

    pitch = Rectangle((-0.06, -0.22), 0.12, 0.44, facecolor="#d6b47a", edgecolor="#fef3c7", linewidth=1.5)
    ax.add_patch(pitch)
    ax.scatter([0], [0], s=35, color="#f8fafc", zorder=5)

    if shot_angle is not None:
        arrow_x, arrow_y = polar_to_xy(shot_angle, 0.88)
        ax.arrow(
            0,
            0,
            arrow_x,
            arrow_y,
            width=0.012,
            head_width=0.06,
            head_length=0.08,
            length_includes_head=True,
            color="#f97316",
            zorder=8,
        )

    for fielder in fielders:
        zone = fielder.get("zone", "Long Off")
        depth = fielder.get("depth", "inner ring")
        name = fielder.get("name", "Fielder")
        marker_x, marker_y = polar_to_xy(ZONE_ANGLES.get(zone, 0), DEPTH_RADIUS.get(depth, 0.52))
        ax.scatter([marker_x], [marker_y], s=65, color="#fde047", edgecolor="#111827", zorder=9)
        ax.text(marker_x, marker_y - 0.055, name, ha="center", va="top", fontsize=8, color="#fefce8")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    return fig


def show_field_map_page():
    page_header(
        "Field Map",
        "Build a top-down wagon-wheel view with cricket field zones and optional fielder markers.",
    )

    if "field_map_fielders" not in st.session_state:
        st.session_state.field_map_fielders = []

    control_col, map_col = st.columns([0.9, 1.4])

    with control_col:
        section_header("Shot Direction")
        shot_angle = st.slider("Shot angle", min_value=0, max_value=359, value=45, step=1)
        selected_zone = classify_shot_zone(shot_angle)
        badge_row(
            [
                status_badge(f"Zone: {selected_zone}", "cyan"),
                status_badge(f"Angle: {shot_angle} deg", "blue"),
            ]
        )

        section_header("Fielder Placement")
        with st.form("add_fielder_form", clear_on_submit=True):
            fielder_name = st.text_input("Fielder name", value="")
            fielder_zone = st.selectbox("Zone", FIELD_ZONES, index=2)
            fielder_depth = st.selectbox("Depth", ["close", "inner ring", "deep"], index=1)
            add_fielder = st.form_submit_button("Add Fielder")

        if add_fielder and fielder_name.strip():
            st.session_state.field_map_fielders.append(
                {
                    "name": fielder_name.strip(),
                    "zone": fielder_zone,
                    "depth": fielder_depth,
                }
            )
            st.rerun()

        if st.button("Clear Fielders"):
            st.session_state.field_map_fielders = []
            st.rerun()

        if st.session_state.field_map_fielders:
            for index, fielder in enumerate(st.session_state.field_map_fielders, start=1):
                st.write(f"{index}. {fielder['name']} - {fielder['zone']} ({fielder['depth']})")
        else:
            card("No Fielders Added", "Add optional fielder markers to see placement on the map.")

    with map_col:
        section_header("Wagon-Wheel Field Map")
        st.pyplot(
            draw_field_map(
                shot_angle=shot_angle,
                selected_zone=selected_zone,
                fielders=st.session_state.field_map_fielders,
            )
        )
