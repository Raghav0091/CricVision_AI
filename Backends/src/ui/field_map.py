"""Field setup and wagon-wheel visualization page."""

import math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, Wedge
import pandas as pd
import streamlit as st

from Backends.src.analysis.field_zones import (
    DEPTH_OPTIONS,
    DETAILED_FIELD_ZONES,
    FIELD_PRESETS,
    classify_shot_zone,
    create_default_fielders,
    get_active_field_setup,
    get_position_defaults,
    set_active_field_setup,
)
from Backends.src.ui.ui_components import badge_row, page_header, section_header, status_badge


def polar_to_xy(angle_degrees, radius=0.8):
    radians = math.radians(angle_degrees)
    return math.sin(radians) * radius, math.cos(radians) * radius


def draw_field_map(shot_angle=None, selected_zone="Unknown", fielders=None):
    fielders = fielders or []
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#07111f")
    ax.set_facecolor("#0c5f3b")
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)

    boundary = Ellipse((0, 0), 2.0, 1.82, facecolor="#137a46", edgecolor="#d8f3dc", linewidth=2.2)
    inner_ring = Ellipse((0, 0), 1.24, 1.12, facecolor="none", edgecolor="#bae6fd", linewidth=1.2, linestyle="--")
    ground = boundary
    ax.add_patch(ground)
    ax.add_patch(inner_ring)

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
        color = "#2dd4bf" if selected_zone in {zone_name, "Third Man", "Gully"} else "#94a3b8"
        wedge = Wedge(
            (0, 0),
            1.0,
            90 - end_angle,
            90 - start_angle,
            width=0.98,
            alpha=0.15 if color == "#2dd4bf" else 0.08,
            facecolor=color,
            edgecolor="#d8f3dc",
            linewidth=0.7,
        )
        ax.add_patch(wedge)
        label_x, label_y = polar_to_xy((start_angle + end_angle) / 2, 0.73)
        ax.text(label_x, label_y, zone_name, ha="center", va="center", fontsize=8, color="#f8fafc")

    for angle in range(0, 360, 45):
        line_x, line_y = polar_to_xy(angle, 0.96)
        ax.plot([0, line_x], [0, line_y], color="#e2e8f0", alpha=0.18, linewidth=0.8)

    pitch = Rectangle((-0.055, -0.24), 0.11, 0.48, facecolor="#d6b47a", edgecolor="#fef3c7", linewidth=1.5)
    ax.add_patch(pitch)
    ax.scatter([0], [0.11], s=42, color="#f8fafc", zorder=5)
    ax.text(0, 0.155, "Batter", ha="center", va="bottom", fontsize=8, color="#f8fafc")
    ax.scatter([0], [-0.11], s=32, color="#93c5fd", zorder=5)
    ax.text(0, -0.155, "Non-striker", ha="center", va="top", fontsize=8, color="#dbeafe")
    ax.scatter([0.12], [0.02], s=24, color="#facc15", zorder=5)
    ax.text(0.14, 0.02, "Umpire", ha="left", va="center", fontsize=7, color="#fef9c3")
    ax.scatter([-0.62], [0.0], s=24, color="#facc15", zorder=5)
    ax.text(-0.64, -0.04, "Square leg umpire", ha="right", va="top", fontsize=7, color="#fef9c3")
    ax.text(0.52, 0.98, "Off side", ha="center", va="center", fontsize=9, color="#bae6fd")
    ax.text(-0.52, 0.98, "Leg side", ha="center", va="center", fontsize=9, color="#fecaca")
    ax.text(0, -0.98, "Boundary", ha="center", va="center", fontsize=8, color="#f8fafc")
    ax.text(0.0, 0.62, "30-yard circle / inner ring", ha="center", va="center", fontsize=8, color="#bae6fd")
    ax.text(0.075, 0.0, "Pitch", ha="left", va="center", fontsize=8, color="#422006")

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
        try:
            marker_x = float(fielder.get("x", 0))
            marker_y = float(fielder.get("y", 0))
        except (TypeError, ValueError):
            continue

        name = fielder.get("name", "Fielder")
        ax.scatter([marker_x], [marker_y], s=72, color="#fde047", edgecolor="#111827", zorder=9)
        ax.text(marker_x, marker_y - 0.052, name, ha="center", va="top", fontsize=7.5, color="#fefce8")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    return fig


def normalize_fielder_table(fielders):
    rows = []

    for fielder in fielders:
        rows.append(
            {
                "name": fielder.get("name", ""),
                "position": fielder.get("position", fielder.get("zone", "Cover")),
                "zone": fielder.get("zone", "Cover"),
                "depth": fielder.get("depth", "inner ring"),
                "x": float(fielder.get("x", 0)),
                "y": float(fielder.get("y", 0)),
            }
        )

    return rows


def field_setup_editor(
    key_prefix,
    default_preset="Attacking Test Field",
    default_handedness="Right-hand batter",
):
    active_setup = get_active_field_setup()
    preset_key = f"{key_prefix}_field_preset"
    fielders_key = f"{key_prefix}_fielders"
    previous_preset_key = f"{key_prefix}_previous_preset"

    initial_preset = active_setup.get("preset", default_preset)
    if initial_preset not in FIELD_PRESETS:
        initial_preset = default_preset

    preset_name = st.selectbox("Field preset", FIELD_PRESETS, index=FIELD_PRESETS.index(initial_preset), key=preset_key)
    initial_handedness = active_setup.get("batter_handedness", default_handedness)
    batter_handedness = st.selectbox(
        "Batter handedness",
        ["Right-hand batter", "Left-hand batter"],
        index=0 if initial_handedness.startswith("Right") else 1,
        key=f"{key_prefix}_batter_handedness",
    )
    initial_bowler_arm = active_setup.get("bowler_arm", "Right-arm bowler")
    bowler_arm = st.selectbox(
        "Bowler arm",
        ["Right-arm bowler", "Left-arm bowler"],
        index=0 if initial_bowler_arm.startswith("Right") else 1,
        key=f"{key_prefix}_bowler_arm",
    )
    initial_camera_view = active_setup.get("camera_view", "Behind bowler")
    camera_options = ["Behind bowler", "Behind batter", "Side-on"]
    camera_view = st.selectbox(
        "Camera view",
        camera_options,
        index=camera_options.index(initial_camera_view) if initial_camera_view in camera_options else 0,
        key=f"{key_prefix}_camera_view",
    )

    if (
        fielders_key not in st.session_state
        or st.session_state.get(previous_preset_key) != preset_name
        and preset_name != "Custom"
    ):
        if st.session_state.get(previous_preset_key) is None and active_setup.get("fielders"):
            st.session_state[fielders_key] = active_setup["fielders"]
        else:
            st.session_state[fielders_key] = create_default_fielders(preset_name, batter_handedness)
        st.session_state[previous_preset_key] = preset_name

    if fielders_key not in st.session_state:
        st.session_state[fielders_key] = create_default_fielders("Attacking Test Field", batter_handedness)

    previous_rows = normalize_fielder_table(st.session_state[fielders_key])
    edited_df = st.data_editor(
        pd.DataFrame(previous_rows),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "position": st.column_config.SelectboxColumn("position", options=DETAILED_FIELD_ZONES),
            "zone": st.column_config.SelectboxColumn("zone", options=DETAILED_FIELD_ZONES),
            "depth": st.column_config.SelectboxColumn("depth", options=DEPTH_OPTIONS),
            "x": st.column_config.NumberColumn("x", min_value=-1.0, max_value=1.0, step=0.05),
            "y": st.column_config.NumberColumn("y", min_value=-1.0, max_value=1.0, step=0.05),
        },
        key=f"{key_prefix}_fielder_editor",
    )
    fielders = []

    for index, row in enumerate(edited_df.to_dict("records")):
        previous = previous_rows[index] if index < len(previous_rows) else {}
        position = row.get("position") or "Cover"
        depth = row.get("depth") or get_position_defaults(position)["depth"]
        position_changed = position != previous.get("position")
        depth_changed = depth != previous.get("depth")
        x_changed = row.get("x") != previous.get("x")
        y_changed = row.get("y") != previous.get("y")
        defaults = get_position_defaults(position, depth)

        if position_changed or depth_changed:
            row["zone"] = defaults["zone"]

            if not x_changed:
                row["x"] = defaults["x"]
            if not y_changed:
                row["y"] = defaults["y"]

        fielders.append(row)

    st.session_state[fielders_key] = fielders

    field_setup = {
        "preset": preset_name,
        "batter_handedness": batter_handedness,
        "bowler_arm": bowler_arm,
        "camera_view": camera_view,
        "fielders": fielders,
    }

    if st.button("Save Current Field Setup", key=f"{key_prefix}_save_field_setup"):
        field_setup.pop("is_default_setup", None)
        set_active_field_setup(field_setup)
        st.success("Field setup saved to outputs/field_setups/latest_field_setup.json")

    return field_setup


def show_field_map_page():
    page_header(
        "Field Map",
        "Set an 11-player cricket field and preview wagon-wheel shot direction zones.",
    )

    control_col, map_col = st.columns([1.05, 1.25])

    with control_col:
        section_header("Field Setup")
        field_setup = field_setup_editor("field_map", default_preset="Attacking Test Field")

        section_header("Shot Direction Preview")
        shot_angle = st.slider("Shot angle", min_value=0, max_value=359, value=45, step=1)
        selected_zone = classify_shot_zone(shot_angle, field_setup["batter_handedness"])
        badge_row(
            [
                status_badge(f"Zone: {selected_zone}", "cyan"),
                status_badge(f"Angle: {shot_angle} deg", "blue"),
            ]
        )

    with map_col:
        section_header("Top-Down Field")
        st.pyplot(
            draw_field_map(
                shot_angle=shot_angle,
                selected_zone=selected_zone,
                fielders=field_setup["fielders"],
            )
        )
        st.caption("Coordinates use a normalized top-down field: x/y range from -1.0 to 1.0.")
