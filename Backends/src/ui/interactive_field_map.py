"""Interactive cricket field map with angle/radius editing."""

from __future__ import annotations

import io
import json
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, Wedge
import streamlit as st

from Backends.src.analysis.field_geometry import (
    DEFAULT_VISUAL_ROTATION,
    UMPIRE_POSITIONS_RH,
    angle_to_field_zone,
    ensure_fielder_polar,
    ensure_umpire_polar,
    fielder_display_xy,
    get_side_labels,
    mirror_angle_for_handedness,
    polar_to_screen,
    screen_to_polar,
    umpire_display_xy,
    umpire_from_name,
)
from Backends.src.data.field_presets import PRESET_NAMES, create_preset_fielders

CANVAS_SIZE = 520
FIELD_COORDINATE_VERSION = 3


def _normalize_handedness(handedness):
    if str(handedness or "").lower().startswith("left"):
        return "Left-handed"
    return "Right-handed"


def _display_to_canvas(x, y, size=CANVAS_SIZE):
    px = int((float(x) + 1.0) * 0.5 * size)
    py = int((1.0 - float(y)) * 0.5 * size)
    return max(0, min(size - 1, px)), max(0, min(size - 1, py))


def _canvas_to_display(px, py, size=CANVAS_SIZE):
    x = (float(px) / size) * 2.0 - 1.0
    y = 1.0 - (float(py) / size) * 2.0
    return x, y


def create_default_umpires():
    return [umpire_from_name(name) for name in UMPIRE_POSITIONS_RH]


def build_field_setup(
    preset_name="Balanced",
    handedness="Right-handed",
    fielders=None,
    umpires=None,
):
    fielders = fielders or create_preset_fielders(preset_name)
    umpires = umpires or create_default_umpires()
    return {
        "preset": preset_name,
        "batter_handedness": _normalize_handedness(handedness),
        "bowler_arm": "Right-arm bowler",
        "camera_view": "Behind bowler",
        "fielders": [ensure_fielder_polar(dict(item)) for item in fielders],
        "umpires": [ensure_umpire_polar(dict(item)) for item in umpires],
        "coordinate_system_version": FIELD_COORDINATE_VERSION,
    }


def draw_cricket_field_figure(
    fielders=None,
    umpires=None,
    handedness="Right-handed",
    shot_angle=None,
    selected_zone="Unknown",
    show_labels=True,
    compact=False,
    visual_rotation_deg=DEFAULT_VISUAL_ROTATION,
):
    fielders = fielders or []
    umpires = umpires or create_default_umpires()
    handedness = _normalize_handedness(handedness)
    sides = get_side_labels(handedness)

    fig_size = 4.8 if compact else 7
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    fig.patch.set_facecolor("#07111f")
    ax.set_facecolor("#0c5f3b")
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)

    boundary = Ellipse((0, 0), 2.0, 1.82, facecolor="#137a46", edgecolor="#d8f3dc", linewidth=2.0)
    inner_ring = Ellipse((0, 0), 1.24, 1.12, facecolor="none", edgecolor="#bae6fd", linewidth=1.1, linestyle="--")
    ax.add_patch(boundary)
    ax.add_patch(inner_ring)

    zone_specs = [
        ("Long On", -38, -8),
        ("Mid On", -72, -38),
        ("Mid Wicket", -122, -72),
        ("Square Leg", -158, -122),
        ("Fine Leg", -180, -158),
        ("Straight", -8, 8),
        ("Long Off", 8, 38),
        ("Mid Off", 38, 72),
        ("Cover", 72, 100),
        ("Point", 100, 122),
        ("Third Man", 122, 158),
    ]

    for zone_name, start_angle, end_angle in zone_specs:
        active = selected_zone in {zone_name, "Third Man", "Gully"}
        color = "#34d399" if active else "#94a3b8"
        wedge_start = apply_wedge_angles(start_angle, end_angle, handedness, visual_rotation_deg)
        wedge = Wedge(
            (0, 0),
            1.0,
            wedge_start[0] - 90,
            wedge_start[1] - 90,
            width=0.98,
            alpha=0.16 if active else 0.07,
            facecolor=color,
            edgecolor="#d8f3dc",
            linewidth=0.6,
        )
        ax.add_patch(wedge)
        if show_labels:
            mid = (start_angle + end_angle) / 2
            label_x, label_y = polar_to_screen(mid, 0.72, handedness, visual_rotation_deg)
            ax.text(label_x, label_y, zone_name, ha="center", va="center", fontsize=7 if compact else 8, color="#f8fafc")

    for angle in range(-180, 181, 45):
        line_x, line_y = polar_to_screen(angle, 0.96, handedness, visual_rotation_deg)
        ax.plot([0, line_x], [0, line_y], color="#e2e8f0", alpha=0.15, linewidth=0.7)

    pitch = Rectangle((-0.055, -0.24), 0.11, 0.48, facecolor="#d6b47a", edgecolor="#fef3c7", linewidth=1.4)
    ax.add_patch(pitch)
    ax.scatter([0], [0.11], s=36 if compact else 42, color="#f8fafc", zorder=5)
    if show_labels:
        ax.annotate("Batter", (0, 0.11), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=7, color="#f8fafc")
    ax.scatter([0], [-0.08], s=28 if compact else 32, color="#93c5fd", zorder=5)
    if show_labels:
        ax.annotate("Bowler end", (0, -0.08), xytext=(0, -12), textcoords="offset points", ha="center", fontsize=7, color="#dbeafe")

    ax.text(-0.62, -0.96, sides["left"], ha="center", va="center", fontsize=8, color="#bae6fd")
    ax.text(0.62, -0.96, sides["right"], ha="center", va="center", fontsize=8, color="#fecaca")

    if shot_angle is not None:
        display_shot = mirror_angle_for_handedness(shot_angle, handedness)
        arrow_x, arrow_y = polar_to_screen(display_shot, 0.88, handedness, visual_rotation_deg)
        ax.arrow(0, 0, arrow_x, arrow_y, width=0.012, head_width=0.06, head_length=0.08, length_includes_head=True, color="#fbbf24", zorder=8)

    for fielder in fielders:
        ensure_fielder_polar(fielder)
        marker_x, marker_y = fielder_display_xy(fielder, handedness, visual_rotation_deg)
        name = fielder.get("name", "Fielder")
        ax.scatter([marker_x], [marker_y], s=58 if compact else 72, color="#fde047", edgecolor="#111827", zorder=9)
        if show_labels:
            ax.annotate(
                name,
                (marker_x, marker_y),
                xytext=(0, -10),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=6.5 if compact else 7.5,
                color="#fefce8",
            )

    for umpire in umpires:
        ensure_umpire_polar(umpire)
        marker_x, marker_y = umpire_display_xy(umpire, handedness, visual_rotation_deg)
        name = umpire.get("name", "Umpire")
        ax.scatter(
            [marker_x],
            [marker_y],
            s=68 if compact else 82,
            color="#f8fafc",
            edgecolor="#2563eb",
            linewidth=2,
            zorder=10,
        )
        if show_labels:
            offset = (0, 12) if "Square Leg" in name else (10, 0)
            ax.annotate(
                name.replace(" Umpire", ""),
                (marker_x, marker_y),
                xytext=offset,
                textcoords="offset points",
                ha="center" if "Square Leg" in name else "left",
                va="center",
                fontsize=6.5 if compact else 7.5,
                color="#dbeafe",
            )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def apply_wedge_angles(start_angle, end_angle, handedness, visual_rotation_deg):
    from Backends.src.analysis.field_geometry import apply_visual_rotation

    start = apply_visual_rotation(mirror_angle_for_handedness(start_angle, handedness), visual_rotation_deg)
    end = apply_visual_rotation(mirror_angle_for_handedness(end_angle, handedness), visual_rotation_deg)
    return start, end


def field_figure_to_png(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, facecolor=fig.patch.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def _render_canvas_editor(field_setup, handedness, show_labels, key):
    try:
        from streamlit_drawable_canvas import st_canvas
    except ImportError:
        return None, "streamlit-drawable-canvas is not installed."

    fielders = field_setup["fielders"]
    background_fig = draw_cricket_field_figure(
        fielders=[],
        umpires=field_setup.get("umpires"),
        handedness=handedness,
        show_labels=show_labels,
        compact=True,
    )
    background_image = field_figure_to_png(background_fig)

    initial_drawing = {"version": "4.4.0", "objects": []}
    radius_px = 14
    for index, fielder in enumerate(fielders):
        x, y = fielder_display_xy(fielder, handedness)
        px, py = _display_to_canvas(x, y)
        initial_drawing["objects"].append(
            {
                "type": "circle",
                "left": px - radius_px,
                "top": py - radius_px,
                "radius": radius_px,
                "fill": "#fde047",
                "stroke": "#111827",
                "strokeWidth": 2,
                "name": str(index),
            }
        )

    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=0,
        background_image=background_image,
        update_streamlit=True,
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="transform",
        initial_drawing=initial_drawing,
        display_toolbar=False,
        key=f"{key}_canvas",
    )

    if canvas_result.json_data is None:
        return field_setup, None

    objects = canvas_result.json_data.get("objects", [])
    updated_fielders = [dict(item) for item in fielders]
    for obj in objects:
        name = obj.get("name")
        if name is None:
            continue
        try:
            index = int(name)
        except ValueError:
            continue
        if index >= len(updated_fielders):
            continue
        left = float(obj.get("left", 0)) + float(obj.get("radius", radius_px))
        top = float(obj.get("top", 0)) + float(obj.get("radius", radius_px))
        display_x, display_y = _canvas_to_display(left, top)
        angle, radius = screen_to_polar(display_x, display_y, handedness)
        updated_fielders[index]["angle"] = round(angle, 2)
        updated_fielders[index]["radius"] = round(min(max(radius, 0.12), 0.98), 3)
        ensure_fielder_polar(updated_fielders[index])

    field_setup = dict(field_setup)
    field_setup["fielders"] = updated_fielders
    field_setup["preset"] = "Custom"
    return field_setup, None


def _render_slider_editor(field_setup, handedness, key):
    fielders = [ensure_fielder_polar(dict(item)) for item in field_setup["fielders"]]
    names = [item.get("name", f"Fielder {index + 1}") for index, item in enumerate(fielders)]
    selected_index = st.selectbox(
        "Select fielder to move",
        list(range(len(names))),
        format_func=lambda index: names[index],
        key=f"{key}_fielder_select",
    )
    selected = fielders[selected_index]
    col1, col2 = st.columns(2)
    with col1:
        angle = st.slider(
            "Angle (off + / leg -)",
            min_value=-180,
            max_value=180,
            value=int(round(selected.get("angle", 0))),
            key=f"{key}_angle_{selected_index}",
        )
    with col2:
        radius = st.slider(
            "Radius",
            min_value=0.12,
            max_value=0.98,
            value=float(selected.get("radius", 0.6)),
            step=0.01,
            key=f"{key}_radius_{selected_index}",
        )
    fielders[selected_index]["angle"] = float(angle)
    fielders[selected_index]["radius"] = float(radius)
    ensure_fielder_polar(fielders[selected_index])
    field_setup = dict(field_setup)
    field_setup["fielders"] = fielders
    field_setup["preset"] = "Custom"
    return field_setup


def render_interactive_field_map(
    field_setup=None,
    handedness="Right-handed",
    preset_name="Balanced",
    editable=True,
    compact=False,
    show_labels=True,
    key="field_map",
    shot_angle=None,
    selected_zone="Unknown",
):
    handedness = _normalize_handedness(handedness or (field_setup or {}).get("batter_handedness"))
    if field_setup is None:
        field_setup = build_field_setup(preset_name, handedness)
    else:
        field_setup = dict(field_setup)
        field_setup["batter_handedness"] = handedness
        field_setup["fielders"] = [ensure_fielder_polar(dict(item)) for item in field_setup.get("fielders", [])]
        field_setup["umpires"] = [ensure_umpire_polar(dict(item)) for item in field_setup.get("umpires", create_default_umpires())]

    state_key = f"{key}_setup"
    if state_key not in st.session_state:
        st.session_state[state_key] = field_setup
    else:
        st.session_state[state_key].update(
            {
                "batter_handedness": handedness,
                "fielders": field_setup["fielders"],
                "umpires": field_setup["umpires"],
                "preset": field_setup.get("preset", preset_name),
            }
        )

    current = st.session_state[state_key]
    fig = draw_cricket_field_figure(
        fielders=current["fielders"],
        umpires=current.get("umpires"),
        handedness=handedness,
        shot_angle=shot_angle,
        selected_zone=selected_zone,
        show_labels=show_labels,
        compact=compact,
    )
    st.pyplot(fig)

    if editable:
        edit_mode = st.toggle("Edit field positions", value=False, key=f"{key}_edit_toggle")
        if edit_mode:
            canvas_setup, canvas_error = _render_canvas_editor(current, handedness, show_labels, key)
            if canvas_error:
                st.caption(canvas_error)
                st.caption("Using slider edit mode instead.")
                current = _render_slider_editor(current, handedness, key)
            else:
                current = canvas_setup or current
            st.session_state[state_key] = current

    current["coordinate_system_version"] = FIELD_COORDINATE_VERSION
    return current


def render_field_setup_card(key_prefix="field", compact=True, default_preset="Balanced"):
    """Compact field setup block for Analyze / Live Session pages."""
    from Backends.src.analysis.field_zones import get_active_field_setup, set_active_field_setup
    from Backends.src.ui.theme import render_section_title

    render_section_title("Field Setup", "Used for wagon-wheel direction and nearest fielder context.")

    active = get_active_field_setup()
    preset_options = PRESET_NAMES + (["Custom"] if active.get("preset") not in PRESET_NAMES else [])
    if active.get("preset") not in preset_options:
        preset_options.append(active.get("preset", "Custom"))

    col1, col2 = st.columns(2)
    with col1:
        handedness = st.selectbox(
            "Batter handedness",
            ["Right-handed", "Left-handed"],
            index=0 if not str(active.get("batter_handedness", "")).lower().startswith("left") else 1,
            key=f"{key_prefix}_handedness",
        )
    with col2:
        selected_preset = st.selectbox(
            "Field preset",
            preset_options,
            index=preset_options.index(active.get("preset", default_preset))
            if active.get("preset", default_preset) in preset_options
            else 0,
            key=f"{key_prefix}_preset",
        )

    if st.button("Reset to preset", key=f"{key_prefix}_reset_preset"):
        active = build_field_setup(selected_preset, handedness)
        st.session_state["current_field_setup"] = active
        st.session_state["current_field_preset"] = selected_preset
        st.session_state["current_batter_handedness"] = handedness
        st.rerun()

    if selected_preset != "Custom" and selected_preset != active.get("preset"):
        active = build_field_setup(selected_preset, handedness)

    prev_preset_key = f"{key_prefix}_last_preset"
    if (
        selected_preset != "Custom"
        and st.session_state.get(prev_preset_key) != selected_preset
    ):
        active = build_field_setup(selected_preset, handedness)
        st.session_state[prev_preset_key] = selected_preset
    else:
        active["batter_handedness"] = handedness
        active["preset"] = selected_preset

    show_labels = st.checkbox("Show field labels", value=True, key=f"{key_prefix}_show_labels")

    with st.expander("Edit Field", expanded=False):
        active = render_interactive_field_map(
            field_setup=active,
            handedness=handedness,
            preset_name=selected_preset,
            editable=True,
            compact=compact,
            show_labels=show_labels,
            key=f"{key_prefix}_interactive_map",
        )

    if st.button("Save Field Setup", type="primary", use_container_width=True, key=f"{key_prefix}_save_field"):
        active["preset"] = selected_preset if selected_preset != "Custom" else active.get("preset", "Custom")
        active["batter_handedness"] = handedness
        set_active_field_setup(active)
        st.session_state["current_field_preset"] = active["preset"]
        st.session_state["current_batter_handedness"] = handedness
        st.success("Field setup saved for this session.")

    st.session_state["current_field_setup"] = active
    return active
