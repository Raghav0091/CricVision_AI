"""Interactive cricket field map with angle/radius editing."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, Wedge
import streamlit as st

from Backends.src.analysis.field_geometry import (
    DEFAULT_VISUAL_ROTATION,
    HANDEDNESS_LABELS,
    HANDEDNESS_LEFT,
    HANDEDNESS_RIGHT,
    PITCH_AXIS_DEGREES,
    UMPIRE_POSITIONS_RH,
    angle_to_field_zone,
    bowler_end_xy,
    display_angle_to_screen,
    ensure_fielder_polar,
    ensure_umpire_polar,
    fielder_display_xy,
    mirror_angle_for_handedness,
    normalize_handedness,
    pitch_polygon_corners,
    polar_to_screen,
    screen_to_polar,
    striker_crease_xy,
    umpire_display_xy,
    umpire_from_name,
)
from Backends.src.data.field_presets import PRESET_NAMES, create_preset_fielders

CANVAS_SIZE = 520
FIELD_COORDINATE_VERSION = 3
FIELDER_RADIUS_PX = 9

ZONE_SPECS = [
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


def _normalize_handedness(handedness):
    return normalize_handedness(handedness)


def get_field_map_handedness(selected_label_or_value):
    """
    Temporary handedness mapping workaround: field renderer currently mirrors handedness,
    so UI value is inverted before rendering.
    """
    normalized = _normalize_handedness(selected_label_or_value)
    if normalized == HANDEDNESS_RIGHT:
        return HANDEDNESS_LEFT
    if normalized == HANDEDNESS_LEFT:
        return HANDEDNESS_RIGHT
    return HANDEDNESS_RIGHT


def _display_to_canvas(x, y, size=CANVAS_SIZE):
    px = int((float(x) + 1.0) * 0.5 * size)
    py = int((float(y) + 1.0) * 0.5 * size)
    return max(0, min(size - 1, px)), max(0, min(size - 1, py))


def _canvas_to_display(px, py, size=CANVAS_SIZE):
    x = (float(px) / size) * 2.0 - 1.0
    y = (float(py) / size) * 2.0 - 1.0
    return x, y


def _fabric_static(obj, kind="field_static"):
    obj.setdefault("selectable", False)
    obj.setdefault("evented", False)
    data = dict(obj.get("data") or {})
    data.setdefault("kind", kind)
    obj["data"] = data
    return obj


def _fabric_line_display(x1, y1, x2, y2, stroke, width=1, dash=None, kind="field_static"):
    px1, py1 = _display_to_canvas(x1, y1)
    px2, py2 = _display_to_canvas(x2, y2)
    left = min(px1, px2)
    top = min(py1, py2)
    obj = {
        "type": "line",
        "left": left,
        "top": top,
        "x1": px1 - left,
        "y1": py1 - top,
        "x2": px2 - left,
        "y2": py2 - top,
        "stroke": stroke,
        "strokeWidth": width,
    }
    if dash:
        obj["strokeDashArray"] = dash
    return _fabric_static(obj, kind=kind)


def _fabric_polygon_display(corners, fill, stroke, stroke_width=2):
    canvas_pts = [_display_to_canvas(x, y) for x, y in corners]
    left = min(point[0] for point in canvas_pts)
    top = min(point[1] for point in canvas_pts)
    rel_pts = [{"x": point[0] - left, "y": point[1] - top} for point in canvas_pts]
    return _fabric_static(
        {
            "type": "polygon",
            "left": left,
            "top": top,
            "points": rel_pts,
            "fill": fill,
            "stroke": stroke,
            "strokeWidth": stroke_width,
        }
    )


def _fabric_text_display(x, y, text, fill="#f8fafc", font_size=11):
    px, py = _display_to_canvas(x, y)
    return _fabric_static(
        {
            "type": "text",
            "left": px,
            "top": py,
            "text": text,
            "fontSize": font_size,
            "fill": fill,
        }
    )


def _fabric_circle_display(x, y, radius_px, fill, stroke, stroke_width=2, kind="field_static", data=None):
    px, py = _display_to_canvas(x, y)
    obj = {
        "type": "circle",
        "left": px - radius_px,
        "top": py - radius_px,
        "radius": radius_px,
        "fill": fill,
        "stroke": stroke,
        "strokeWidth": stroke_width,
    }
    if kind == "fielder":
        obj["selectable"] = True
        obj["evented"] = True
        obj["hasControls"] = True
        obj["hasBorders"] = True
        obj["data"] = data or {"kind": "fielder"}
    else:
        obj = _fabric_static(obj, kind=kind)
        if data:
            obj["data"].update(data)
    return obj


def build_canvas_field_drawing(
    fielders,
    umpires,
    handedness,
    show_labels=True,
    visual_rotation_deg=DEFAULT_VISUAL_ROTATION,
):
    """Build Fabric.js objects for the full cricket field (static) plus draggable fielders."""
    handedness = _normalize_handedness(handedness)
    umpires = umpires or create_default_umpires()
    objects = []

    center_x, center_y = _display_to_canvas(0, 0)
    boundary_radius = int(0.96 * CANVAS_SIZE / 2)
    inner_radius = int(0.62 * CANVAS_SIZE / 2)

    objects.append(
        _fabric_static(
            {
                "type": "circle",
                "left": center_x - boundary_radius,
                "top": center_y - boundary_radius,
                "radius": boundary_radius,
                "fill": "#137a46",
                "stroke": "#d8f3dc",
                "strokeWidth": 3,
            }
        )
    )
    objects.append(
        _fabric_static(
            {
                "type": "circle",
                "left": center_x - inner_radius,
                "top": center_y - inner_radius,
                "radius": inner_radius,
                "fill": "rgba(0,0,0,0)",
                "stroke": "#bae6fd",
                "strokeWidth": 1,
                "strokeDashArray": [6, 6],
            }
        )
    )

    for angle in range(-180, 181, 45):
        line_x, line_y = polar_to_screen(angle, 0.96, handedness, visual_rotation_deg)
        objects.append(_fabric_line_display(0, 0, line_x, line_y, "#e2e8f0", width=1))

    straight_x, straight_y = polar_to_screen(0, 0.95, handedness, visual_rotation_deg)
    objects.append(_fabric_line_display(0, 0, straight_x, straight_y, "#fef3c7", width=2, kind="straight_axis"))

    objects.append(
        _fabric_polygon_display(
            pitch_polygon_corners(handedness, visual_rotation_deg),
            fill="#d6b47a",
            stroke="#fef3c7",
            stroke_width=2,
        )
    )

    striker_x, striker_y = striker_crease_xy(handedness, visual_rotation_deg)
    bowler_x, bowler_y = bowler_end_xy(handedness, visual_rotation_deg)
    perp_x, perp_y = polar_to_screen(90, 0.09, handedness, visual_rotation_deg)
    objects.append(
        _fabric_line_display(
            striker_x - perp_x,
            striker_y - perp_y,
            striker_x + perp_x,
            striker_y + perp_y,
            "#fef3c7",
            width=2,
        )
    )
    objects.append(
        _fabric_line_display(
            bowler_x - perp_x,
            bowler_y - perp_y,
            bowler_x + perp_x,
            bowler_y + perp_y,
            "#fef3c7",
            width=2,
        )
    )

    objects.append(_fabric_circle_display(striker_x, striker_y, 6, "#f8fafc", "#f8fafc", kind="batter"))
    objects.append(_fabric_circle_display(bowler_x, bowler_y, 5, "#93c5fd", "#93c5fd", kind="bowler"))

    for umpire in umpires:
        ensure_umpire_polar(umpire)
        ux, uy = umpire_display_xy(umpire, handedness, visual_rotation_deg)
        objects.append(
            _fabric_circle_display(
                ux,
                uy,
                9,
                "#f8fafc",
                "#2563eb",
                stroke_width=2,
                kind="umpire",
                data={"name": umpire.get("name", "Umpire")},
            )
        )

    if show_labels:
        for zone_name, start_angle, end_angle in ZONE_SPECS:
            mid = (start_angle + end_angle) / 2
            label_x, label_y = polar_to_screen(mid, 0.72, handedness, visual_rotation_deg)
            objects.append(_fabric_text_display(label_x, label_y, zone_name, font_size=10))

        leg_x, leg_y = polar_to_screen(-90, 0.92, handedness, visual_rotation_deg)
        off_x, off_y = polar_to_screen(90, 0.92, handedness, visual_rotation_deg)
        objects.append(_fabric_text_display(leg_x, leg_y, "Leg side", fill="#bae6fd", font_size=11))
        objects.append(_fabric_text_display(off_x, off_y, "Off side", fill="#fecaca", font_size=11))
        objects.append(_fabric_text_display(striker_x, striker_y + 0.06, "Batter", font_size=10))
        objects.append(_fabric_text_display(bowler_x, bowler_y - 0.06, "Bowler end", fill="#dbeafe", font_size=10))

        for umpire in umpires:
            ensure_umpire_polar(umpire)
            ux, uy = umpire_display_xy(umpire, handedness, visual_rotation_deg)
            short_name = umpire.get("name", "Umpire").replace(" Umpire", "")
            objects.append(_fabric_text_display(ux + 0.04, uy, short_name, fill="#dbeafe", font_size=9))

    for index, fielder in enumerate(fielders):
        ensure_fielder_polar(fielder)
        x, y = fielder_display_xy(fielder, handedness, visual_rotation_deg)
        name = fielder.get("name", f"F{index + 1}")
        objects.append(
            _fabric_circle_display(
                x,
                y,
                FIELDER_RADIUS_PX,
                "#F7C948",
                "#111827",
                stroke_width=2,
                kind="fielder",
                data={"kind": "fielder", "index": index, "name": name},
            )
        )
        if show_labels:
            px, py = _display_to_canvas(x, y)
            objects.append(
                _fabric_static(
                    {
                        "type": "text",
                        "left": px - 12,
                        "top": py - FIELDER_RADIUS_PX - 14,
                        "text": name[:10],
                        "fontSize": 10,
                        "fill": "#fefce8",
                    }
                )
            )

    return {"version": "4.4.0", "objects": objects}


def create_default_umpires():
    return [umpire_from_name(name) for name in UMPIRE_POSITIONS_RH]


def build_field_setup(
    preset_name="Balanced",
    handedness=HANDEDNESS_RIGHT,
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


def apply_wedge_angles(start_angle, end_angle, handedness, visual_rotation_deg):
    from Backends.src.analysis.field_geometry import apply_visual_rotation

    start = apply_visual_rotation(mirror_angle_for_handedness(start_angle, handedness), visual_rotation_deg)
    end = apply_visual_rotation(mirror_angle_for_handedness(end_angle, handedness), visual_rotation_deg)
    if start > end:
        start, end = end, start
    return start, end


def draw_field_map(
    shot_angle=None,
    selected_zone="Unknown",
    fielders=None,
    batter_handedness=HANDEDNESS_RIGHT,
    umpires=None,
    show_labels=True,
    compact=False,
):
    """Matplotlib field preview used by analysis result views."""
    return draw_cricket_field_figure(
        fielders=fielders,
        umpires=umpires,
        handedness=batter_handedness,
        shot_angle=shot_angle,
        selected_zone=selected_zone,
        show_labels=show_labels,
        compact=compact,
    )


def draw_cricket_field_figure(
    fielders=None,
    umpires=None,
    handedness=HANDEDNESS_RIGHT,
    shot_angle=None,
    selected_zone="Unknown",
    show_labels=True,
    compact=False,
    visual_rotation_deg=DEFAULT_VISUAL_ROTATION,
):
    handedness = get_field_map_handedness(handedness)
    fielders = fielders or []
    umpires = umpires or create_default_umpires()
    handedness = _normalize_handedness(handedness)

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

    for zone_name, start_angle, end_angle in ZONE_SPECS:
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

    straight_x, straight_y = polar_to_screen(0, 0.95, handedness, visual_rotation_deg)
    ax.plot([0, straight_x], [0, straight_y], color="#fef3c7", alpha=0.85, linewidth=1.8, zorder=4)

    pitch = Polygon(
        pitch_polygon_corners(handedness, visual_rotation_deg),
        facecolor="#d6b47a",
        edgecolor="#fef3c7",
        linewidth=1.4,
    )
    ax.add_patch(pitch)

    striker_x, striker_y = striker_crease_xy(handedness, visual_rotation_deg)
    bowler_x, bowler_y = bowler_end_xy(handedness, visual_rotation_deg)
    ax.scatter([striker_x], [striker_y], s=36 if compact else 42, color="#f8fafc", zorder=5)
    if show_labels:
        ax.annotate("Batter", (striker_x, striker_y), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=7, color="#f8fafc")
    ax.scatter([bowler_x], [bowler_y], s=28 if compact else 32, color="#93c5fd", zorder=5)
    if show_labels:
        ax.annotate("Bowler end", (bowler_x, bowler_y), xytext=(0, -12), textcoords="offset points", ha="center", fontsize=7, color="#dbeafe")

    ax.text(
        *polar_to_screen(-90, 0.92, handedness, visual_rotation_deg),
        "Leg side",
        ha="center",
        va="center",
        fontsize=8,
        color="#bae6fd",
    )
    ax.text(
        *polar_to_screen(90, 0.92, handedness, visual_rotation_deg),
        "Off side",
        ha="center",
        va="center",
        fontsize=8,
        color="#fecaca",
    )

    if shot_angle is not None:
        arrow_x, arrow_y = display_angle_to_screen(shot_angle, 0.88, visual_rotation_deg)
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


def parse_canvas_fielders(canvas_result, fielders, handedness, visual_rotation_deg=DEFAULT_VISUAL_ROTATION):
    if canvas_result is None or canvas_result.json_data is None:
        return fielders

    objects = canvas_result.json_data.get("objects", [])
    updated_fielders = [dict(item) for item in fielders]

    for obj in objects:
        if obj.get("type") != "circle":
            continue
        data = obj.get("data") or {}
        if data.get("kind") != "fielder":
            continue
        index = data.get("index")
        if index is None:
            continue
        if index >= len(updated_fielders):
            continue
        left = float(obj.get("left", 0)) + float(obj.get("radius", FIELDER_RADIUS_PX))
        top = float(obj.get("top", 0)) + float(obj.get("radius", FIELDER_RADIUS_PX))
        display_x, display_y = _canvas_to_display(left, top)
        angle, radius = screen_to_polar(display_x, display_y, handedness, visual_rotation_deg)
        updated_fielders[index]["angle"] = round(angle, 2)
        updated_fielders[index]["radius"] = round(min(max(radius, 0.12), 0.98), 3)
        ensure_fielder_polar(updated_fielders[index])

    return updated_fielders


def _render_canvas_editor(field_setup, handedness, show_labels, key):
    try:
        from streamlit_drawable_canvas import st_canvas
    except ImportError:
        return None, "streamlit-drawable-canvas is not installed."

    render_handedness = get_field_map_handedness(handedness)
    fielders = field_setup["fielders"]
    initial_drawing = build_canvas_field_drawing(
        fielders=fielders,
        umpires=field_setup.get("umpires"),
        handedness=render_handedness,
        show_labels=show_labels,
    )

    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=0,
        background_color="#0B3D2E",
        update_streamlit=True,
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="transform",
        initial_drawing=initial_drawing,
        display_toolbar=False,
        key=f"{key}_canvas",
    )

    updated_fielders = parse_canvas_fielders(canvas_result, fielders, render_handedness)
    field_setup = dict(field_setup)
    field_setup["fielders"] = updated_fielders
    if updated_fielders != fielders:
        field_setup["preset"] = "Custom"
    st.session_state["current_field_setup"] = field_setup
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
    handedness=HANDEDNESS_RIGHT,
    preset_name="Balanced",
    editable=True,
    compact=False,
    show_labels=True,
    key="field_map",
    shot_angle=None,
    selected_zone="Unknown",
    use_canvas=True,
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

    if editable and use_canvas:
        st.caption("Drag yellow circles to reposition fielders.")
        canvas_setup, canvas_error = _render_canvas_editor(current, handedness, show_labels, key)
        if canvas_error:
            st.warning(canvas_error)
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
        else:
            current = canvas_setup or current
    else:
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

    st.session_state[state_key] = current
    current["coordinate_system_version"] = FIELD_COORDINATE_VERSION
    return current


def render_field_setup_card(key_prefix="field", compact=True, default_preset="Balanced"):
    """Compact field setup block for Analyze / Live Session pages."""
    from Backends.src.analysis.field_zones import get_active_field_setup, set_active_field_setup
    from Backends.src.ui.theme import render_section_title

    render_section_title("Field Setup", "Drag fielders on the map. Used for wagon-wheel direction and nearest fielder context.")

    active = get_active_field_setup()
    preset_options = PRESET_NAMES + (["Custom"] if active.get("preset") not in PRESET_NAMES else [])
    if active.get("preset") not in preset_options:
        preset_options.append(active.get("preset", "Custom"))

    col1, col2 = st.columns(2)
    with col1:
        handedness = st.selectbox(
            "Batter handedness",
            [HANDEDNESS_RIGHT, HANDEDNESS_LEFT],
            index=1 if _normalize_handedness(active.get("batter_handedness")) == HANDEDNESS_LEFT else 0,
            format_func=lambda value: HANDEDNESS_LABELS[value],
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

    active = render_interactive_field_map(
        field_setup=active,
        handedness=handedness,
        preset_name=selected_preset,
        editable=True,
        compact=compact,
        show_labels=show_labels,
        key=f"{key_prefix}_interactive_map",
        use_canvas=True,
    )

    with st.expander("Developer / Advanced", expanded=False):
        st.caption(f"Pitch axis: {PITCH_AXIS_DEGREES}° | coordinate version {FIELD_COORDINATE_VERSION}")
        active = _render_slider_editor(active, handedness, f"{key_prefix}_advanced")
        if st.checkbox("Show raw fielder coordinates", value=False, key=f"{key_prefix}_show_raw"):
            st.json(
                [
                    {
                        "name": item.get("name"),
                        "angle": item.get("angle"),
                        "radius": item.get("radius"),
                        "zone": angle_to_field_zone(item.get("angle"), handedness),
                    }
                    for item in active.get("fielders", [])
                ]
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


# Backward-compatible aliases
build_canvas_objects = build_canvas_field_drawing
build_fabric_field_drawing = build_canvas_field_drawing
