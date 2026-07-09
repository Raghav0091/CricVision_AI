"""Single-camera replay calibration geometry helpers.

Pure geometry only: no Streamlit, no YOLO, no model loading.
"""

from __future__ import annotations

import math


STUMP_KEYS = (
    "near_left",
    "near_middle",
    "near_right",
    "far_left",
    "far_middle",
    "far_right",
)


def _normalize_frame_size(frame_size):
    if frame_size is None:
        return None
    if isinstance(frame_size, dict):
        width = frame_size.get("width")
        height = frame_size.get("height")
    elif isinstance(frame_size, (list, tuple)) and len(frame_size) >= 2:
        width, height = frame_size[:2]
    else:
        return None
    try:
        width = int(width)
        height = int(height)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return (width, height)


def _coerce_point(value):
    if isinstance(value, dict):
        x = value.get("x")
        y = value.get("y")
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        x, y = value[:2]
    else:
        return None
    try:
        return {"x": float(x), "y": float(y)}
    except (TypeError, ValueError):
        return None


def _point_in_frame(point, frame_size):
    if frame_size is None:
        return True
    width, height = frame_size
    return 0 <= point["x"] <= width and 0 <= point["y"] <= height


def _distance(a, b):
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def _point_dict(point):
    return {"x": float(point["x"]), "y": float(point["y"])}


def normalize_stump_calibration(calibration, frame_size=None):
    """Normalize manual near/far stump base points.

    At minimum, near_middle and far_middle are required for a usable stump line.
    """
    notes = []
    normalized_frame_size = _normalize_frame_size(frame_size)
    if frame_size is not None and normalized_frame_size is None:
        notes.append("Invalid frame_size ignored.")

    points = {}
    if not isinstance(calibration, dict):
        calibration = {}
        notes.append("Calibration input is missing or invalid.")

    for key in STUMP_KEYS:
        raw_point = calibration.get(key)
        if raw_point in (None, "", {}):
            continue
        point = _coerce_point(raw_point)
        if point is None:
            notes.append(f"{key} is invalid and was ignored.")
            continue
        if not _point_in_frame(point, normalized_frame_size):
            notes.append(f"{key} is outside the frame and was ignored.")
            continue
        points[key] = point

    if "near_middle" not in points or "far_middle" not in points:
        notes.append("near_middle and far_middle are required for replay calibration.")

    return {
        "available": "near_middle" in points and "far_middle" in points,
        "points": points,
        "frame_size": normalized_frame_size,
        "notes": notes,
    }


def _wicket_line(points, left_key, right_key):
    left = points.get(left_key)
    right = points.get(right_key)
    if not left or not right:
        return None
    return {"start": _point_dict(left), "end": _point_dict(right)}


def _corridor_polygon(near_middle, far_middle, width_px):
    dx = far_middle["x"] - near_middle["x"]
    dy = far_middle["y"] - near_middle["y"]
    length = math.hypot(dx, dy)
    if length < 1 or width_px <= 0:
        return None
    nx = -dy / length
    ny = dx / length
    half_width = width_px / 2.0
    return [
        {"x": near_middle["x"] + nx * half_width, "y": near_middle["y"] + ny * half_width},
        {"x": far_middle["x"] + nx * half_width, "y": far_middle["y"] + ny * half_width},
        {"x": far_middle["x"] - nx * half_width, "y": far_middle["y"] - ny * half_width},
        {"x": near_middle["x"] - nx * half_width, "y": near_middle["y"] - ny * half_width},
    ]


def build_pitch_geometry(calibration, frame_size=None):
    """Build estimated pixel-space pitch geometry from stump calibration."""
    normalized = normalize_stump_calibration(calibration, frame_size=frame_size)
    notes = list(normalized.get("notes") or [])
    points = normalized.get("points") or {}
    if not normalized.get("available"):
        return {"available": False, "notes": notes}

    near_middle = points["near_middle"]
    far_middle = points["far_middle"]
    near_wicket_line = _wicket_line(points, "near_left", "near_right")
    far_wicket_line = _wicket_line(points, "far_left", "far_right")

    stump_width_px_near = None
    if points.get("near_left") and points.get("near_right"):
        stump_width_px_near = _distance(points["near_left"], points["near_right"])
    stump_width_px_far = None
    if points.get("far_left") and points.get("far_right"):
        stump_width_px_far = _distance(points["far_left"], points["far_right"])

    widths = [value for value in (stump_width_px_near, stump_width_px_far) if value]
    corridor = None
    if widths:
        # ponytail: v1 corridor is an estimated pixel guide, not a measured pitch model.
        corridor = _corridor_polygon(near_middle, far_middle, max(widths) * 4.0)
    else:
        notes.append("Pitch corridor unavailable without left/right stump points.")

    axis = {
        "start": _point_dict(near_middle),
        "end": _point_dict(far_middle),
    }
    return {
        "available": True,
        "stump_line": axis,
        "near_wicket_line": near_wicket_line,
        "far_wicket_line": far_wicket_line,
        "pitch_axis": axis,
        "pitch_corridor": corridor,
        "stump_width_px_near": stump_width_px_near,
        "stump_width_px_far": stump_width_px_far,
        "notes": notes,
    }


def _line_projection(point, start, end):
    px, py = float(point["x"]), float(point["y"])
    ax, ay = float(start["x"]), float(start["y"])
    bx, by = float(end["x"]), float(end["y"])
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 0:
        return None
    progress = ((px - ax) * dx + (py - ay) * dy) / denom
    projected_x = ax + progress * dx
    projected_y = ay + progress * dy
    offset = math.hypot(px - projected_x, py - projected_y)
    return progress, offset


def point_near_stump_line(point, pitch_geometry, max_distance_px=80):
    notes = []
    point = _coerce_point(point)
    geometry = pitch_geometry or {}
    line = geometry.get("stump_line") or geometry.get("pitch_axis") or {}
    if point is None or not geometry.get("available") or not line.get("start") or not line.get("end"):
        return {"near": False, "distance_px": None, "score_bonus": 0.0, "notes": ["Stump line unavailable."]}

    projection = _line_projection(point, line["start"], line["end"])
    if projection is None:
        return {"near": False, "distance_px": None, "score_bonus": 0.0, "notes": ["Invalid stump line."]}

    progress, distance_px = projection
    try:
        max_distance_px = float(max_distance_px)
    except (TypeError, ValueError):
        max_distance_px = 80.0
    near = distance_px <= max_distance_px
    if progress < 0 or progress > 1:
        notes.append("Point projects outside the calibrated stump-to-stump range.")
    score_bonus = max(0.0, 1.0 - (distance_px / max(max_distance_px, 1.0))) if near else 0.0
    return {
        "near": near,
        "distance_px": float(distance_px),
        "score_bonus": round(float(score_bonus), 4),
        "notes": notes,
    }


def project_point_to_pitch_axis(point, pitch_geometry):
    point = _coerce_point(point)
    geometry = pitch_geometry or {}
    line = geometry.get("pitch_axis") or geometry.get("stump_line") or {}
    if point is None or not geometry.get("available") or not line.get("start") or not line.get("end"):
        return {
            "available": False,
            "axis_progress": None,
            "lateral_offset_px": None,
            "notes": ["Pitch axis unavailable."],
        }

    projection = _line_projection(point, line["start"], line["end"])
    if projection is None:
        return {
            "available": False,
            "axis_progress": None,
            "lateral_offset_px": None,
            "notes": ["Invalid pitch axis."],
        }
    progress, offset = projection
    notes = []
    if progress < 0 or progress > 1:
        notes.append("Point is outside the calibrated stump-to-stump range.")
    return {
        "available": True,
        "axis_progress": float(progress),
        "lateral_offset_px": float(offset),
        "notes": notes,
    }


def build_replay_calibration_report(calibration, frame_size=None):
    normalized = normalize_stump_calibration(calibration, frame_size=frame_size)
    geometry = build_pitch_geometry(normalized.get("points"), frame_size=normalized.get("frame_size"))
    points = normalized.get("points") or {}
    valid_count = len(points)
    if not normalized.get("available"):
        quality = "Unavailable"
    elif valid_count == 6:
        quality = "Good"
    elif valid_count > 2:
        quality = "Partial"
    else:
        quality = "Basic"
    notes = list(normalized.get("notes") or [])
    for note in geometry.get("notes") or []:
        if note not in notes:
            notes.append(note)
    return {
        "available": bool(normalized.get("available")),
        "calibration_points": points,
        "pitch_geometry": geometry,
        "quality": quality,
        "notes": notes,
    }
