"""FullTrack-style session calibration v1 — stump boxes to pitch corridor.

Pure geometry helpers. No Streamlit, YOLO, OpenCV, or model loading.
Estimated single-camera geometry only — never official LBW/DRS.
"""

from __future__ import annotations

from math import hypot, isfinite
from typing import Any

from Backends.src.pitch_calibration import (
    normalize_pitch_roi,
    score_point_against_pitch_roi,
)

# ponytail: bonuses mirror pitch_calibration so ball tracker scoring stays consistent.
INSIDE_BONUS = 0.5
NEAR_BONUS = 0.15
OUTSIDE_PENALTY = -0.6


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if isfinite(result) else default


def _parse_frame_size(frame_size: Any) -> tuple[float, float] | None:
    if isinstance(frame_size, dict):
        width = _safe_float(frame_size.get("width") or frame_size.get("frame_width"))
        height = _safe_float(frame_size.get("height") or frame_size.get("frame_height"))
    elif isinstance(frame_size, (list, tuple)) and len(frame_size) >= 2:
        width = _safe_float(frame_size[0])
        height = _safe_float(frame_size[1])
    else:
        return None
    if width and height and width > 0 and height > 0:
        return width, height
    return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_box(box: Any, frame_size: Any = None) -> dict[str, float] | None:
    """Normalize a stump box to {x1,y1,x2,y2}, clamped to the frame when known."""
    if isinstance(box, dict):
        values = [_safe_float(box.get(key)) for key in ("x1", "y1", "x2", "y2")]
        if None in values:
            return None
        x1, y1, x2, y2 = values
    elif isinstance(box, (list, tuple)) and len(box) >= 4:
        values = [_safe_float(value) for value in box[:4]]
        if None in values:
            return None
        x1, y1, x2, y2 = values
    else:
        return None

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if (x2 - x1) < 2.0 or (y2 - y1) < 2.0:
        return None

    size = _parse_frame_size(frame_size)
    if size is not None:
        width, height = size
        x1 = _clamp(x1, 0.0, width)
        x2 = _clamp(x2, 0.0, width)
        y1 = _clamp(y1, 0.0, height)
        y2 = _clamp(y2, 0.0, height)
        if (x2 - x1) < 2.0 or (y2 - y1) < 2.0:
            return None

    return {
        "x1": round(x1, 3),
        "y1": round(y1, 3),
        "x2": round(x2, 3),
        "y2": round(y2, 3),
    }


def _box_center(box: dict[str, float]) -> dict[str, float]:
    return {
        "x": round((box["x1"] + box["x2"]) / 2.0, 3),
        "y": round((box["y1"] + box["y2"]) / 2.0, 3),
    }


def _box_width(box: dict[str, float]) -> float:
    return max(0.0, box["x2"] - box["x1"])


def _unavailable_layout(notes: list[str]) -> dict[str, Any]:
    return {
        "available": False,
        "near_stumps_box": None,
        "far_stumps_box": None,
        "pitch_center_line": None,
        "notes": notes,
    }


def _unavailable_calibration(notes: list[str]) -> dict[str, Any]:
    return {
        "available": False,
        "near_stumps_box": None,
        "far_stumps_box": None,
        "stump_line": None,
        "pitch_corridor": [],
        "quality": "Unavailable",
        "notes": notes,
    }


def build_default_stump_box_layout(frame_size: Any) -> dict[str, Any]:
    """Create FullTrack-like near/far stump boxes for a typical behind-bowler camera."""
    notes: list[str] = []
    size = _parse_frame_size(frame_size)
    if size is None:
        notes.append("Invalid frame_size; default stump boxes unavailable.")
        return _unavailable_layout(notes)

    width, height = size
    # ponytail: side-on / behind-non-striker defaults — near lower, far higher.
    near = _normalize_box(
        {
            "x1": width * 0.42,
            "y1": height * 0.72,
            "x2": width * 0.58,
            "y2": height * 0.92,
        },
        frame_size=size,
    )
    far = _normalize_box(
        {
            "x1": width * 0.45,
            "y1": height * 0.22,
            "x2": width * 0.55,
            "y2": height * 0.38,
        },
        frame_size=size,
    )
    if near is None or far is None:
        notes.append("Could not build default stump boxes for this frame size.")
        return _unavailable_layout(notes)

    near_c = _box_center(near)
    far_c = _box_center(far)
    notes.append("Default stump boxes for typical behind-non-striker / side-on camera.")
    notes.append("Estimated single-camera geometry only — not official LBW/DRS.")
    return {
        "available": True,
        "near_stumps_box": near,
        "far_stumps_box": far,
        "pitch_center_line": {"start": near_c, "end": far_c},
        "notes": notes,
    }


def _corridor_around_line(
    near_center: dict[str, float],
    far_center: dict[str, float],
    half_width: float,
) -> list[dict[str, float]] | None:
    dx = far_center["x"] - near_center["x"]
    dy = far_center["y"] - near_center["y"]
    length = hypot(dx, dy)
    if length < 1.0 or half_width <= 0:
        return None
    nx = -dy / length
    ny = dx / length
    return [
        {"x": round(near_center["x"] + nx * half_width, 3), "y": round(near_center["y"] + ny * half_width, 3)},
        {"x": round(far_center["x"] + nx * half_width, 3), "y": round(far_center["y"] + ny * half_width, 3)},
        {"x": round(far_center["x"] - nx * half_width, 3), "y": round(far_center["y"] - ny * half_width, 3)},
        {"x": round(near_center["x"] - nx * half_width, 3), "y": round(near_center["y"] - ny * half_width, 3)},
    ]


def _assess_quality(
    near: dict[str, float],
    far: dict[str, float],
    frame_size: tuple[float, float] | None,
) -> str:
    near_c = _box_center(near)
    far_c = _box_center(far)
    separation = hypot(far_c["x"] - near_c["x"], far_c["y"] - near_c["y"])
    min_sep = 40.0
    if frame_size is not None:
        min_sep = max(40.0, frame_size[1] * 0.12)

    # Vertical separation preferred for behind-bowler; horizontal drift is weak alignment.
    vertical_ok = abs(far_c["y"] - near_c["y"]) >= min_sep * 0.6
    lateral_drift = abs(far_c["x"] - near_c["x"])
    max_drift = max(_box_width(near), _box_width(far)) * 1.5
    if frame_size is not None:
        max_drift = max(max_drift, frame_size[0] * 0.12)
    aligned = lateral_drift <= max_drift
    far_above_near = far_c["y"] < near_c["y"]

    if separation >= min_sep and vertical_ok and aligned and far_above_near:
        return "Good"
    if separation >= min_sep * 0.5:
        return "Basic"
    return "Unavailable"


def build_calibration_from_boxes(box_layout: Any, frame_size: Any = None) -> dict[str, Any]:
    """Estimate stump line + pitch corridor from near/far stump boxes."""
    notes: list[str] = [
        "Estimated single-camera geometry only — not official LBW or DRS.",
    ]
    if not isinstance(box_layout, dict):
        notes.append("Box layout missing or invalid.")
        return _unavailable_calibration(notes)

    size = _parse_frame_size(frame_size)
    if size is None and box_layout.get("frame_size") is not None:
        size = _parse_frame_size(box_layout.get("frame_size"))

    near = _normalize_box(box_layout.get("near_stumps_box"), frame_size=size)
    far = _normalize_box(box_layout.get("far_stumps_box"), frame_size=size)
    if near is None or far is None:
        notes.append("Near and far stump boxes are required and must be valid.")
        return _unavailable_calibration(notes)

    quality = _assess_quality(near, far, size)
    if quality == "Unavailable":
        notes.append("Stump boxes are too close or poorly placed for a usable corridor.")
        return {
            "available": False,
            "near_stumps_box": near,
            "far_stumps_box": far,
            "stump_line": None,
            "pitch_corridor": [],
            "quality": "Unavailable",
            "notes": notes,
        }

    near_c = _box_center(near)
    far_c = _box_center(far)
    stump_line = {"start": near_c, "end": far_c}

    # Corridor half-width from stump box widths, with a frame-width floor.
    avg_box_w = (_box_width(near) + _box_width(far)) / 2.0
    half_width = max(avg_box_w * 1.8, 40.0)
    if size is not None:
        half_width = max(half_width, size[0] * 0.06)
        half_width = min(half_width, size[0] * 0.28)

    corridor = _corridor_around_line(near_c, far_c, half_width)
    if not corridor:
        notes.append("Could not build pitch corridor from stump line.")
        return {
            "available": False,
            "near_stumps_box": near,
            "far_stumps_box": far,
            "stump_line": stump_line,
            "pitch_corridor": [],
            "quality": "Unavailable",
            "notes": notes,
        }

    notes.append(f"Calibration quality: {quality}.")
    return {
        "available": True,
        "near_stumps_box": near,
        "far_stumps_box": far,
        "stump_line": stump_line,
        "pitch_corridor": corridor,
        "quality": quality,
        "notes": notes,
    }


def point_inside_calibrated_corridor(
    point: Any,
    calibration: Any,
    margin: float = 40,
) -> dict[str, Any]:
    """Score a ball candidate against the session-calibrated pitch corridor."""
    notes: list[str] = []
    calibration = calibration or {}
    if not isinstance(calibration, dict) or not calibration.get("available"):
        notes.append("Session calibration unavailable; no corridor scoring applied.")
        return {"inside": False, "near": False, "score_bonus": 0.0, "notes": notes}

    corridor = calibration.get("pitch_corridor")
    roi = normalize_pitch_roi(corridor)
    if not roi.get("available"):
        # Fall back: wrap polygon list / stump-line corridor into ROI shape.
        roi = normalize_pitch_roi({"polygon": corridor} if corridor else None)
    if not roi.get("available"):
        notes.append("Pitch corridor invalid; no corridor scoring applied.")
        return {"inside": False, "near": False, "score_bonus": 0.0, "notes": notes}

    scored = score_point_against_pitch_roi(point, roi, margin=margin)
    scored_notes = list(scored.get("notes") or [])
    scored_notes.extend(notes)
    scored["notes"] = scored_notes
    return scored


def session_calibration_as_pitch_roi(calibration: Any) -> dict[str, Any] | None:
    """Convert session calibration corridor into a normalize_pitch_roi-compatible ROI."""
    calibration = calibration or {}
    if not isinstance(calibration, dict) or not calibration.get("available"):
        return None
    corridor = calibration.get("pitch_corridor")
    roi = normalize_pitch_roi({"polygon": corridor} if corridor else corridor)
    if not roi.get("available"):
        return None
    roi["source"] = "session_calibration"
    return roi
