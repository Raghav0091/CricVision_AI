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


def _box_as_ints(box: Any) -> dict[str, int] | None:
    """Convert a normalized stump box to integer pixel coords for live UI/overlays."""
    normalized = _normalize_box(box)
    if normalized is None:
        return None
    return {
        "x1": int(round(normalized["x1"])),
        "y1": int(round(normalized["y1"])),
        "x2": int(round(normalized["x2"])),
        "y2": int(round(normalized["y2"])),
    }


def build_default_live_stump_boxes(frame_size: Any) -> dict[str, Any]:
    """Live-oriented default near/far stump boxes (wraps Video Analysis layout)."""
    layout = build_default_stump_box_layout(frame_size)
    notes = list(layout.get("notes") or [])
    if not layout.get("available"):
        return {
            "available": False,
            "near_stumps_box": None,
            "far_stumps_box": None,
            "notes": notes or ["Invalid frame_size; default live stump boxes unavailable."],
        }

    near = _box_as_ints(layout.get("near_stumps_box"))
    far = _box_as_ints(layout.get("far_stumps_box"))
    if near is None or far is None:
        notes.append("Could not convert default stump boxes to integer live boxes.")
        return {
            "available": False,
            "near_stumps_box": None,
            "far_stumps_box": None,
            "notes": notes,
        }

    notes.append("Live default stump boxes adapted from Video Analysis layout.")
    return {
        "available": True,
        "near_stumps_box": near,
        "far_stumps_box": far,
        "notes": notes,
    }


def build_calibration_from_stump_boxes(
    near_box: Any,
    far_box: Any,
    frame_size: Any = None,
) -> dict[str, Any]:
    """Build live calibration from near/far stump boxes (wraps build_calibration_from_boxes)."""
    report = build_calibration_from_boxes(
        {"near_stumps_box": near_box, "far_stumps_box": far_box},
        frame_size=frame_size,
    )
    # Keep Video Analysis float geometry; expose int boxes for live UI convenience.
    near_ints = _box_as_ints(report.get("near_stumps_box"))
    far_ints = _box_as_ints(report.get("far_stumps_box"))
    if near_ints is not None:
        report["near_stumps_box"] = near_ints
    if far_ints is not None:
        report["far_stumps_box"] = far_ints
    return report


def point_inside_live_pitch_corridor(
    point: Any,
    calibration: Any,
    margin: float = 40,
) -> dict[str, Any]:
    """Live alias for point_inside_calibrated_corridor."""
    return point_inside_calibrated_corridor(point, calibration, margin=margin)


def build_live_calibration_report(
    near_box: Any,
    far_box: Any,
    frame_size: Any = None,
) -> dict[str, Any]:
    """Compact live calibration status wrapper for Streamlit Live Session."""
    calibration = build_calibration_from_stump_boxes(near_box, far_box, frame_size=frame_size)
    notes = list(calibration.get("notes") or [])
    available = bool(calibration.get("available"))
    stump_line = calibration.get("stump_line")
    corridor = calibration.get("pitch_corridor") or []
    return {
        "available": available,
        "quality": calibration.get("quality", "Unavailable"),
        "stump_line_available": bool(stump_line),
        "pitch_corridor_available": bool(corridor),
        "calibration": calibration,
        "notes": notes,
    }


def _unavailable_alignment_layout(notes: list[str]) -> dict[str, Any]:
    return {
        "available": False,
        "striker_stumps_box": None,
        "non_striker_stumps_box": None,
        "notes": notes,
    }


def _unavailable_alignment_calibration(notes: list[str]) -> dict[str, Any]:
    return {
        "available": False,
        "quality": "Unavailable",
        "striker_stumps_box": None,
        "non_striker_stumps_box": None,
        "stump_line": None,
        "pitch_corridor": [],
        "notes": notes,
    }


def build_premium_stump_alignment_boxes(frame_size: Any) -> dict[str, Any]:
    """Mobile/tablet-first FullTrack boxes — portrait striker upper, non-striker lower/bigger."""
    notes: list[str] = []
    size = _parse_frame_size(frame_size)
    if size is None:
        notes.append("Invalid frame_size; premium alignment boxes unavailable.")
        return _unavailable_alignment_layout(notes)

    width, height = size
    portrait = height > width
    # ponytail: portrait = phone/tablet upright; landscape = wider webcam/tablet.
    if portrait:
        striker_w, striker_h, striker_cy = 0.34, 0.22, 0.28
        non_w, non_h, non_cy = 0.68, 0.32, 0.74
    else:
        striker_w, striker_h, striker_cy = 0.24, 0.22, 0.26
        non_w, non_h, non_cy = 0.40, 0.30, 0.76

    def _centered_box(box_w_frac: float, box_h_frac: float, cy_frac: float) -> dict[str, float] | None:
        box_w = width * box_w_frac
        box_h = height * box_h_frac
        cx = width * 0.5
        cy = height * cy_frac
        return _normalize_box(
            {
                "x1": cx - box_w / 2.0,
                "y1": cy - box_h / 2.0,
                "x2": cx + box_w / 2.0,
                "y2": cy + box_h / 2.0,
            },
            frame_size=size,
        )

    striker = _centered_box(striker_w, striker_h, striker_cy)
    non_striker = _centered_box(non_w, non_h, non_cy)
    if striker is None or non_striker is None:
        notes.append("Could not build premium alignment boxes for this frame size.")
        return _unavailable_alignment_layout(notes)

    striker_ints = _box_as_ints(striker)
    non_striker_ints = _box_as_ints(non_striker)
    if striker_ints is None or non_striker_ints is None:
        notes.append("Could not convert premium alignment boxes to integer pixels.")
        return _unavailable_alignment_layout(notes)

    notes.append("Premium mobile/tablet alignment boxes — move camera until stumps fit.")
    notes.append("Estimated single-camera geometry only — not official LBW/DRS.")
    return {
        "available": True,
        "striker_stumps_box": striker_ints,
        "non_striker_stumps_box": non_striker_ints,
        "frame_size": {"width": width, "height": height},
        "orientation": "portrait" if portrait else "landscape",
        "notes": notes,
    }


def build_fulltrack_style_box_layout(frame_size: Any) -> dict[str, Any]:
    """Fixed FullTrack-style alignment boxes — wraps premium layout for backward compat."""
    return build_premium_stump_alignment_boxes(frame_size)


def build_environment_context_from_validated_stumps(
    box_layout: Any,
    validation_result: Any,
    frame_size: Any = None,
) -> dict[str, Any]:
    """Hidden pitch/environment context from validated stumps — not official LBW/DRS."""
    notes: list[str] = [
        "Estimated single-camera geometry only — not official LBW/DRS.",
    ]
    validation = validation_result if isinstance(validation_result, dict) else {}
    layout = box_layout if isinstance(box_layout, dict) else {}

    striker_found = bool((validation.get("striker") or {}).get("found"))
    non_found = bool((validation.get("non_striker") or {}).get("found"))
    stumps_validated = bool(validation.get("valid"))
    quality = validation.get("quality") or "Not Found"

    view_direction = "Unknown"
    if not stumps_validated:
        view_direction = "Partial" if striker_found or non_found else "Unknown"
    elif quality == "Strong":
        view_direction = "Good"
    elif quality in {"Found", "Weak"}:
        view_direction = "Usable"
    else:
        view_direction = "Poor"

    striker_box = _normalize_box(layout.get("striker_stumps_box"), frame_size=frame_size)
    non_striker_box = _normalize_box(layout.get("non_striker_stumps_box"), frame_size=frame_size)
    striker_center = _box_center(striker_box) if striker_box else None
    non_striker_center = _box_center(non_striker_box) if non_striker_box else None

    pitch_axis = None
    pitch_corridor = None
    scale_reference = None
    available = stumps_validated and striker_box is not None and non_striker_box is not None

    if available:
        pitch_axis = {"start": non_striker_center, "end": striker_center}
        notes.append("Pitch axis: non-striker centre → striker centre.")
        base = build_calibration_from_alignment_boxes(layout, frame_size=frame_size)
        if base.get("available"):
            pitch_corridor = list(base.get("pitch_corridor") or [])
            notes.append("Pitch corridor estimated from alignment boxes.")
        avg_w = 0.0
        if striker_box and non_striker_box:
            avg_w = (_box_width(striker_box) + _box_width(non_striker_box)) / 2.0
        scale_reference = {"stump_box_width_px": round(avg_w, 2)}
    elif not stumps_validated:
        notes.append("Validation failed — environment context not created.")
        available = False

    return {
        "available": available,
        "stumps_validated": stumps_validated,
        "view_direction": view_direction,
        "striker_box": striker_box,
        "non_striker_box": non_striker_box,
        "striker_center": striker_center,
        "non_striker_center": non_striker_center,
        "pitch_axis": pitch_axis,
        "pitch_corridor": pitch_corridor,
        "scale_reference": scale_reference,
        "quality": quality,
        # backward compat keys
        "camera_view": view_direction,
        "striker_stumps_found": striker_found,
        "non_striker_stumps_found": non_found,
        "stump_line": pitch_axis,
        "notes": notes,
    }


def build_calibration_from_alignment_boxes(
    box_layout: Any,
    frame_size: Any = None,
) -> dict[str, Any]:
    """Build stump line + corridor from striker / non-striker alignment boxes."""
    notes: list[str] = [
        "Estimated single-camera geometry only — not official LBW or DRS.",
    ]
    if not isinstance(box_layout, dict):
        notes.append("Alignment box layout missing or invalid.")
        return _unavailable_alignment_calibration(notes)

    size = _parse_frame_size(frame_size)
    if size is None and box_layout.get("frame_size") is not None:
        size = _parse_frame_size(box_layout.get("frame_size"))

    striker = _normalize_box(box_layout.get("striker_stumps_box"), frame_size=size)
    non_striker = _normalize_box(box_layout.get("non_striker_stumps_box"), frame_size=size)
    if striker is None or non_striker is None:
        notes.append("Striker and non-striker stump boxes are required and must be valid.")
        return _unavailable_alignment_calibration(notes)

    # Map alignment names → near/far Video Analysis geometry, then re-key for live UX.
    base = build_calibration_from_boxes(
        {"near_stumps_box": non_striker, "far_stumps_box": striker},
        frame_size=size,
    )
    notes.extend(list(base.get("notes") or []))

    striker_ints = _box_as_ints(striker)
    non_striker_ints = _box_as_ints(non_striker)
    stump_line = base.get("stump_line")
    # Explicit blue-line direction: non-striker centre → striker centre.
    if striker_ints is not None and non_striker_ints is not None:
        stump_line = {
            "start": _box_center(
                {
                    "x1": float(non_striker_ints["x1"]),
                    "y1": float(non_striker_ints["y1"]),
                    "x2": float(non_striker_ints["x2"]),
                    "y2": float(non_striker_ints["y2"]),
                }
            ),
            "end": _box_center(
                {
                    "x1": float(striker_ints["x1"]),
                    "y1": float(striker_ints["y1"]),
                    "x2": float(striker_ints["x2"]),
                    "y2": float(striker_ints["y2"]),
                }
            ),
        }

    if not base.get("available"):
        return {
            "available": False,
            "quality": base.get("quality", "Unavailable"),
            "striker_stumps_box": striker_ints,
            "non_striker_stumps_box": non_striker_ints,
            "stump_line": stump_line,
            "pitch_corridor": list(base.get("pitch_corridor") or []),
            "notes": notes,
        }

    return {
        "available": True,
        "quality": base.get("quality", "Basic"),
        "striker_stumps_box": striker_ints,
        "non_striker_stumps_box": non_striker_ints,
        "stump_line": stump_line,
        "pitch_corridor": list(base.get("pitch_corridor") or []),
        "notes": notes,
    }


def build_live_alignment_report(
    box_layout: Any,
    frame_size: Any = None,
) -> dict[str, Any]:
    """Compact live alignment calibration status for Streamlit Live Session."""
    calibration = build_calibration_from_alignment_boxes(box_layout, frame_size=frame_size)
    notes = list(calibration.get("notes") or [])
    available = bool(calibration.get("available"))
    stump_line = calibration.get("stump_line")
    corridor = calibration.get("pitch_corridor") or []
    return {
        "available": available,
        "quality": calibration.get("quality", "Unavailable"),
        "stump_line_available": bool(stump_line),
        "pitch_corridor_available": bool(corridor),
        "calibration": calibration,
        "notes": notes,
    }


def build_calibration_from_validated_stumps(
    box_layout: Any,
    validation_result: Any,
    frame_size: Any = None,
) -> dict[str, Any]:
    """Build live calibration only after stump validation passes."""
    notes: list[str] = [
        "Estimated single-camera geometry only — not official LBW or DRS.",
    ]
    validation = validation_result if isinstance(validation_result, dict) else {}
    if not validation.get("valid"):
        notes.append("Stump validation did not pass; calibration unavailable.")
        notes.extend(list(validation.get("notes") or []))
        striker_box = None
        non_striker_box = None
        if isinstance(box_layout, dict):
            striker_box = box_layout.get("striker_stumps_box")
            non_striker_box = box_layout.get("non_striker_stumps_box")
        return {
            "available": False,
            "quality": validation.get("quality", "Not Found"),
            "stumps_validated": False,
            "striker_stumps_box": striker_box,
            "non_striker_stumps_box": non_striker_box,
            "stump_line": None,
            "pitch_corridor": None,
            "environment_context": build_environment_context_from_validated_stumps(
                box_layout,
                validation,
                frame_size=frame_size,
            ),
            "notes": notes,
        }

    base = build_calibration_from_alignment_boxes(box_layout, frame_size=frame_size)
    notes.extend(list(base.get("notes") or []))
    environment_context = build_environment_context_from_validated_stumps(
        box_layout,
        validation,
        frame_size=frame_size,
    )
    quality = validation.get("quality") or base.get("quality", "Partial")
    available = bool(base.get("available"))
    if not available:
        notes.append("Validated stumps present but corridor geometry unavailable.")

    return {
        "available": available,
        "quality": quality,
        "stumps_validated": True,
        "striker_stumps_box": base.get("striker_stumps_box"),
        "non_striker_stumps_box": base.get("non_striker_stumps_box"),
        "stump_line": base.get("stump_line") if available else None,
        "pitch_corridor": list(base.get("pitch_corridor") or []) if available else None,
        "environment_context": environment_context,
        "notes": notes,
    }
