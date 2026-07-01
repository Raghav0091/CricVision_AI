"""Deterministic 2D pitch-corridor and stump-line reference estimates."""

from __future__ import annotations

from math import isfinite


def estimate_pitch_corridor(
    stump_reference,
    frame_width,
    frame_height,
    camera_view="umpire_end",
):
    """Estimate a frame-coordinate pitch corridor from a stump reference."""
    width = _dimension(frame_width, 1280)
    height = _dimension(frame_height, 720)
    stump = stump_reference if isinstance(stump_reference, dict) else {}
    center = _point(stump.get("center")) or [width * 0.5, height * 0.68]
    bbox = _bbox(stump.get("bbox"))
    stump_width = (
        max(4.0, bbox[2] - bbox[0])
        if bbox is not None
        else width * 0.05
    )
    camera_view = str(camera_view or "unknown").lower()

    if camera_view == "side_view":
        left = width * 0.06
        right = width * 0.94
        top = max(0.0, center[1] - height * 0.14)
        bottom = min(float(height), center[1] + height * 0.14)
        polygon = [[left, top], [right, top], [right, bottom], [left, bottom]]
        batter_end_y = center[1]
        bowler_end_y = center[1]
    else:
        far_y = max(0.0, min(center[1], height * 0.76))
        near_y = height * 0.96
        far_half_width = max(stump_width * 0.9, width * 0.045)
        near_half_width = max(stump_width * 3.5, width * 0.22)
        polygon = [
            [center[0] - far_half_width, far_y],
            [center[0] + far_half_width, far_y],
            [center[0] + near_half_width, near_y],
            [center[0] - near_half_width, near_y],
        ]
        if camera_view == "batter_view":
            batter_end_y, bowler_end_y = near_y, far_y
        else:
            batter_end_y, bowler_end_y = far_y, near_y

    polygon = [
        [
            round(min(max(point[0], 0.0), float(width)), 3),
            round(min(max(point[1], 0.0), float(height)), 3),
        ]
        for point in polygon
    ]
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    confidence = _confidence(stump)
    pitch_corridor = {
        "polygon": polygon,
        "bbox": [min(xs), min(ys), max(xs), max(ys)],
        "source": "estimated",
        "status": "estimated",
        "confidence": round(max(0.1, confidence * 0.8), 3),
    }
    return {
        **pitch_corridor,
        "pitch_corridor": pitch_corridor,
        "pitch_ends": {
            "batter_end_y": round(batter_end_y, 3),
            "bowler_end_y": round(bowler_end_y, 3),
            "source": "estimated",
        },
        "crease_line": {
            "y": round(batter_end_y, 3),
            "source": "estimated",
            "status": "estimated",
        },
        "notes": [
            "Pitch corridor, crease, and pitch ends are approximate 2D "
            "frame references."
        ],
    }


def estimate_line_reference(stump_reference, batter_handedness="right"):
    """Estimate off-, middle-, and leg-stump x references."""
    stump = stump_reference if isinstance(stump_reference, dict) else {}
    bbox = _bbox(stump.get("bbox"))
    center = _point(stump.get("center"))
    if center is None and bbox is not None:
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    if center is None:
        return {
            "off_stump_x": None,
            "middle_stump_x": None,
            "leg_stump_x": None,
            "source": "missing",
            "confidence": 0.0,
        }

    stump_width = max(6.0, bbox[2] - bbox[0]) if bbox else 36.0
    offset = stump_width / 3
    handedness = str(batter_handedness or "unknown").lower()
    off_x = center[0] - offset
    leg_x = center[0] + offset
    if handedness == "left":
        off_x, leg_x = leg_x, off_x
    return {
        "off_stump_x": round(off_x, 3),
        "middle_stump_x": round(center[0], 3),
        "leg_stump_x": round(leg_x, 3),
        "source": "estimated",
        "confidence": round(_confidence(stump) * 0.85, 3),
    }


def _point(value):
    try:
        if value is None or len(value) < 2:
            return None
        point = [float(value[0]), float(value[1])]
        return point if all(isfinite(item) for item in point) else None
    except (TypeError, ValueError):
        return None


def _bbox(value):
    try:
        if value is None or len(value) < 4:
            return None
        box = [float(item) for item in value[:4]]
        return box if all(isfinite(item) for item in box) else None
    except (TypeError, ValueError):
        return None


def _confidence(stump):
    try:
        return min(max(float(stump.get("confidence", 0.0)), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.0


def _dimension(value, fallback):
    try:
        value = int(value)
        return value if value > 0 else fallback
    except (TypeError, ValueError):
        return fallback
