"""Manual pitch calibration v1 — pitch ROI normalization and point scoring.

Pure Python helpers. No Streamlit, YOLO, OpenCV, or model loading.
Handles missing/invalid pitch ROI safely; consumers must treat the ROI
as an optional hint, never a hard requirement.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

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


def _parse_xy(item: Any) -> tuple[float, float] | None:
    if isinstance(item, dict):
        x = _safe_float(item.get("x"))
        y = _safe_float(item.get("y"))
        if x is not None and y is not None:
            return x, y
        return None
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        x = _safe_float(item[0])
        y = _safe_float(item[1])
        if x is not None and y is not None:
            return x, y
    return None


def _parse_polygon(points: Any) -> list[list[float]]:
    if not isinstance(points, (list, tuple)):
        return []
    polygon: list[list[float]] = []
    for vertex in points:
        xy = _parse_xy(vertex)
        if xy is not None:
            polygon.append([round(xy[0], 3), round(xy[1], 3)])
    return polygon


def _parse_bbox(bbox: Any) -> tuple[float, float, float, float] | None:
    if isinstance(bbox, dict):
        values = [_safe_float(bbox.get(key)) for key in ("x1", "y1", "x2", "y2")]
        if None in values:
            return None
        x1, y1, x2, y2 = values
    elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        values = [_safe_float(value) for value in bbox[:4]]
        if None in values:
            return None
        x1, y1, x2, y2 = values
    else:
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def normalize_pitch_roi(pitch_roi: Any, frame_size: Any = None) -> dict[str, Any]:
    """Normalize rectangle/polygon/dict pitch ROI inputs into one safe format."""
    notes: list[str] = []
    unavailable = {"available": False, "polygon": [], "bbox": None, "notes": notes}

    if pitch_roi is None:
        notes.append("No pitch ROI provided.")
        return unavailable

    polygon: list[list[float]] = []
    bbox: tuple[float, float, float, float] | None = None

    if isinstance(pitch_roi, dict):
        if pitch_roi.get("available") is not None and pitch_roi.get("polygon"):
            # Already normalized; re-parse to stay defensive.
            polygon = _parse_polygon(pitch_roi.get("polygon"))
        else:
            bbox = _parse_bbox(pitch_roi.get("bbox") or pitch_roi.get("roi_box"))
            if bbox is None:
                polygon = _parse_polygon(pitch_roi.get("polygon") or pitch_roi.get("corners"))
    elif isinstance(pitch_roi, (list, tuple)):
        if len(pitch_roi) == 4 and all(isinstance(v, (int, float)) for v in pitch_roi):
            bbox = _parse_bbox(pitch_roi)
        else:
            polygon = _parse_polygon(pitch_roi)

    if bbox is None and len(polygon) >= 3:
        xs = [vertex[0] for vertex in polygon]
        ys = [vertex[1] for vertex in polygon]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        notes.append(f"Pitch ROI from {len(polygon)}-point polygon.")
    elif bbox is not None and not polygon:
        x1, y1, x2, y2 = bbox
        polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        notes.append("Pitch ROI from rectangle.")

    if bbox is None:
        notes.append("Pitch ROI input was invalid; ignoring it.")
        return unavailable

    x1, y1, x2, y2 = bbox
    if (x2 - x1) < 2.0 or (y2 - y1) < 2.0:
        notes.append("Pitch ROI area is degenerate; ignoring it.")
        return unavailable

    size = _parse_frame_size(frame_size)
    if size is not None:
        width, height = size
        if x1 >= width or y1 >= height or x2 <= 0 or y2 <= 0:
            notes.append("Pitch ROI lies fully outside the frame; ignoring it.")
            return unavailable

    return {
        "available": True,
        "polygon": polygon,
        "bbox": {"x1": round(x1, 3), "y1": round(y1, 3), "x2": round(x2, 3), "y2": round(y2, 3)},
        "notes": notes,
    }


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


def point_inside_pitch_roi(point: Any, pitch_roi: Any, margin: float = 0) -> bool:
    """True when the point sits inside (or within `margin` px of) the pitch ROI."""
    xy = _parse_xy(point)
    if xy is None:
        return False
    roi = normalize_pitch_roi(pitch_roi)
    if not roi["available"]:
        return False
    # ponytail: bbox containment is enough for MVP; exact polygon test not needed yet.
    bbox = roi["bbox"]
    m = _safe_float(margin, 0.0) or 0.0
    return (
        bbox["x1"] - m <= xy[0] <= bbox["x2"] + m
        and bbox["y1"] - m <= xy[1] <= bbox["y2"] + m
    )


def score_point_against_pitch_roi(point: Any, pitch_roi: Any, margin: float = 40) -> dict[str, Any]:
    """Score a point against the pitch ROI: inside > near > outside."""
    notes: list[str] = []
    roi = normalize_pitch_roi(pitch_roi)
    if not roi["available"]:
        notes.append("No usable pitch ROI; no pitch scoring applied.")
        return {"inside": False, "near": False, "score_bonus": 0.0, "notes": notes}
    if _parse_xy(point) is None:
        notes.append("Point coordinates invalid; no pitch scoring applied.")
        return {"inside": False, "near": False, "score_bonus": 0.0, "notes": notes}

    inside = point_inside_pitch_roi(point, roi, margin=0)
    near = inside or point_inside_pitch_roi(point, roi, margin=margin)
    if inside:
        bonus = INSIDE_BONUS
        notes.append("Point is inside the pitch ROI.")
    elif near:
        bonus = NEAR_BONUS
        notes.append(f"Point is within {margin}px of the pitch ROI.")
    else:
        bonus = OUTSIDE_PENALTY
        notes.append("Point is far outside the pitch ROI.")
    return {"inside": inside, "near": near, "score_bonus": bonus, "notes": notes}
