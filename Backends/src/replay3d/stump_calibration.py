"""Build stump/pitch calibration context for estimated 3D replay (read-only inputs)."""

from __future__ import annotations

from math import isfinite
from typing import Any

from Backends.src.analysis.frame_detection_utils import (
    best_detection_center,
    normalize_frame_detections,
)

PITCH_LENGTH_FT = 66.0
PITCH_WIDTH_FT = 10.0
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720

CAMERA_VIEW_ALIASES = {
    "umpire end": "umpire_end",
    "umpire_end": "umpire_end",
    "behind bowler": "umpire_end",
    "batter view": "batter_view",
    "batter_view": "batter_view",
    "bowler end": "bowler_end",
    "bowler_end": "bowler_end",
    "side view": "side_view",
    "side_view": "side_view",
    "unknown": "unknown",
}


def build_stump_calibration_context(
    frame_size: Any = None,
    stump_detections: Any = None,
    pitch_roi: Any = None,
    camera_height_ft: Any = None,
    camera_view: Any = None,
) -> dict[str, Any]:
    """Return JSON-safe calibration context for 3D replay estimation."""
    width, height = _parse_frame_size(frame_size)
    view = _normalize_camera_view(camera_view)
    height_ft = _positive_float(camera_height_ft, default=8.0)

    stumps = _collect_stump_detections(stump_detections)
    batter_stumps, bowler_stumps = _split_stumps_by_end(stumps, width, height, view)
    roi = _normalize_pitch_roi(pitch_roi, batter_stumps, bowler_stumps, width, height, view)
    centerline = _pitch_centerline(roi, batter_stumps, bowler_stumps, width, height, view)

    quality = _calibration_quality(
        enabled=width > 0 and height > 0,
        batter_stumps=batter_stumps,
        bowler_stumps=bowler_stumps,
        pitch_roi=roi,
    )
    notes = _build_notes(quality, batter_stumps, bowler_stumps, roi, view)

    return {
        "calibration_quality": quality,
        "frame_width": width,
        "frame_height": height,
        "camera_view": view,
        "camera_height_ft": round(height_ft, 2),
        "batter_stumps": batter_stumps,
        "bowler_stumps": bowler_stumps,
        "pitch_roi": roi,
        "pitch_centerline": centerline,
        "pitch_length_ft": PITCH_LENGTH_FT,
        "pitch_width_ft": PITCH_WIDTH_FT,
        "notes": notes,
    }


def _parse_frame_size(frame_size: Any) -> tuple[int, int]:
    if isinstance(frame_size, dict):
        width = frame_size.get("width") or frame_size.get("frame_width")
        height = frame_size.get("height") or frame_size.get("frame_height")
        return _dimension(width, DEFAULT_FRAME_WIDTH), _dimension(height, DEFAULT_FRAME_HEIGHT)
    if isinstance(frame_size, (list, tuple)) and len(frame_size) >= 2:
        return _dimension(frame_size[0], DEFAULT_FRAME_WIDTH), _dimension(
            frame_size[1], DEFAULT_FRAME_HEIGHT
        )
    return DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT


def _normalize_camera_view(value: Any) -> str:
    if value is None:
        return "unknown"
    key = str(value).strip().lower().replace("-", " ").replace("_", " ")
    return CAMERA_VIEW_ALIASES.get(key, CAMERA_VIEW_ALIASES.get(key.replace(" ", "_"), "unknown"))


def _collect_stump_detections(stump_detections: Any) -> list[dict[str, Any]]:
    if not stump_detections:
        return []
    if isinstance(stump_detections, dict):
        if any(key in stump_detections for key in ("bbox", "box", "xyxy", "center")):
            normalized = _normalize_stump(stump_detections)
            return [normalized] if normalized else []
        nested = stump_detections.get("stump_detections") or stump_detections.get("stumps")
        if nested:
            return _collect_stump_detections(nested)
        return []
    if isinstance(stump_detections, (list, tuple)):
        if stump_detections and all(
            isinstance(item, dict)
            and any(key in item for key in ("frame_index", "stump_detections", "stumps"))
            for item in stump_detections
        ):
            collected: list[dict[str, Any]] = []
            for frame in normalize_frame_detections(stump_detections):
                collected.extend(_collect_stump_detections(frame.get("stump_detections")))
            return collected
        result = []
        for item in stump_detections:
            normalized = _normalize_stump(item)
            if normalized:
                result.append(normalized)
        return result
    return []


def _normalize_stump(detection: Any) -> dict[str, Any] | None:
    if not isinstance(detection, dict):
        return None
    center = best_detection_center([detection])
    if center is None or not all(isfinite(value) for value in center):
        return None
    bbox = _bbox(detection)
    confidence = _confidence(detection)
    return {
        "center": [round(center[0], 2), round(center[1], 2)],
        "bbox": bbox,
        "confidence": round(confidence, 3),
    }


def _split_stumps_by_end(
    stumps: list[dict[str, Any]],
    width: int,
    height: int,
    camera_view: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not stumps:
        return (
            [
                {
                    "center": [round(width * 0.5, 2), round(height * 0.72, 2)],
                    "bbox": None,
                    "confidence": 0.1,
                    "source": "estimated",
                }
            ],
            [
                {
                    "center": [round(width * 0.5, 2), round(height * 0.28, 2)],
                    "bbox": None,
                    "confidence": 0.1,
                    "source": "estimated",
                }
            ],
        )

    sorted_stumps = sorted(stumps, key=lambda item: item["center"][1])
    if camera_view == "batter_view":
        bowler_stumps = sorted_stumps[: max(1, len(sorted_stumps) // 2)]
        batter_stumps = sorted_stumps[len(sorted_stumps) // 2 :]
    else:
        bowler_stumps = sorted_stumps[: max(1, len(sorted_stumps) // 2)]
        batter_stumps = sorted_stumps[len(sorted_stumps) // 2 :]

    if not batter_stumps:
        batter_stumps = [
            {
                "center": [round(width * 0.5, 2), round(height * 0.72, 2)],
                "bbox": None,
                "confidence": 0.1,
                "source": "estimated",
            }
        ]
    if not bowler_stumps:
        bowler_stumps = [
            {
                "center": [round(width * 0.5, 2), round(height * 0.28, 2)],
                "bbox": None,
                "confidence": 0.1,
                "source": "estimated",
            }
        ]
    return batter_stumps, bowler_stumps


def _normalize_pitch_roi(
    pitch_roi: Any,
    batter_stumps: list[dict[str, Any]],
    bowler_stumps: list[dict[str, Any]],
    width: int,
    height: int,
    camera_view: str,
) -> dict[str, Any] | None:
    if isinstance(pitch_roi, dict):
        polygon = pitch_roi.get("polygon")
        bbox = pitch_roi.get("bbox") or pitch_roi.get("roi_box")
        if polygon or bbox:
            normalized_bbox = _bbox_from_any(bbox) or _bbox_from_polygon(polygon)
            return {
                "polygon": _clean_polygon(polygon, width, height) if polygon else [],
                "bbox": normalized_bbox,
                "source": str(pitch_roi.get("source") or "provided"),
            }
    if isinstance(pitch_roi, (list, tuple)):
        if len(pitch_roi) == 4 and all(isinstance(item, (int, float)) for item in pitch_roi):
            return {
                "polygon": [],
                "bbox": [float(pitch_roi[0]), float(pitch_roi[1]), float(pitch_roi[2]), float(pitch_roi[3])],
                "source": "provided",
            }
        if pitch_roi and isinstance(pitch_roi[0], (list, tuple)):
            return {
                "polygon": _clean_polygon(pitch_roi, width, height),
                "bbox": _bbox_from_polygon(pitch_roi),
                "source": "provided",
            }

    batter_y = batter_stumps[0]["center"][1]
    bowler_y = bowler_stumps[0]["center"][1]
    center_x = (batter_stumps[0]["center"][0] + bowler_stumps[0]["center"][0]) / 2.0
    top_y = min(batter_y, bowler_y)
    bottom_y = max(batter_y, bowler_y)
    half_width = max(width * 0.12, 40.0)
    if camera_view == "side_view":
        half_width = width * 0.4
    polygon = [
        [center_x - half_width, top_y],
        [center_x + half_width, top_y],
        [center_x + half_width * 1.4, bottom_y],
        [center_x - half_width * 1.4, bottom_y],
    ]
    return {
        "polygon": _clean_polygon(polygon, width, height),
        "bbox": _bbox_from_polygon(polygon),
        "source": "estimated",
    }


def _pitch_centerline(
    pitch_roi: dict[str, Any] | None,
    batter_stumps: list[dict[str, Any]],
    bowler_stumps: list[dict[str, Any]],
    width: int,
    height: int,
    camera_view: str,
) -> dict[str, Any]:
    batter_center = batter_stumps[0]["center"]
    bowler_center = bowler_stumps[0]["center"]
    if pitch_roi and pitch_roi.get("bbox"):
        x1, _, x2, _ = pitch_roi["bbox"]
        center_x = (x1 + x2) / 2.0
    else:
        center_x = (batter_center[0] + bowler_center[0]) / 2.0

    if camera_view == "batter_view":
        bowler_end = bowler_center
        batter_end = batter_center
    else:
        bowler_end = bowler_center if bowler_center[1] < batter_center[1] else batter_center
        batter_end = batter_center if batter_center[1] > bowler_center[1] else bowler_center

    return {
        "image_line": [
            [round(center_x, 2), round(bowler_end[1], 2)],
            [round(center_x, 2), round(batter_end[1], 2)],
        ],
        "bowler_end_image": [round(bowler_end[0], 2), round(bowler_end[1], 2)],
        "batter_end_image": [round(batter_end[0], 2), round(batter_end[1], 2)],
        "pitch_length_ft": PITCH_LENGTH_FT,
    }


def _calibration_quality(
    *,
    enabled: bool,
    batter_stumps: list[dict[str, Any]],
    bowler_stumps: list[dict[str, Any]],
    pitch_roi: dict[str, Any] | None,
) -> str:
    if not enabled:
        return "Disabled"

    detected_batter = any(item.get("confidence", 0) >= 0.35 for item in batter_stumps)
    detected_bowler = any(item.get("confidence", 0) >= 0.35 for item in bowler_stumps)
    has_roi = bool(pitch_roi and pitch_roi.get("bbox"))
    roi_estimated = pitch_roi and pitch_roi.get("source") == "estimated"

    if detected_batter and detected_bowler and has_roi and not roi_estimated:
        return "Good"
    if (detected_batter or detected_bowler) and has_roi:
        return "Partial"
    if detected_batter or detected_bowler or has_roi:
        return "Low"
    return "Low"


def _build_notes(
    quality: str,
    batter_stumps: list[dict[str, Any]],
    bowler_stumps: list[dict[str, Any]],
    pitch_roi: dict[str, Any] | None,
    camera_view: str,
) -> list[str]:
    notes = [
        "Estimated 3D replay uses tracked image points and stump/pitch calibration only.",
        "Speed, swing, spin, and LBW are not available in v1.",
    ]
    if quality == "Disabled":
        notes.append("Calibration disabled: frame dimensions unavailable.")
    elif quality == "Low":
        notes.append("Low calibration quality: geometry is approximate.")
    elif quality == "Partial":
        notes.append("Partial calibration: one end or estimated pitch corridor in use.")
    if camera_view == "unknown":
        notes.append("Camera view unknown; default pitch orientation applied.")
    if pitch_roi and pitch_roi.get("source") == "estimated":
        notes.append("Pitch ROI estimated from stump positions.")
    if not any(item.get("confidence", 0) >= 0.35 for item in batter_stumps + bowler_stumps):
        notes.append("No high-confidence stump detections; using frame-relative estimates.")
    return notes


def _bbox(detection: dict[str, Any]) -> list[float] | None:
    value = detection.get("bbox") or detection.get("box") or detection.get("xyxy")
    return _bbox_from_any(value)


def _bbox_from_any(value: Any) -> list[float] | None:
    try:
        if value is None or len(value) < 4:
            return None
        box = [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    if not all(isfinite(item) for item in box):
        return None
    x1, y1, x2, y2 = box
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def _bbox_from_polygon(polygon: Any) -> list[float] | None:
    if not polygon:
        return None
    try:
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
    except (TypeError, ValueError, IndexError):
        return None
    return [round(min(xs), 2), round(min(ys), 2), round(max(xs), 2), round(max(ys), 2)]


def _clean_polygon(polygon: Any, width: int, height: int) -> list[list[float]]:
    cleaned: list[list[float]] = []
    for point in polygon or []:
        try:
            x = min(max(float(point[0]), 0.0), float(width))
            y = min(max(float(point[1]), 0.0), float(height))
        except (TypeError, ValueError, IndexError):
            continue
        cleaned.append([round(x, 2), round(y, 2)])
    return cleaned


def _confidence(detection: dict[str, Any]) -> float:
    try:
        return min(max(float(detection.get("confidence", 0.0)), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.0


def _dimension(value: Any, fallback: int) -> int:
    try:
        value = int(value)
        return value if value > 0 else fallback
    except (TypeError, ValueError):
        return fallback


def _positive_float(value: Any, default: float = 8.0) -> float:
    try:
        number = float(value)
        return number if number > 0 else default
    except (TypeError, ValueError):
        return default
