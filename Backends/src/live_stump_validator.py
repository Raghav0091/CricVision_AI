"""Live stump validation for alignment boxes — pure geometry, no models."""

from __future__ import annotations

from math import isfinite
from typing import Any

STUMP_LABELS = frozenset({"stump", "stumps", "wicket", "wickets"})
REJECTED_LABELS = frozenset(
    {
        "ball",
        "bat",
        "person",
        "face",
        "head",
        "player",
        "background",
        "pitch",
        "field",
    }
)
STRONG_CONFIDENCE = 0.45
WEAK_CONFIDENCE = 0.25


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if isfinite(result) else default


def _normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _is_stump_label(label: str | None) -> bool:
    if not label:
        return False
    if label in REJECTED_LABELS:
        return False
    if label in STUMP_LABELS:
        return True
    return any(token in label for token in ("stump", "wicket"))


def _parse_bbox(raw: Any) -> dict[str, float] | None:
    if isinstance(raw, dict):
        if "bbox" in raw and isinstance(raw["bbox"], dict):
            return _parse_bbox(raw["bbox"])
        values = [
            _safe_float(raw.get("x1")),
            _safe_float(raw.get("y1")),
            _safe_float(raw.get("x2")),
            _safe_float(raw.get("y2")),
        ]
        if None not in values:
            x1, y1, x2, y2 = values
        else:
            xyxy = raw.get("xyxy") or raw.get("box")
            if isinstance(xyxy, (list, tuple)) and len(xyxy) >= 4:
                values = [_safe_float(v) for v in xyxy[:4]]
                if None in values:
                    return None
                x1, y1, x2, y2 = values
            else:
                return None
    elif isinstance(raw, (list, tuple)) and len(raw) >= 4:
        values = [_safe_float(v) for v in raw[:4]]
        if None in values:
            return None
        x1, y1, x2, y2 = values
    else:
        return None

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
        return None
    return {
        "x1": round(x1, 3),
        "y1": round(y1, 3),
        "x2": round(x2, 3),
        "y2": round(y2, 3),
    }


def _bbox_center(bbox: dict[str, float]) -> dict[str, float]:
    return {
        "x": round((bbox["x1"] + bbox["x2"]) / 2.0, 3),
        "y": round((bbox["y1"] + bbox["y2"]) / 2.0, 3),
    }


def _extract_label(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None
    for key in ("class_name", "label", "name", "class"):
        label = _normalize_label(item.get(key))
        if label:
            return label
    cls_value = item.get("cls")
    if cls_value is None:
        cls_value = item.get("class_id")
    if isinstance(cls_value, str):
        return _normalize_label(cls_value)
    # ponytail: numeric cls without mapping is skipped — no model names here.
    return None


def normalize_stump_detections(raw_detections: Any) -> list[dict[str, Any]]:
    """Normalize raw detector outputs to stump-only detection dicts."""
    if raw_detections is None:
        return []
    if isinstance(raw_detections, dict):
        if "stump_detections" in raw_detections:
            raw_detections = raw_detections.get("stump_detections")
        elif "detections" in raw_detections:
            raw_detections = raw_detections.get("detections")
        else:
            raw_detections = [raw_detections]
    if not isinstance(raw_detections, (list, tuple)):
        return []

    normalized: list[dict[str, Any]] = []
    for item in raw_detections:
        if item is None:
            continue
        label = _extract_label(item)
        if not _is_stump_label(label):
            continue

        if isinstance(item, dict):
            bbox = _parse_bbox(item.get("bbox") or item.get("box") or item)
        else:
            bbox = _parse_bbox(item)
        if bbox is None:
            continue

        confidence = None
        if isinstance(item, dict):
            confidence = _safe_float(item.get("confidence") or item.get("conf"))
        center = None
        if isinstance(item, dict) and isinstance(item.get("center"), (list, tuple)) and len(item["center"]) >= 2:
            cx = _safe_float(item["center"][0])
            cy = _safe_float(item["center"][1])
            if cx is not None and cy is not None:
                center = {"x": round(cx, 3), "y": round(cy, 3)}
        elif isinstance(item, dict) and isinstance(item.get("center"), dict):
            cx = _safe_float(item["center"].get("x"))
            cy = _safe_float(item["center"].get("y"))
            if cx is not None and cy is not None:
                center = {"x": round(cx, 3), "y": round(cy, 3)}
        if center is None:
            center = _bbox_center(bbox)

        entry: dict[str, Any] = {
            "class_name": label or "stump",
            "confidence": confidence,
            "bbox": bbox,
            "center": center,
        }
        if isinstance(item, dict) and item.get("source"):
            entry["source"] = item.get("source")
        normalized.append(entry)
    return normalized


def _box_area(box: dict[str, float]) -> float:
    return max(0.0, box["x2"] - box["x1"]) * max(0.0, box["y2"] - box["y1"])


def _intersection_area(a: dict[str, float], b: dict[str, float]) -> float:
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def detection_inside_box(
    detection: Any,
    box: Any,
    min_overlap: float = 0.25,
) -> dict[str, Any]:
    """Check whether a detection overlaps an alignment box enough to count."""
    notes: list[str] = []
    det_bbox = None
    if isinstance(detection, dict):
        det_bbox = _parse_bbox(detection.get("bbox") or detection.get("box") or detection)
    else:
        det_bbox = _parse_bbox(detection)
    align_box = _parse_bbox(box)
    if det_bbox is None:
        notes.append("Detection bbox missing or invalid.")
        return {"inside": False, "overlap_ratio": 0.0, "notes": notes}
    if align_box is None:
        notes.append("Alignment box missing or invalid.")
        return {"inside": False, "overlap_ratio": 0.0, "notes": notes}

    det_area = _box_area(det_bbox)
    if det_area <= 0:
        notes.append("Detection bbox has zero area.")
        return {"inside": False, "overlap_ratio": 0.0, "notes": notes}

    overlap = _intersection_area(det_bbox, align_box) / det_area
    inside = overlap >= float(min_overlap)
    if inside:
        notes.append(f"Detection overlaps box ({overlap:.2f} >= {min_overlap:.2f}).")
    else:
        notes.append(f"Detection overlap too low ({overlap:.2f} < {min_overlap:.2f}).")
    return {"inside": inside, "overlap_ratio": round(overlap, 4), "notes": notes}


def _side_result() -> dict[str, Any]:
    return {
        "found": False,
        "detections": [],
        "count": 0,
        "confidence": None,
        "notes": [],
    }


def _assess_side(
    detections: list[dict[str, Any]],
    box: Any,
    min_overlap: float = 0.25,
) -> dict[str, Any]:
    result = _side_result()
    align_box = _parse_bbox(box)
    if align_box is None:
        result["notes"].append("Alignment box invalid.")
        return result

    matched: list[dict[str, Any]] = []
    confidences: list[float] = []
    for detection in detections:
        check = detection_inside_box(detection, align_box, min_overlap=min_overlap)
        if check.get("inside"):
            matched.append(detection)
            conf = _safe_float(detection.get("confidence"))
            if conf is not None:
                confidences.append(conf)

    result["detections"] = matched
    result["count"] = len(matched)
    result["found"] = result["count"] > 0
    if confidences:
        result["confidence"] = round(max(confidences), 4)
    if result["found"]:
        result["notes"].append(f"{result['count']} stump detection(s) inside box.")
    else:
        result["notes"].append("No stump detections inside box.")
    return result


def _quality_from_sides(striker: dict[str, Any], non_striker: dict[str, Any]) -> str:
    striker_found = bool(striker.get("found"))
    non_found = bool(non_striker.get("found"))
    if not striker_found and not non_found:
        return "Unavailable"
    if not striker_found or not non_found:
        return "Poor"

    striker_conf = _safe_float(striker.get("confidence")) or 0.0
    non_conf = _safe_float(non_striker.get("confidence")) or 0.0
    if striker_conf >= STRONG_CONFIDENCE and non_conf >= STRONG_CONFIDENCE:
        return "Good"
    if striker_conf >= WEAK_CONFIDENCE and non_conf >= WEAK_CONFIDENCE:
        return "Partial"
    return "Partial"


def validate_stumps_in_alignment_boxes(
    raw_detections: Any,
    box_layout: Any,
    frame_size: Any = None,
) -> dict[str, Any]:
    """Validate stump detections inside striker / non-striker alignment boxes."""
    notes: list[str] = [
        "Estimated single-camera geometry only — not official LBW/DRS.",
    ]
    if not isinstance(box_layout, dict) or not box_layout.get("available"):
        notes.append("Alignment box layout unavailable.")
        return {
            "valid": False,
            "quality": "Unavailable",
            "striker": _side_result(),
            "non_striker": _side_result(),
            "notes": notes,
        }

    stump_detections = normalize_stump_detections(raw_detections)
    if not stump_detections:
        notes.append("No stump-class detections in frame.")
    striker = _assess_side(
        stump_detections,
        box_layout.get("striker_stumps_box"),
    )
    non_striker = _assess_side(
        stump_detections,
        box_layout.get("non_striker_stumps_box"),
    )
    quality = _quality_from_sides(striker, non_striker)
    valid = bool(striker.get("found") and non_striker.get("found"))
    if valid:
        notes.append("Stump evidence found in both alignment boxes.")
    elif striker.get("found") or non_striker.get("found"):
        notes.append("Stump evidence found in only one alignment box.")
    else:
        notes.append("No stump evidence inside alignment boxes.")

    return {
        "valid": valid,
        "quality": quality,
        "striker": striker,
        "non_striker": non_striker,
        "notes": notes,
    }


def build_environment_context_from_stumps(
    validation_result: Any,
    box_layout: Any,
    frame_size: Any = None,
) -> dict[str, Any]:
    """Compact environment context from validated stump alignment."""
    notes: list[str] = []
    validation = validation_result if isinstance(validation_result, dict) else {}
    layout = box_layout if isinstance(box_layout, dict) else {}

    striker_found = bool((validation.get("striker") or {}).get("found"))
    non_found = bool((validation.get("non_striker") or {}).get("found"))
    stumps_validated = bool(validation.get("valid"))
    quality = validation.get("quality", "Unavailable")

    camera_view = "Unknown"
    if not stumps_validated:
        camera_view = "Poor" if striker_found or non_found else "Unknown"
    elif quality == "Good":
        camera_view = "Good"
    elif quality == "Partial":
        camera_view = "Usable"
    else:
        camera_view = "Poor"

    stump_line = None
    pitch_corridor = None
    striker_box = _parse_bbox(layout.get("striker_stumps_box"))
    non_box = _parse_bbox(layout.get("non_striker_stumps_box"))
    if stumps_validated and striker_box and non_box:
        non_c = _bbox_center(non_box)
        striker_c = _bbox_center(striker_box)
        stump_line = {"start": non_c, "end": striker_c}
        notes.append("Estimated stump line from validated alignment boxes.")

    return {
        "available": stumps_validated,
        "camera_view": camera_view,
        "stumps_validated": stumps_validated,
        "striker_stumps_found": striker_found,
        "non_striker_stumps_found": non_found,
        "stump_line": stump_line,
        "pitch_corridor": pitch_corridor,
        "notes": notes,
    }
