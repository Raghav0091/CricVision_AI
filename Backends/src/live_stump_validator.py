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
DEFAULT_MIN_OVERLAP = 0.2


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


def _extract_label(item: Any, class_names: dict[int, str] | None = None) -> str | None:
    if not isinstance(item, dict):
        return None
    for key in ("class_name", "label", "name", "class"):
        label = _normalize_label(item.get(key))
        if label and not label.isdigit():
            return label
    cls_value = item.get("cls")
    if cls_value is None:
        cls_value = item.get("class_id")
    if isinstance(cls_value, str):
        mapped = _normalize_label(cls_value)
        if mapped and not mapped.isdigit():
            return mapped
    if class_names and cls_value is not None:
        try:
            class_id = int(cls_value)
            mapped = class_names.get(class_id)
            if mapped:
                return _normalize_label(mapped)
        except (TypeError, ValueError):
            pass
    return None


def normalize_stump_detections(
    raw_detections: Any,
    class_names: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
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
        label = _extract_label(item, class_names=class_names)
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


def _center_inside_box(center: dict[str, float], align_box: dict[str, float]) -> bool:
    cx = _safe_float(center.get("x"))
    cy = _safe_float(center.get("y"))
    if cx is None or cy is None:
        return False
    return (
        align_box["x1"] <= cx <= align_box["x2"]
        and align_box["y1"] <= cy <= align_box["y2"]
    )


def detection_inside_box(
    detection: Any,
    box: Any,
    min_center_inside: bool = True,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
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
        return {
            "inside": False,
            "overlap_ratio": 0.0,
            "center_inside": False,
            "notes": notes,
        }
    if align_box is None:
        notes.append("Alignment box missing or invalid.")
        return {
            "inside": False,
            "overlap_ratio": 0.0,
            "center_inside": False,
            "notes": notes,
        }

    det_area = _box_area(det_bbox)
    if det_area <= 0:
        notes.append("Detection bbox has zero area.")
        return {
            "inside": False,
            "overlap_ratio": 0.0,
            "center_inside": False,
            "notes": notes,
        }

    overlap = _intersection_area(det_bbox, align_box) / det_area
    center = _bbox_center(det_bbox)
    if isinstance(detection, dict) and isinstance(detection.get("center"), dict):
        cx = _safe_float(detection["center"].get("x"))
        cy = _safe_float(detection["center"].get("y"))
        if cx is not None and cy is not None:
            center = {"x": cx, "y": cy}
    center_inside = _center_inside_box(center, align_box)

    inside = overlap >= float(min_overlap)
    if min_center_inside:
        inside = inside and center_inside

    if inside:
        notes.append(f"Detection inside box (overlap {overlap:.2f}, center_inside={center_inside}).")
    else:
        notes.append(
            f"Detection not inside box (overlap {overlap:.2f} < {min_overlap:.2f}"
            + (", center outside" if min_center_inside and not center_inside else "")
            + ")."
        )
    return {
        "inside": inside,
        "overlap_ratio": round(overlap, 4),
        "center_inside": center_inside,
        "notes": notes,
    }


def _side_quality(found: bool, best_conf: float | None) -> str:
    if not found:
        return "Not Found"
    conf = best_conf or 0.0
    if conf >= STRONG_CONFIDENCE:
        return "Strong"
    if conf >= WEAK_CONFIDENCE:
        return "Found"
    return "Weak"


def _side_result() -> dict[str, Any]:
    return {
        "found": False,
        "quality": "Not Found",
        "detections": [],
        "count": 0,
        "best_confidence": None,
        "confidence": None,  # backward compat alias
        "notes": [],
    }


def validate_stump_box(
    raw_detections: Any,
    box: Any,
    side_name: str,
    frame_size: Any = None,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
) -> dict[str, Any]:
    """Validate stump detections inside a single alignment box."""
    _ = frame_size  # reserved for future frame-relative thresholds
    result = _side_result()
    align_box = _parse_bbox(box)
    if align_box is None:
        result["notes"].append(f"{side_name}: alignment box invalid.")
        return result

    stump_detections = normalize_stump_detections(raw_detections)
    matched: list[dict[str, Any]] = []
    confidences: list[float] = []
    for detection in stump_detections:
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
        best = round(max(confidences), 4)
        result["best_confidence"] = best
        result["confidence"] = best
    result["quality"] = _side_quality(result["found"], result.get("best_confidence"))
    if result["found"]:
        result["notes"].append(
            f"{side_name}: {result['count']} stump detection(s) — {result['quality']}."
        )
    else:
        result["notes"].append(f"{side_name}: no stump detections inside box.")
    return result


def _assess_side(
    detections: list[dict[str, Any]],
    box: Any,
    side_name: str = "Side",
    min_overlap: float = DEFAULT_MIN_OVERLAP,
) -> dict[str, Any]:
    """Assess one alignment box from pre-normalized stump detections."""
    result = _side_result()
    align_box = _parse_bbox(box)
    if align_box is None:
        result["notes"].append(f"{side_name}: alignment box invalid.")
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
        best = round(max(confidences), 4)
        result["best_confidence"] = best
        result["confidence"] = best
    result["quality"] = _side_quality(result["found"], result.get("best_confidence"))
    if result["found"]:
        result["notes"].append(f"{result['count']} stump detection(s) inside {side_name} box.")
    else:
        result["notes"].append(f"No stump detections inside {side_name} box.")
    return result


def _overall_quality(striker: dict[str, Any], non_striker: dict[str, Any]) -> str:
    striker_q = striker.get("quality") or "Not Found"
    non_q = non_striker.get("quality") or "Not Found"
    if striker_q == "Not Found" and non_q == "Not Found":
        return "Not Found"
    if striker_q == "Not Found" or non_q == "Not Found":
        return "Weak"
    order = {"Weak": 0, "Found": 1, "Strong": 2}
    combined = min(order.get(striker_q, 0), order.get(non_q, 0))
    if combined == 2:
        return "Strong"
    if combined == 1:
        return "Found"
    return "Weak"


def _legacy_quality_label(quality: str) -> str:
    """Map new quality labels for callers expecting Good/Partial/Poor."""
    return {
        "Strong": "Good",
        "Found": "Partial",
        "Weak": "Poor",
        "Not Found": "Unavailable",
    }.get(quality, quality)


def validate_stumps_in_alignment_boxes(
    raw_detections: Any,
    box_layout: Any,
    frame_size: Any = None,
) -> dict[str, Any]:
    """Validate stump detections inside striker / non-striker alignment boxes."""
    _ = frame_size
    notes: list[str] = [
        "Estimated single-camera geometry only — not official LBW/DRS.",
    ]
    if not isinstance(box_layout, dict) or not box_layout.get("available"):
        notes.append("Alignment box layout unavailable.")
        return {
            "valid": False,
            "quality": "Not Found",
            "legacy_quality": "Unavailable",
            "striker": _side_result(),
            "non_striker": _side_result(),
            "stable": False,
            "notes": notes,
        }

    stump_detections = normalize_stump_detections(raw_detections)
    if not stump_detections:
        notes.append("No stump-class detections in frame.")

    striker = validate_stump_box(
        stump_detections,
        box_layout.get("striker_stumps_box"),
        "Striker",
    )
    non_striker = validate_stump_box(
        stump_detections,
        box_layout.get("non_striker_stumps_box"),
        "Non-Striker",
    )
    quality = _overall_quality(striker, non_striker)
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
        "legacy_quality": _legacy_quality_label(quality),
        "striker": striker,
        "non_striker": non_striker,
        "stable": False,
        "notes": notes,
    }


def update_stump_validation_history(
    current_validation: Any,
    history: list[dict[str, Any]] | None = None,
    required_stable_frames: int = 5,
) -> dict[str, Any]:
    """Track recent frames; ready only after both stump sets stay found."""
    notes: list[str] = []
    validation = current_validation if isinstance(current_validation, dict) else {}
    history = list(history or [])

    entry = {
        "valid": bool(validation.get("valid")),
        "striker_found": bool((validation.get("striker") or {}).get("found")),
        "non_striker_found": bool((validation.get("non_striker") or {}).get("found")),
        "quality": validation.get("quality"),
    }
    history.append(entry)
    max_len = max(required_stable_frames * 2, 10)
    if len(history) > max_len:
        history = history[-max_len:]

    stable_count = 0
    for past in reversed(history):
        if past.get("valid") and past.get("striker_found") and past.get("non_striker_found"):
            stable_count += 1
        else:
            break

    stable = stable_count >= required_stable_frames
    ready = stable
    if ready:
        notes.append(
            f"Both stump sets stable for {stable_count} recent frame(s) "
            f"(need {required_stable_frames})."
        )
    elif validation.get("valid"):
        notes.append(
            f"Both found but only {stable_count}/{required_stable_frames} stable frames."
        )
    else:
        notes.append("Waiting for stump evidence in both boxes.")

    return {
        "stable": stable,
        "ready": ready,
        "history": history,
        "stable_count": stable_count,
        "notes": notes,
    }


def build_environment_context_from_stumps(
    validation_result: Any,
    box_layout: Any,
    frame_size: Any = None,
) -> dict[str, Any]:
    """Compact environment context from validated stump alignment (legacy alias)."""
    from Backends.src.session_calibration import build_environment_context_from_validated_stumps

    return build_environment_context_from_validated_stumps(
        box_layout,
        validation_result,
        frame_size=frame_size,
    )
