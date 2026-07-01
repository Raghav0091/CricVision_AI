"""Build and normalize JSON-safe practice-environment calibration context."""

from __future__ import annotations

from copy import deepcopy
from math import isfinite


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
HANDEDNESS_ALIASES = {
    "right": "right",
    "right-handed": "right",
    "right handed": "right",
    "left": "left",
    "left-handed": "left",
    "left handed": "left",
    "unknown": "unknown",
}


def _disabled_calibration_template() -> dict:
    """Return the raw disabled template used before quality finalization."""
    return {
        "enabled": False,
        "confirmed": False,
        "auto_estimate": True,
        "camera_view": "unknown",
        "batter_handedness": "unknown",
        "calibration_quality": "Low",
        "calibration_score": 0.0,
        "frame_width": None,
        "frame_height": None,
        "stumps": {
            "batter_end": {
                "bbox": None,
                "center": None,
                "confidence": 0.0,
                "source": "missing",
                "status": "missing",
            }
        },
        "crease_line": {
            "y": None,
            "source": "missing",
            "status": "missing",
        },
        "pitch_corridor": {
            "polygon": [],
            "bbox": None,
            "source": "missing",
            "status": "missing",
            "confidence": 0.0,
        },
        "pitch_ends": {
            "batter_end_y": None,
            "bowler_end_y": None,
            "source": "missing",
        },
        "line_reference": {
            "off_stump_x": None,
            "middle_stump_x": None,
            "leg_stump_x": None,
            "source": "missing",
            "confidence": 0.0,
        },
        "notes": [],
        "calibration_version": 1,
    }


def default_calibration_context() -> dict:
    """Return a disabled, JSON-safe calibration context."""
    return finalize_calibration_quality(deepcopy(_disabled_calibration_template()))


def normalize_calibration_context(context) -> dict:
    """Normalize missing, old, or partial context without raising."""
    default = _disabled_calibration_template()
    if not isinstance(context, dict):
        return default_calibration_context()

    result = deepcopy(default)
    result["enabled"] = bool(context.get("enabled", False))
    result["confirmed"] = bool(context.get("confirmed", result["enabled"]))
    result["auto_estimate"] = bool(context.get("auto_estimate", True))
    result["camera_view"] = _camera_view(context.get("camera_view"))
    result["batter_handedness"] = _handedness(
        context.get("batter_handedness")
    )
    result["frame_width"] = _positive_int(context.get("frame_width"))
    result["frame_height"] = _positive_int(context.get("frame_height"))

    stump_data = context.get("stumps")
    if isinstance(stump_data, dict):
        stump_data = stump_data.get("batter_end", stump_data)
    result["stumps"]["batter_end"] = _normalize_stump(stump_data)

    crease = context.get("crease_line")
    if isinstance(crease, dict):
        result["crease_line"] = {
            "y": _number_or_none(crease.get("y")),
            "source": _source(crease.get("source")),
            "status": _status(crease.get("status"), crease.get("y")),
        }

    corridor = context.get("pitch_corridor")
    if isinstance(corridor, dict):
        polygon = corridor.get("polygon")
        result["pitch_corridor"] = {
            "polygon": _polygon(polygon),
            "bbox": _box(corridor.get("bbox")),
            "source": _source(corridor.get("source")),
            "status": _status(
                corridor.get("status"),
                corridor.get("bbox") or polygon,
            ),
            "confidence": _score(corridor.get("confidence")),
        }

    pitch_ends = context.get("pitch_ends")
    if isinstance(pitch_ends, dict):
        result["pitch_ends"] = {
            "batter_end_y": _number_or_none(pitch_ends.get("batter_end_y")),
            "bowler_end_y": _number_or_none(pitch_ends.get("bowler_end_y")),
            "source": _source(pitch_ends.get("source")),
        }

    line_reference = context.get("line_reference")
    if isinstance(line_reference, dict):
        result["line_reference"] = {
            "off_stump_x": _number_or_none(
                line_reference.get("off_stump_x")
            ),
            "middle_stump_x": _number_or_none(
                line_reference.get("middle_stump_x")
            ),
            "leg_stump_x": _number_or_none(
                line_reference.get("leg_stump_x")
            ),
            "source": _source(line_reference.get("source")),
            "confidence": _score(line_reference.get("confidence")),
        }

    raw_score = context.get("calibration_score")
    if raw_score is None:
        raw_score = {
            "low": 0.1,
            "medium": 0.5,
            "good": 0.75,
            "high": 0.9,
        }.get(str(context.get("calibration_quality") or "").lower(), 0.0)
    score = _score(raw_score)
    result["calibration_score"] = score
    result["calibration_quality"] = calibration_quality_label(score)
    notes = context.get("notes") or []
    if isinstance(notes, str):
        notes = [notes]
    if isinstance(notes, (list, tuple)):
        result["notes"] = [
            str(note) for note in notes if str(note).strip()
        ]
    return finalize_calibration_quality(result)


def build_calibration_context(
    context=None,
    *,
    frame_detections=None,
    first_frame_detections=None,
    frame_width=None,
    frame_height=None,
) -> dict:
    """Build a context from user choices and already-produced detections."""
    result = normalize_calibration_context(context)
    if frame_width is not None:
        result["frame_width"] = _positive_int(frame_width)
    if frame_height is not None:
        result["frame_height"] = _positive_int(frame_height)
    if not result["enabled"]:
        return result

    from Backends.src.calibration.pitch_calibration import (
        estimate_line_reference,
        estimate_pitch_corridor,
    )
    from Backends.src.calibration.stump_calibration import (
        estimate_stump_reference,
    )

    use_detections = result.get("auto_estimate", True)
    stump_result = estimate_stump_reference(
        frame_detections=frame_detections if use_detections else None,
        first_frame_detections=(
            first_frame_detections if use_detections else None
        ),
        frame_width=result.get("frame_width"),
        frame_height=result.get("frame_height"),
    )
    stump_reference = stump_result["stump_reference"]
    width = result.get("frame_width") or stump_result["frame_width"]
    height = result.get("frame_height") or stump_result["frame_height"]
    result["frame_width"] = width
    result["frame_height"] = height
    result["stumps"]["batter_end"] = stump_reference

    corridor = estimate_pitch_corridor(
        stump_reference,
        width,
        height,
        camera_view=result["camera_view"],
    )
    result["pitch_corridor"] = corridor["pitch_corridor"]
    result["pitch_ends"] = corridor["pitch_ends"]
    result["crease_line"] = corridor["crease_line"]
    result["line_reference"] = estimate_line_reference(
        stump_reference,
        batter_handedness=result["batter_handedness"],
    )

    score = _calibration_score(result)
    result["calibration_score"] = score
    result["calibration_quality"] = calibration_quality_label(score)
    notes = [
        *result.get("notes", []),
        *stump_result.get("notes", []),
        *corridor.get("notes", []),
    ]
    if result["calibration_quality"] in {"Low", "Medium"}:
        notes.append(
            "Practice-environment geometry is approximate and should be "
            "treated as estimated context."
        )
    result["notes"] = list(dict.fromkeys(note for note in notes if note))
    return finalize_calibration_quality(normalize_calibration_context(result))


def validate_calibration_context(context) -> list[str]:
    """Return validation messages; an empty list means the context is usable."""
    result = normalize_calibration_context(context)
    issues = []
    if not result["enabled"]:
        return issues
    if not result["frame_width"] or not result["frame_height"]:
        issues.append("Frame dimensions are missing.")
    if result["camera_view"] == "unknown":
        issues.append("Camera view is unknown.")
    if result["stumps"]["batter_end"]["center"] is None:
        issues.append("Batter-end stump reference is missing.")
    if not result["pitch_corridor"]["polygon"]:
        issues.append("Pitch corridor is missing.")
    return issues


def calibration_quality_label(score) -> str:
    """Convert a numeric score into Low, Medium, Good, or High."""
    value = _score(score)
    if value >= 0.85:
        return "High"
    if value >= 0.7:
        return "Good"
    if value >= 0.4:
        return "Medium"
    return "Low"


def finalize_calibration_quality(context) -> dict:
    """Apply honest quality caps so labels match stump/corridor sources."""
    result = context
    if not result["enabled"]:
        result["calibration_quality"] = "Disabled"
        result["calibration_score"] = 0.0
        if not result["notes"]:
            result["notes"] = [
                "Practice environment calibration was disabled for this analysis."
            ]
        return result

    stump = result["stumps"]["batter_end"]
    corridor = result["pitch_corridor"] or {}
    stump_source = str(stump.get("source") or "missing").lower()
    stump_status = str(stump.get("status") or "missing").lower()
    stump_conf = _score(stump.get("confidence"))
    corridor_status = str(corridor.get("status") or "missing").lower()
    corridor_source = str(corridor.get("source") or "missing").lower()

    score = _calibration_score(result)
    result["calibration_score"] = score
    quality = calibration_quality_label(score)

    if stump_source in {"estimated", "missing"} or stump_status == "estimated":
        quality = _cap_quality(quality, "Medium")
        if stump_conf < 0.35:
            quality = "Low"
    elif stump_conf < 0.65:
        quality = _cap_quality(quality, "Good")

    if corridor_status == "estimated" or corridor_source == "estimated":
        quality = _cap_quality(quality, "Medium")

    if quality == "High":
        if not (
            stump_source == "auto"
            and stump_status == "detected"
            and stump_conf >= 0.65
        ):
            quality = "Good"

    result["calibration_quality"] = quality
    result["notes"] = _harmonize_calibration_notes(result, quality)
    return result


def _cap_quality(current: str, maximum: str) -> str:
    order = {"Low": 0, "Medium": 1, "Good": 2, "High": 3, "Disabled": -1}
    current_rank = order.get(current, 0)
    max_rank = order.get(maximum, 2)
    if current_rank <= max_rank:
        return current
    for label, rank in order.items():
        if rank == max_rank:
            return label
    return maximum


def _harmonize_calibration_notes(context, quality: str) -> list[str]:
    stump = context["stumps"]["batter_end"]
    notes = [str(note).strip() for note in (context.get("notes") or []) if str(note).strip()]
    stump_source = str(stump.get("source") or "missing").lower()
    stump_status = str(stump.get("status") or "missing").lower()

    filtered = []
    for note in notes:
        lowered = note.lower()
        if stump_source == "estimated" and "detected (auto)" in lowered:
            continue
        if stump_status == "estimated" and "estimated from existing detections" in lowered:
            continue
        filtered.append(note)

    if stump_source == "estimated" or stump_status == "estimated":
        filtered.append(
            "No usable stump detection was available; approximate frame-based geometry was used."
        )
    elif stump_source == "auto" and stump_status == "detected":
        confidence = _score(stump.get("confidence"))
        filtered.append(
            f"Batter-end stumps detected from model output ({confidence * 100:.0f}% confidence)."
        )

    if quality in {"Low", "Medium"}:
        filtered.append(
            "Practice-environment geometry is approximate and should be treated as estimated context."
        )
    return list(dict.fromkeys(filtered))


def _calibration_score(context) -> float:
    if not context.get("enabled"):
        return 0.0
    stump = context["stumps"]["batter_end"]
    stump_source = str(stump.get("source") or "missing").lower()
    stump_status = str(stump.get("status") or "missing").lower()
    stump_conf = _score(stump.get("confidence"))

    if stump_source in {"estimated", "missing"} or stump_status == "estimated":
        score = min(stump_conf, 0.35) * 0.45
    else:
        score = stump_conf * 0.55
    if context.get("frame_width") and context.get("frame_height"):
        score += 0.15
    if context.get("camera_view") != "unknown":
        score += 0.15
    if context.get("batter_handedness") != "unknown":
        score += 0.1
    if context.get("confirmed"):
        score += 0.05
    return round(min(score, 1.0), 3)


def _normalize_stump(stump) -> dict:
    if not isinstance(stump, dict):
        stump = {}
    bbox = _box(stump.get("bbox") or stump.get("box") or stump.get("xyxy"))
    center = _point(stump.get("center"))
    if center is None and bbox is not None:
        center = [
            round((bbox[0] + bbox[2]) / 2, 3),
            round((bbox[1] + bbox[3]) / 2, 3),
        ]
    source = _source(stump.get("source"))
    return {
        "bbox": bbox,
        "center": center,
        "confidence": _score(stump.get("confidence")),
        "source": source,
        "status": _status(stump.get("status"), center),
    }


def _camera_view(value) -> str:
    key = str(value or "unknown").strip().lower().replace("-", " ")
    return CAMERA_VIEW_ALIASES.get(key, "unknown")


def _handedness(value) -> str:
    key = str(value or "unknown").strip().lower().replace("_", " ")
    return HANDEDNESS_ALIASES.get(key, "unknown")


def _source(value) -> str:
    source = str(value or "missing").strip().lower()
    return source if source in {"auto", "manual", "estimated", "missing"} else "estimated"


def _status(value, fallback) -> str:
    status = str(value or "").strip().lower()
    if status in {"detected", "estimated", "manual", "missing"}:
        return status
    return "estimated" if fallback is not None else "missing"


def _positive_int(value):
    try:
        value = int(value)
        return value if value > 0 else None
    except (TypeError, ValueError):
        return None


def _number_or_none(value):
    try:
        value = float(value)
        return round(value, 3) if isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _score(value) -> float:
    numeric = _number_or_none(value)
    if numeric is None:
        return 0.0
    return round(min(max(numeric, 0.0), 1.0), 3)


def _point(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    point = [_number_or_none(value[0]), _number_or_none(value[1])]
    return point if None not in point else None


def _box(value):
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    box = [_number_or_none(item) for item in value[:4]]
    if None in box:
        return None
    x1, y1, x2, y2 = box
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _polygon(value):
    if not isinstance(value, (list, tuple)):
        return []
    points = [_point(item) for item in value]
    return points if len(points) >= 4 and all(points) else []
