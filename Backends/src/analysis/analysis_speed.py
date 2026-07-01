"""Analysis speed/quality presets for uploaded video processing."""

from __future__ import annotations

import copy
from typing import Any

from Backends.src.analysis.smart_pipeline import get_smart_analysis_settings


def get_analysis_mode_settings(mode: str) -> dict[str, Any]:
    """Return smart pipeline settings (legacy wrapper)."""
    return get_smart_analysis_settings(mode)


def resolve_frame_limit(enabled: bool, choice) -> int | None:
    """Convert UI frame-limit selection to an integer cap or None."""
    if not enabled:
        return None
    if choice in {None, "", "All frames"}:
        return None
    try:
        return max(1, int(choice))
    except (TypeError, ValueError):
        return None


def resize_frame_for_inference(frame, target_width: int | None):
    """Resize a frame for inference and return the frame plus inverse scale."""
    if target_width is None:
        return frame, 1.0

    height, width = frame.shape[:2]
    if width <= target_width:
        return frame, 1.0

    scale = target_width / float(width)
    new_height = max(1, int(height * scale))
    from Backends.src.utils.cv2_loader import cv2

    resized = cv2.resize(frame, (target_width, new_height))
    return resized, scale


def extract_detection_box(detection):
    """Safely extract bbox from different detection formats."""
    if detection is None:
        return None, None

    if isinstance(detection, (list, tuple)) and len(detection) >= 4:
        try:
            box = [float(detection[0]), float(detection[1]), float(detection[2]), float(detection[3])]
            return "box", _valid_box(box)
        except (TypeError, ValueError):
            return None, None

    if not isinstance(detection, dict):
        return None, None

    for key in ("box", "bbox", "xyxy", "coordinates"):
        value = detection.get(key)
        if value is not None:
            try:
                if len(value) < 4:
                    continue
                box = [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
                valid = _valid_box(box)
                if valid is not None:
                    return key, valid
            except (TypeError, ValueError):
                continue

    if all(key in detection for key in ("x1", "y1", "x2", "y2")):
        try:
            box = [
                float(detection["x1"]),
                float(detection["y1"]),
                float(detection["x2"]),
                float(detection["y2"]),
            ]
            valid = _valid_box(box)
            if valid is not None:
                return "xy_fields", valid
        except (TypeError, ValueError):
            return None, None

    return None, None


def normalize_detections(detections, stats: dict | None = None):
    """Normalize detections into dict format with a box key without scaling."""
    if not detections:
        return []

    normalized = []
    for detection in detections:
        box_key, box = extract_detection_box(detection)
        if box is None:
            if stats is not None:
                stats["invalid_detection_count"] = stats.get("invalid_detection_count", 0) + 1
            continue

        if isinstance(detection, dict):
            item = copy.copy(detection)
        else:
            item = {
                "confidence": None,
                "class_name": "unknown",
            }

        int_box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
        item["box"] = tuple(int_box)
        if box_key in {"bbox", "xyxy", "coordinates"}:
            item[box_key] = int_box
        elif box_key == "xy_fields":
            item["x1"], item["y1"], item["x2"], item["y2"] = int_box

        center = item.get("center")
        if center is None or len(center) < 2:
            item["center"] = (
                int((int_box[0] + int_box[2]) / 2),
                int((int_box[1] + int_box[3]) / 2),
            )
        else:
            try:
                item["center"] = (int(center[0]), int(center[1]))
            except (TypeError, ValueError, IndexError):
                item["center"] = (
                    int((int_box[0] + int_box[2]) / 2),
                    int((int_box[1] + int_box[3]) / 2),
                )
        normalized.append(item)
    return normalized


def scale_detections_to_original(detections, scale: float, stats: dict | None = None):
    """Scale detections from resized inference frame back to original frame size."""
    normalized = normalize_detections(detections, stats=stats)
    if not normalized:
        return []

    try:
        scale_value = float(scale)
    except (TypeError, ValueError):
        scale_value = 1.0

    if scale_value in {0, 1.0}:
        return normalized

    inverse = 1.0 / scale_value
    scaled = []
    for detection in normalized:
        x1, y1, x2, y2 = detection["box"]
        center_x, center_y = detection.get("center", ((x1 + x2) / 2, (y1 + y2) / 2))
        int_box = (
            int(x1 * inverse),
            int(y1 * inverse),
            int(x2 * inverse),
            int(y2 * inverse),
        )
        int_center = (int(center_x * inverse), int(center_y * inverse))
        item = copy.copy(detection)
        item["box"] = int_box
        item["center"] = int_center
        for key in ("bbox", "xyxy", "coordinates"):
            if key in item:
                item[key] = list(int_box)
        if all(key in item for key in ("x1", "y1", "x2", "y2")):
            item["x1"], item["y1"], item["x2"], item["y2"] = int_box
        scaled.append(item)
    return scaled


def _valid_box(box):
    try:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    except (TypeError, ValueError, IndexError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 == x1 and y2 == y1:
        return None
    return [x1, y1, x2, y2]
