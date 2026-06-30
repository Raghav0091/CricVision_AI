"""Shared helpers for normalizing per-frame detection timelines."""

from __future__ import annotations

from math import hypot
from typing import Any


def normalize_frame_detections(frame_detections) -> list[dict[str, Any]]:
    """Normalize list or dict frame timelines into sorted frame records."""
    if not frame_detections:
        return []

    items = (
        frame_detections.items()
        if isinstance(frame_detections, dict)
        else enumerate(frame_detections)
    )
    normalized = []
    for fallback_index, raw_frame in items:
        if raw_frame is None:
            raw_frame = {}
        if not isinstance(raw_frame, dict):
            continue

        frame_index = raw_frame.get("frame_index", fallback_index)
        try:
            frame_index = int(frame_index)
        except (TypeError, ValueError):
            frame_index = len(normalized)

        normalized.append(
            {
                "frame_index": frame_index,
                "ball_detections": list(
                    raw_frame.get("ball_detections") or raw_frame.get("balls") or []
                ),
                "bat_detections": list(
                    raw_frame.get("bat_detections") or raw_frame.get("bats") or []
                ),
                "stump_detections": list(
                    raw_frame.get("stump_detections")
                    or raw_frame.get("stumps")
                    or []
                ),
                "fielder_detections": list(
                    raw_frame.get("fielder_detections")
                    or raw_frame.get("player_detections")
                    or raw_frame.get("fielders")
                    or raw_frame.get("players")
                    or []
                ),
            }
        )
    return sorted(normalized, key=lambda item: item["frame_index"])


def find_ball_center_at_or_before(frames, impact_frame):
    """Return the highest-confidence ball center at or before impact."""
    try:
        impact_frame = int(impact_frame)
    except (TypeError, ValueError):
        return None

    candidates = [
        item
        for item in frames
        if item.get("frame_index", -1) <= impact_frame and item.get("ball_detections")
    ]
    if not candidates:
        return None
    return best_detection_center(candidates[-1]["ball_detections"])


def best_detection_center(detections):
    """Return the center of the highest-confidence detection."""
    if not detections:
        return None

    def confidence(detection):
        try:
            return float(detection.get("confidence", 0))
        except (TypeError, ValueError, AttributeError):
            return 0

    best = max(detections, key=confidence)
    if not isinstance(best, dict):
        return None

    center = best.get("center")
    if center is not None and len(center) >= 2:
        try:
            return float(center[0]), float(center[1])
        except (TypeError, ValueError, IndexError):
            pass

    bbox = best.get("bbox") or best.get("box") or best.get("xyxy")
    try:
        if bbox is None or len(bbox) < 4:
            return None
        return (float(bbox[0]) + float(bbox[2])) / 2, (float(bbox[1]) + float(bbox[3])) / 2
    except (TypeError, ValueError):
        return None


def calculate_point_distance(point_a, point_b) -> float | None:
    """Euclidean distance between two points."""
    if point_a is None or point_b is None:
        return None
    try:
        return hypot(float(point_a[0]) - float(point_b[0]), float(point_a[1]) - float(point_b[1]))
    except (TypeError, ValueError, IndexError):
        return None
