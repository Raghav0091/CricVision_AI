"""Estimate a stump reference from detections already produced by analysis."""

from __future__ import annotations

from math import isfinite

from Backends.src.analysis.frame_detection_utils import (
    best_detection_center,
    normalize_frame_detections,
)

DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720


def estimate_stump_reference(
    frame_detections=None,
    first_frame_detections=None,
    frame_width=None,
    frame_height=None,
):
    """Return the best detected stump or a conservative estimated reference."""
    width = _dimension(frame_width, DEFAULT_FRAME_WIDTH)
    height = _dimension(frame_height, DEFAULT_FRAME_HEIGHT)
    candidates = _stump_candidates(first_frame_detections)
    for frame in normalize_frame_detections(frame_detections):
        candidates.extend(frame.get("stump_detections") or [])

    valid = [
        detection
        for detection in candidates
        if isinstance(detection, dict)
        and _center(detection) is not None
    ]
    if valid:
        detection = max(valid, key=_confidence)
        center = _center(detection)
        bbox = _bbox(detection)
        if bbox is None:
            bbox = _box_around_center(center, width, height)
        confidence = _confidence(detection)
        stump_reference = {
            "bbox": bbox,
            "center": [round(center[0], 3), round(center[1], 3)],
            "confidence": round(confidence, 3),
            "source": "auto",
            "status": "detected",
        }
        return {
            **stump_reference,
            "stump_reference": stump_reference,
            "frame_width": width,
            "frame_height": height,
            "notes": [
                f"Batter-end stumps detected from model output ({confidence * 100:.0f}% confidence)."
            ],
        }

    center = [round(width * 0.5, 3), round(height * 0.68, 3)]
    stump_reference = {
        "bbox": _box_around_center(center, width, height),
        "center": center,
        "confidence": 0.15,
        "source": "estimated",
        "status": "estimated",
    }
    return {
        **stump_reference,
        "stump_reference": stump_reference,
        "frame_width": width,
        "frame_height": height,
        "notes": [
            "No usable stump detection was available; a low-confidence "
            "frame-relative stump position was estimated."
        ],
    }


def _stump_candidates(value):
    if not value:
        return []
    if isinstance(value, dict):
        if any(
            key in value
            for key in ("stump_detections", "stumps")
        ):
            return list(
                value.get("stump_detections")
                or value.get("stumps")
                or []
            )
        if any(key in value for key in ("bbox", "box", "xyxy", "center")):
            return [value]
        frames = normalize_frame_detections(value)
        return [
            detection
            for frame in frames
            for detection in frame.get("stump_detections") or []
        ]
    if isinstance(value, (list, tuple)):
        if value and all(
            isinstance(item, dict)
            and any(
                key in item
                for key in ("frame_index", "stump_detections", "stumps")
            )
            for item in value
        ):
            return [
                detection
                for frame in normalize_frame_detections(value)
                for detection in frame.get("stump_detections") or []
            ]
        return list(value)
    return []


def _bbox(detection):
    value = (
        detection.get("bbox")
        or detection.get("box")
        or detection.get("xyxy")
    )
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
    return [
        round(x1, 3),
        round(y1, 3),
        round(x2, 3),
        round(y2, 3),
    ]


def _center(detection):
    center = best_detection_center([detection])
    if center is None or not all(isfinite(value) for value in center):
        return None
    return center


def _box_around_center(center, width, height):
    half_width = max(4.0, width * 0.025)
    half_height = max(8.0, height * 0.09)
    return [
        round(max(0.0, center[0] - half_width), 3),
        round(max(0.0, center[1] - half_height), 3),
        round(min(float(width), center[0] + half_width), 3),
        round(min(float(height), center[1] + half_height), 3),
    ]


def _confidence(detection):
    try:
        return min(max(float(detection.get("confidence", 0.0)), 0.0), 1.0)
    except (TypeError, ValueError, AttributeError):
        return 0.0


def _dimension(value, fallback):
    try:
        value = int(value)
        return value if value > 0 else fallback
    except (TypeError, ValueError):
        return fallback
