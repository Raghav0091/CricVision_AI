"""Deterministic 2D repair for short gaps in frame-level ball detections."""

from __future__ import annotations

from copy import deepcopy
from math import hypot
from statistics import median
from typing import Any


def extract_ball_path(frame_detections) -> list[dict[str, Any]]:
    """Return one safe ball-path point for every frame in the known range."""
    frames = _copy_frame_records(frame_detections)
    if not frames:
        return []

    frames_by_index = {frame["frame_index"]: frame for frame in frames}
    first_index = min(frames_by_index)
    last_index = max(frames_by_index)
    path = []

    for frame_index in range(first_index, last_index + 1):
        frame = frames_by_index.get(frame_index)
        detections = _ball_detections(frame)
        detection_index, detection = _best_ball_detection(detections)
        center = _detection_center(detection)
        bbox = _detection_bbox(detection)
        confidence = _confidence(detection)
        trusted = bool(detection.get("trusted", True)) if detection else False

        path.append(
            {
                "frame_index": frame_index,
                "x": center[0] if center else None,
                "y": center[1] if center else None,
                "confidence": confidence,
                "bbox": bbox,
                "source": _detection_source(detection),
                "trusted": trusted and center is not None,
                "repaired": bool(detection.get("repaired", False)) if detection else False,
                "detection_index": detection_index,
            }
        )
    return path


def detect_tracking_anomalies(
    ball_path,
    frame_width=None,
    frame_height=None,
    *,
    max_jump_ratio=0.25,
    min_confidence=0.15,
) -> list[dict[str, Any]]:
    """Find missing points, low-confidence detections, and implausible jumps."""
    path = list(ball_path or [])
    anomalies: list[dict[str, Any]] = []
    valid_positions = [
        index
        for index, point in enumerate(path)
        if point.get("x") is not None and point.get("y") is not None
    ]

    for point in path:
        if point.get("x") is None or point.get("y") is None:
            anomalies.append(
                {
                    "frame_index": point.get("frame_index"),
                    "anomaly_type": "missing_ball",
                    "severity": "Medium",
                }
            )
        elif _float(point.get("confidence")) < float(min_confidence):
            anomalies.append(
                {
                    "frame_index": point.get("frame_index"),
                    "anomaly_type": "low_confidence",
                    "severity": "Medium",
                    "confidence": _float(point.get("confidence")),
                    "detection_index": point.get("detection_index"),
                }
            )

    jump_threshold = _jump_threshold(
        path,
        frame_width=frame_width,
        frame_height=frame_height,
        max_jump_ratio=max_jump_ratio,
    )
    isolated_positions = set()
    for offset in range(1, len(valid_positions) - 1):
        previous_position = valid_positions[offset - 1]
        current_position = valid_positions[offset]
        next_position = valid_positions[offset + 1]
        previous = path[previous_position]
        current = path[current_position]
        following = path[next_position]

        previous_distance = _distance_per_frame(previous, current)
        next_distance = _distance_per_frame(current, following)
        bridge_distance = _distance_per_frame(previous, following)
        local_threshold = max(50.0, bridge_distance * 4.0)
        if (
            previous_distance > max(jump_threshold, local_threshold)
            and next_distance > max(jump_threshold, local_threshold)
        ):
            isolated_positions.add(current_position)
            anomalies.append(
                {
                    "frame_index": current.get("frame_index"),
                    "anomaly_type": "impossible_jump",
                    "severity": "High",
                    "reason": "isolated_outlier",
                    "jump_distance_px": round(
                        min(previous_distance, next_distance),
                        2,
                    ),
                    "detection_index": current.get("detection_index"),
                }
            )

    for offset in range(1, len(valid_positions)):
        previous_position = valid_positions[offset - 1]
        current_position = valid_positions[offset]
        if current_position in isolated_positions or previous_position in isolated_positions:
            continue
        previous = path[previous_position]
        current = path[current_position]
        jump = _distance_per_frame(previous, current)
        if jump > jump_threshold:
            anomalies.append(
                {
                    "frame_index": current.get("frame_index"),
                    "anomaly_type": "impossible_jump",
                    "severity": "High",
                    "reason": "sudden_position_change",
                    "jump_distance_px": round(jump, 2),
                    "detection_index": current.get("detection_index"),
                }
            )

    return anomalies


def repair_ball_tracking(
    frame_detections,
    max_gap_frames=4,
    max_jump_ratio=0.25,
    min_confidence=0.15,
    frame_width=None,
    frame_height=None,
):
    """Repair bounded short gaps and downgrade suspicious ball detections."""
    raw_frame_detections = deepcopy(frame_detections)
    repaired_frames = _copy_frame_records(frame_detections)
    raw_path = extract_ball_path(frame_detections)
    anomalies = detect_tracking_anomalies(
        raw_path,
        frame_width=frame_width,
        frame_height=frame_height,
        max_jump_ratio=max_jump_ratio,
        min_confidence=min_confidence,
    )

    suspicious_by_frame = _suspicious_anomalies_by_frame(anomalies)
    frames_by_index = {frame["frame_index"]: frame for frame in repaired_frames}
    for point in raw_path:
        frames_by_index.setdefault(
            point["frame_index"],
            {
                "frame_index": point["frame_index"],
                "ball_detections": [],
                "bat_detections": [],
                "stump_detections": [],
            },
        )
    repaired_frames = [frames_by_index[index] for index in sorted(frames_by_index)]

    for frame_index, anomaly in suspicious_by_frame.items():
        frame = frames_by_index.get(frame_index)
        detections = _ball_detections(frame)
        detection_index = anomaly.get("detection_index")
        if not isinstance(detection_index, int) or detection_index >= len(detections):
            continue
        detection = detections[detection_index]
        if not isinstance(detection, dict):
            continue
        original_confidence = _confidence(detection)
        detection["trusted"] = False
        detection["anomaly_type"] = anomaly["anomaly_type"]
        detection["downgraded"] = True
        detection["original_confidence"] = original_confidence
        detection["confidence"] = min(original_confidence, 0.01)

    repair_candidates = {
        point["frame_index"]
        for point in raw_path
        if point.get("x") is None
        or point.get("y") is None
        or point["frame_index"] in suspicious_by_frame
    }
    trusted_points = {
        point["frame_index"]: point
        for point in raw_path
        if point.get("x") is not None
        and point.get("y") is not None
        and point["frame_index"] not in suspicious_by_frame
        and point.get("trusted", True)
    }
    repaired_indices = []

    for gap in _consecutive_groups(sorted(repair_candidates)):
        if not gap or len(gap) > int(max_gap_frames):
            continue
        before = trusted_points.get(gap[0] - 1)
        after = trusted_points.get(gap[-1] + 1)
        if before is None or after is None:
            continue

        span = after["frame_index"] - before["frame_index"]
        if span <= 1:
            continue
        repair_confidence = _repair_confidence(before, after, len(gap))
        repaired_detection_confidence = round(
            max(0.01, min(before["confidence"], after["confidence"]) * 0.6),
            4,
        )

        for frame_index in gap:
            fraction = (frame_index - before["frame_index"]) / span
            center = [
                _interpolate(before["x"], after["x"], fraction),
                _interpolate(before["y"], after["y"], fraction),
            ]
            bbox = _interpolate_bbox(before.get("bbox"), after.get("bbox"), fraction)
            detection = {
                "center": center,
                "confidence": repaired_detection_confidence,
                "source": "observer_repair",
                "repaired": True,
                "repair_confidence": repair_confidence,
                "trusted": True,
                "class_name": "ball",
            }
            if bbox is not None:
                detection["bbox"] = bbox
                detection["box"] = list(bbox)
            _ball_detections(frames_by_index[frame_index], create=True).append(detection)
            repaired_indices.append(frame_index)

    repaired_frames = [frames_by_index[index] for index in sorted(frames_by_index)]
    repaired_path = extract_ball_path(repaired_frames)
    total_frames = len(raw_path)
    raw_detected_frames = sum(
        point.get("x") is not None and point.get("y") is not None
        for point in raw_path
    )
    repaired_detected_frames = sum(
        point.get("x") is not None
        and point.get("y") is not None
        and point.get("trusted", True)
        for point in repaired_path
    )
    missing_frames = sum(
        anomaly["anomaly_type"] == "missing_ball" for anomaly in anomalies
    )
    false_detection_candidates = sum(
        anomaly["anomaly_type"] == "impossible_jump" for anomaly in anomalies
    )
    low_confidence_detections = sum(
        anomaly["anomaly_type"] == "low_confidence" for anomaly in anomalies
    )

    return {
        "raw_frame_detections": raw_frame_detections,
        "repaired_frame_detections": repaired_frames,
        "repair_report": {
            "total_frames": total_frames,
            "original_detected_frames": raw_detected_frames,
            "repaired_detected_frames": repaired_detected_frames,
            "original_coverage": _coverage(raw_detected_frames, total_frames),
            "repaired_coverage": _coverage(repaired_detected_frames, total_frames),
            "missing_frames": missing_frames,
            "repaired_frames": len(set(repaired_indices)),
            "repaired_frame_indices": sorted(set(repaired_indices)),
            "suspicious_detections": len(suspicious_by_frame),
            "removed_or_downgraded_frames": len(suspicious_by_frame),
            "false_detection_candidates": false_detection_candidates,
            "low_confidence_detections": low_confidence_detections,
        },
        "ball_path_raw": raw_path,
        "ball_path_repaired": repaired_path,
        "anomalies": anomalies,
    }


def _copy_frame_records(frame_detections) -> list[dict[str, Any]]:
    if not frame_detections:
        return []
    items = (
        frame_detections.items()
        if isinstance(frame_detections, dict)
        else enumerate(frame_detections)
    )
    frames = []
    for fallback_index, raw_frame in items:
        if raw_frame is None:
            raw_frame = {}
        if not isinstance(raw_frame, dict):
            continue
        frame = deepcopy(raw_frame)
        try:
            frame_index = int(frame.get("frame_index", fallback_index))
        except (TypeError, ValueError):
            frame_index = len(frames)
        frame["frame_index"] = frame_index
        frames.append(frame)
    return sorted(frames, key=lambda item: item["frame_index"])


def _ball_detections(frame, create=False) -> list:
    if not isinstance(frame, dict):
        return []
    key = "ball_detections"
    if "ball_detections" not in frame and "balls" in frame:
        key = "balls"
    detections = frame.get(key)
    if isinstance(detections, list):
        return detections
    if isinstance(detections, tuple):
        detections = list(detections)
        if create:
            frame[key] = detections
        return detections
    if create:
        frame[key] = []
        return frame[key]
    return []


def _best_ball_detection(detections):
    candidates = []
    for index, detection in enumerate(detections or []):
        if not isinstance(detection, dict):
            continue
        if _detection_center(detection) is None:
            continue
        candidates.append((index, detection))
    if not candidates:
        return None, None
    return max(candidates, key=lambda item: _confidence(item[1]))


def _detection_center(detection):
    if not isinstance(detection, dict):
        return None
    center = detection.get("center")
    try:
        if center is not None and len(center) >= 2:
            return float(center[0]), float(center[1])
    except (TypeError, ValueError):
        pass
    bbox = _detection_bbox(detection)
    if bbox is None:
        return None
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _detection_bbox(detection):
    if not isinstance(detection, dict):
        return None
    bbox = detection.get("bbox") or detection.get("box") or detection.get("xyxy")
    try:
        if bbox is None or len(bbox) < 4:
            return None
        return [float(bbox[index]) for index in range(4)]
    except (TypeError, ValueError):
        return None


def _detection_source(detection):
    if not isinstance(detection, dict):
        return None
    return (
        detection.get("source")
        or detection.get("model_name")
        or detection.get("model")
        or "detector"
    )


def _confidence(detection) -> float:
    if not isinstance(detection, dict):
        return 0.0
    return _float(detection.get("confidence"))


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _distance_per_frame(point_a, point_b) -> float:
    distance = hypot(
        float(point_b["x"]) - float(point_a["x"]),
        float(point_b["y"]) - float(point_a["y"]),
    )
    frame_span = max(
        1,
        int(point_b["frame_index"]) - int(point_a["frame_index"]),
    )
    return distance / frame_span


def _jump_threshold(path, frame_width, frame_height, max_jump_ratio) -> float:
    try:
        width = float(frame_width)
        height = float(frame_height)
        if width > 0 and height > 0:
            return max(1.0, hypot(width, height) * float(max_jump_ratio))
    except (TypeError, ValueError):
        pass

    valid = [
        point
        for point in path
        if point.get("x") is not None and point.get("y") is not None
    ]
    speeds = [
        _distance_per_frame(valid[index - 1], valid[index])
        for index in range(1, len(valid))
    ]
    if not speeds:
        return 180.0
    return max(50.0, min(180.0, median(speeds) * 4.0))


def _suspicious_anomalies_by_frame(anomalies):
    priority = {"low_confidence": 1, "impossible_jump": 2}
    result = {}
    for anomaly in anomalies:
        anomaly_type = anomaly.get("anomaly_type")
        if anomaly_type not in priority:
            continue
        frame_index = anomaly.get("frame_index")
        existing = result.get(frame_index)
        if existing is None or priority[anomaly_type] > priority[existing["anomaly_type"]]:
            result[frame_index] = anomaly
    return result


def _consecutive_groups(indices):
    groups = []
    for frame_index in indices:
        if not groups or frame_index != groups[-1][-1] + 1:
            groups.append([frame_index])
        else:
            groups[-1].append(frame_index)
    return groups


def _repair_confidence(before, after, gap_length):
    anchor_confidence = min(
        _float(before.get("confidence")),
        _float(after.get("confidence")),
    )
    if anchor_confidence >= 0.75 and gap_length <= 2:
        return "High"
    if anchor_confidence >= 0.4 and gap_length <= 3:
        return "Medium"
    return "Low"


def _interpolate(start, end, fraction):
    return round(float(start) + (float(end) - float(start)) * fraction, 3)


def _interpolate_bbox(before, after, fraction):
    if before is None or after is None or len(before) < 4 or len(after) < 4:
        return None
    return [
        _interpolate(before[index], after[index], fraction)
        for index in range(4)
    ]


def _coverage(detected_frames, total_frames):
    if total_frames <= 0:
        return 0.0
    return round((detected_frames / total_frames) * 100.0, 1)
