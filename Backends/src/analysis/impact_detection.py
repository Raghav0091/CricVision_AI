"""Lightweight bat-ball impact detection helpers."""

from math import hypot
from pathlib import Path

from Backends.src.utils.cv2_loader import cv2


IMPACT_HIGH_DISTANCE_PX = 25
IMPACT_MEDIUM_DISTANCE_PX = 60
IMPACT_LOW_DISTANCE_PX = 120
IMPACT_STABILITY_WINDOW = 2


def calculate_box_center(box):
    """Return center x, center y from a bbox-like value."""
    bbox = _extract_bbox(box)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2, (y1 + y2) / 2]


def calculate_distance(point_a, point_b):
    """Euclidean distance between two points."""
    if point_a is None or point_b is None:
        return None
    return hypot(float(point_a[0]) - float(point_b[0]), float(point_a[1]) - float(point_b[1]))


def estimate_impact_confidence(
    min_distance,
    ball_detected_count,
    bat_detected_count,
    direction_change_score=None,
    stable_frame_count=0,
):
    """Return High / Medium / Low / Not Detected."""
    if min_distance is None or ball_detected_count <= 0 or bat_detected_count <= 0:
        return "Not Detected"

    if min_distance <= IMPACT_HIGH_DISTANCE_PX:
        if stable_frame_count >= IMPACT_STABILITY_WINDOW:
            return "High"
        return "Medium"
    if min_distance <= IMPACT_MEDIUM_DISTANCE_PX:
        return "Medium"
    if min_distance <= IMPACT_LOW_DISTANCE_PX:
        return "Low"
    return "Not Detected"


def detect_bat_ball_impact(frame_detections, fps=None):
    """Find likely bat-ball impact from per-frame ball and bat detections."""
    frame_items = _normalize_frame_detections(frame_detections)
    frames_processed = len(frame_items)
    ball_detected_count = 0
    bat_detected_count = 0
    frames_with_ball_and_bat = 0
    best = None
    closest_candidates = []

    for frame_item in frame_items:
        frame_index = frame_item["frame_index"]
        ball_detections = frame_item["ball_detections"]
        bat_detections = frame_item["bat_detections"]

        if ball_detections:
            ball_detected_count += 1
        if bat_detections:
            bat_detected_count += 1
        if ball_detections and bat_detections:
            frames_with_ball_and_bat += 1

        for ball in ball_detections:
            ball_center = _extract_center(ball)
            ball_bbox = _extract_bbox(ball)
            if ball_center is None:
                continue

            for bat in bat_detections:
                bat_bbox = _extract_bbox(bat)
                bat_center = _extract_center(bat)
                distance = (
                    _distance_point_to_bbox(ball_center, bat_bbox)
                    if bat_bbox is not None
                    else calculate_distance(ball_center, bat_center)
                )
                if distance is None:
                    continue

                candidate = {
                    "frame_index": frame_index,
                    "distance": float(distance),
                    "ball_bbox": ball_bbox,
                    "bat_bbox": bat_bbox,
                    "ball_center": _integer_point(ball_center),
                    "bat_center": _integer_point(bat_center or calculate_box_center(bat_bbox)),
                    "ball_confidence": _extract_confidence(ball),
                    "bat_confidence": _extract_confidence(bat),
                }
                closest_candidates.append(candidate)
                if best is None or candidate["distance"] < best["distance"]:
                    best = candidate

    if ball_detected_count == 0:
        return _empty_result(
            "Impact not detected: ball not detected.",
            frames_processed,
            ball_detected_count,
            bat_detected_count,
            frames_with_ball_and_bat,
        )
    if bat_detected_count == 0:
        return _empty_result(
            "Impact not detected: bat not detected.",
            frames_processed,
            ball_detected_count,
            bat_detected_count,
            frames_with_ball_and_bat,
        )
    if best is None:
        return _empty_result(
            "Impact not detected: no frame contained usable ball and bat detections.",
            frames_processed,
            ball_detected_count,
            bat_detected_count,
            frames_with_ball_and_bat,
        )

    stable_frame_count = _count_stable_neighbor_frames(frame_items, best["frame_index"])
    confidence = estimate_impact_confidence(
        best["distance"],
        ball_detected_count,
        bat_detected_count,
        stable_frame_count=stable_frame_count,
    )
    impact_detected = confidence != "Not Detected"
    impact_frame = best["frame_index"] if impact_detected else None
    impact_time_sec = (
        round(impact_frame / fps, 3)
        if impact_frame is not None and fps is not None and fps > 0
        else None
    )

    reason = _build_reason(
        confidence,
        best["distance"],
        frames_with_ball_and_bat,
        stable_frame_count,
    )
    closest_candidates = sorted(closest_candidates, key=lambda item: item["distance"])[:5]

    return {
        "impact_detected": impact_detected,
        "impact_frame": impact_frame,
        "impact_time_sec": impact_time_sec,
        "min_ball_bat_distance_px": round(best["distance"], 2),
        "impact_confidence": confidence,
        "reason": reason,
        "impact_reason": reason,
        "ball_bbox": _integer_box(best["ball_bbox"]),
        "bat_bbox": _integer_box(best["bat_bbox"]),
        "ball_center": best["ball_center"],
        "bat_center": best["bat_center"],
        "min_distance": round(best["distance"], 2),
        "impact_frame_image_path": None,
        "debug": {
            "frames_processed": frames_processed,
            "ball_detected_frames": ball_detected_count,
            "bat_detected_frames": bat_detected_count,
            "frames_with_ball_and_bat": frames_with_ball_and_bat,
            "stable_neighbor_frames": stable_frame_count,
            "closest_candidates": [
                {
                    "frame_index": item["frame_index"],
                    "distance": round(item["distance"], 2),
                    "ball_confidence": item["ball_confidence"],
                    "bat_confidence": item["bat_confidence"],
                }
                for item in closest_candidates
            ],
        },
    }


def save_impact_frame_preview(frame, impact, output_dir="outputs/impact_frames", prefix="impact"):
    """Save a clean impact preview frame and return its path, or None."""
    if frame is None or not impact or impact.get("impact_frame") is None:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_index = impact.get("impact_frame")
    output_path = output_dir / f"{prefix}_{int(frame_index):04d}.jpg"
    preview = frame.copy()
    draw_impact_overlay(preview, impact, title="Likely Impact Frame")
    cv2.imwrite(str(output_path), preview)
    return output_path


def draw_impact_marker(frame, impact_info, current_frame_index):
    """Draw a subtle marker on the selected impact frame in a processed video."""
    if not impact_info or impact_info.get("impact_frame") != current_frame_index:
        return frame
    draw_impact_overlay(frame, impact_info, title="IMPACT?")
    return frame


def draw_impact_overlay(frame, impact_info, title="Likely Impact Frame"):
    ball_bbox = _extract_bbox(impact_info.get("ball_bbox"))
    bat_bbox = _extract_bbox(impact_info.get("bat_bbox"))
    ball_center = impact_info.get("ball_center")
    bat_center = impact_info.get("bat_center")

    if ball_bbox is not None:
        _draw_box(frame, ball_bbox, (0, 255, 255), "ball")
    if bat_bbox is not None:
        _draw_box(frame, bat_bbox, (255, 0, 255), "bat")

    point = ball_center or calculate_box_center(ball_bbox) or bat_center or calculate_box_center(bat_bbox)
    if point is not None:
        cv2.circle(frame, tuple(map(int, point)), 22, (0, 0, 255), 3)

    cv2.rectangle(frame, (18, 18), (360, 70), (0, 0, 0), -1)
    cv2.putText(
        frame,
        title,
        (30, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 0, 255),
        3,
    )
    return frame


def _empty_result(reason, frames_processed, ball_detected_count, bat_detected_count, frames_with_ball_and_bat):
    return {
        "impact_detected": False,
        "impact_frame": None,
        "impact_time_sec": None,
        "min_ball_bat_distance_px": None,
        "impact_confidence": "Not Detected",
        "reason": reason,
        "impact_reason": reason,
        "ball_bbox": None,
        "bat_bbox": None,
        "ball_center": None,
        "bat_center": None,
        "min_distance": None,
        "impact_frame_image_path": None,
        "debug": {
            "frames_processed": frames_processed,
            "ball_detected_frames": ball_detected_count,
            "bat_detected_frames": bat_detected_count,
            "frames_with_ball_and_bat": frames_with_ball_and_bat,
            "closest_candidates": [],
        },
    }


def _normalize_frame_detections(frame_detections):
    if not frame_detections:
        return []

    items = frame_detections.items() if isinstance(frame_detections, dict) else enumerate(frame_detections)
    normalized = []
    for fallback_index, raw_frame in items:
        if raw_frame is None:
            raw_frame = {}
        if isinstance(raw_frame, dict):
            frame_index = raw_frame.get("frame_index", fallback_index)
            ball_detections = raw_frame.get("ball_detections") or raw_frame.get("balls") or []
            bat_detections = raw_frame.get("bat_detections") or raw_frame.get("bats") or []
        else:
            frame_index = fallback_index
            ball_detections = []
            bat_detections = []

        try:
            frame_index = int(frame_index)
        except (TypeError, ValueError):
            frame_index = int(fallback_index) if str(fallback_index).isdigit() else len(normalized)

        normalized.append(
            {
                "frame_index": frame_index,
                "ball_detections": list(ball_detections or []),
                "bat_detections": list(bat_detections or []),
            }
        )
    return normalized


def _extract_bbox(value):
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("bbox") or value.get("box")
    try:
        if value is None or len(value) < 4:
            return None
    except TypeError:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    return [x1, y1, x2, y2]


def _extract_center(detection):
    if not isinstance(detection, dict):
        return calculate_box_center(detection)
    center = detection.get("center") or detection.get("ball_center") or detection.get("bat_center")
    if center is not None and len(center) >= 2:
        return [float(center[0]), float(center[1])]
    return calculate_box_center(detection)


def _extract_confidence(detection):
    if not isinstance(detection, dict):
        return None
    confidence = detection.get("confidence")
    if confidence is None:
        return None
    try:
        return round(float(confidence), 4)
    except (TypeError, ValueError):
        return None


def _distance_point_to_bbox(point, bbox):
    if point is None or bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    closest_x = min(max(float(point[0]), x1), x2)
    closest_y = min(max(float(point[1]), y1), y2)
    return calculate_distance(point, [closest_x, closest_y])


def _count_stable_neighbor_frames(frame_items, impact_frame):
    stable_count = 0
    for frame_item in frame_items:
        if abs(frame_item["frame_index"] - impact_frame) > IMPACT_STABILITY_WINDOW:
            continue
        if frame_item["ball_detections"] and frame_item["bat_detections"]:
            stable_count += 1
    return stable_count


def _build_reason(confidence, distance, frames_with_ball_and_bat, stable_frame_count):
    if confidence == "Not Detected":
        return (
            "Impact not detected: closest ball-to-bat distance was "
            f"{distance:.1f}px, above the {IMPACT_LOW_DISTANCE_PX}px threshold."
        )
    return (
        f"Closest ball-to-bat distance was {distance:.1f}px with "
        f"{frames_with_ball_and_bat} frame(s) containing both bat and ball. "
        f"Stable nearby detections: {stable_frame_count}."
    )


def _integer_point(point):
    if point is None:
        return None
    return [int(round(float(point[0]))), int(round(float(point[1])))]


def _integer_box(box):
    if box is None:
        return None
    return [int(round(float(value))) for value in box[:4]]


def _draw_box(frame, bbox, color, label):
    x1, y1, x2, y2 = _integer_box(bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(22, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
    )
