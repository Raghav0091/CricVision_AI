"""Rule-based cricket shot type classification."""

from math import hypot


POST_IMPACT_LOOKAHEAD_FRAMES = 8
MIN_POST_IMPACT_MOVEMENT_PX = 20
STRAIGHT_DIRECTION_THRESHOLD_PX = 25
AERIAL_VERTICAL_THRESHOLD_PX = 40
SQUARE_SHOT_RATIO = 1.4

SHOT_TYPES = {
    "Defence",
    "Straight Drive",
    "Cover Drive",
    "On Drive",
    "Cut Shot",
    "Pull Shot",
    "Hook Shot",
    "Flick",
    "Sweep",
    "Lofted Shot",
    "Unknown",
}


def classify_shot_type(frame_detections, impact_result, batter_handedness=None, fps=None):
    """Classify likely cricket shot type using simple rule-based logic."""
    impact_result = impact_result or {}
    if not impact_result.get("impact_detected") and impact_result.get("impact_frame") is None:
        return _unknown("Shot type unavailable because bat-ball impact was not detected.")

    frames = _normalize_frame_detections(frame_detections)
    if not frames:
        return _unknown("Shot type detection requires impact frame and post-impact ball tracking.")

    impact_frame = impact_result.get("impact_frame")
    try:
        impact_frame = int(impact_frame)
    except (TypeError, ValueError):
        return _unknown("Shot type detection requires a valid impact frame.")

    handedness = _normalize_handedness(batter_handedness)
    handedness_note = ""
    if batter_handedness is None:
        handedness_note = " Batter handedness was missing, so right-handed rules were used."

    impact_point = _find_ball_center_at_or_before(frames, impact_frame)
    if impact_point is None:
        impact_point = impact_result.get("ball_center")
    if impact_point is None:
        return _unknown("Shot type detection requires a ball position at impact.")

    post_points = _collect_post_impact_points(frames, impact_frame)
    if not post_points:
        return _unknown("Shot type detection requires impact frame and post-impact ball tracking.")

    post_point = post_points[-1]
    dx = float(post_point[0]) - float(impact_point[0])
    dy = float(post_point[1]) - float(impact_point[1])
    movement = hypot(dx, dy)
    shot_direction = _estimate_direction(dx, handedness)
    shot_height = _estimate_height(dy)

    if movement < MIN_POST_IMPACT_MOVEMENT_PX:
        return _shot_result(
            "Defence",
            "Medium",
            shot_direction,
            "Grounded",
            (
                f"Ball movement after impact was small ({movement:.1f}px), "
                "which suggests a defensive shot."
            )
            + handedness_note,
            impact_frame,
            post_points,
            dx,
            dy,
            movement,
            "minimal_movement_defence",
        )

    shot_type, chosen_rule = _choose_shot_type(dx, dy, movement, shot_direction, shot_height)
    confidence = _estimate_shot_confidence(
        movement,
        len(post_points),
        impact_result.get("impact_confidence"),
        shot_type,
    )
    reason = _build_reason(
        shot_type,
        shot_direction,
        shot_height,
        movement,
        dx,
        dy,
        chosen_rule,
        handedness_note,
    )

    return _shot_result(
        shot_type,
        confidence,
        shot_direction,
        shot_height,
        reason,
        impact_frame,
        post_points,
        dx,
        dy,
        movement,
        chosen_rule,
    )


def _unknown(reason):
    return {
        "shot_type": "Unknown",
        "shot_confidence": "Unknown",
        "shot_direction": "Unknown",
        "shot_height": "Unknown",
        "reason": reason,
        "shot_reason": reason,
        "debug": {},
    }


def _shot_result(
    shot_type,
    confidence,
    direction,
    height,
    reason,
    impact_frame,
    post_points,
    dx,
    dy,
    movement,
    chosen_rule,
):
    return {
        "shot_type": shot_type if shot_type in SHOT_TYPES else "Unknown",
        "shot_confidence": confidence,
        "shot_direction": direction,
        "shot_height": height,
        "reason": reason,
        "shot_reason": reason,
        "debug": {
            "impact_frame": impact_frame,
            "post_impact_ball_points": [[int(point[0]), int(point[1])] for point in post_points],
            "dx": round(dx, 2),
            "dy": round(dy, 2),
            "movement_px": round(movement, 2),
            "chosen_rule": chosen_rule,
        },
    }


def _normalize_frame_detections(frame_detections):
    if not frame_detections:
        return []

    items = frame_detections.items() if isinstance(frame_detections, dict) else enumerate(frame_detections)
    normalized = []
    for fallback_index, raw_frame in items:
        raw_frame = raw_frame or {}
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
                "ball_detections": list(raw_frame.get("ball_detections") or raw_frame.get("balls") or []),
            }
        )
    return sorted(normalized, key=lambda item: item["frame_index"])


def _find_ball_center_at_or_before(frames, impact_frame):
    candidates = [
        item for item in frames if item["frame_index"] <= impact_frame and item["ball_detections"]
    ]
    if not candidates:
        return None
    return _best_ball_center(candidates[-1]["ball_detections"])


def _collect_post_impact_points(frames, impact_frame):
    points = []
    max_frame = impact_frame + POST_IMPACT_LOOKAHEAD_FRAMES
    for frame_item in frames:
        frame_index = frame_item["frame_index"]
        if frame_index <= impact_frame or frame_index > max_frame:
            continue
        center = _best_ball_center(frame_item["ball_detections"])
        if center is not None:
            points.append(center)
    return points


def _best_ball_center(ball_detections):
    if not ball_detections:
        return None

    def confidence(detection):
        try:
            return float(detection.get("confidence", 0))
        except (TypeError, ValueError, AttributeError):
            return 0

    best = max(ball_detections, key=confidence)
    center = best.get("center") if isinstance(best, dict) else None
    if center is not None and len(center) >= 2:
        return [float(center[0]), float(center[1])]

    bbox = None
    if isinstance(best, dict):
        bbox = best.get("bbox") or best.get("box")
    try:
        if bbox is None or len(bbox) < 4:
            return None
        return [(float(bbox[0]) + float(bbox[2])) / 2, (float(bbox[1]) + float(bbox[3])) / 2]
    except (TypeError, ValueError):
        return None


def _normalize_handedness(handedness):
    value = str(handedness or "right").strip().lower()
    return "left" if value.startswith("left") else "right"


def _estimate_direction(dx, handedness):
    if abs(dx) <= STRAIGHT_DIRECTION_THRESHOLD_PX:
        return "Straight"
    moves_to_image_right = dx > 0
    if handedness == "left":
        return "Leg Side" if moves_to_image_right else "Off Side"
    return "Off Side" if moves_to_image_right else "Leg Side"


def _estimate_height(dy):
    if dy < -AERIAL_VERTICAL_THRESHOLD_PX:
        return "Aerial"
    if dy > -AERIAL_VERTICAL_THRESHOLD_PX / 2:
        return "Grounded"
    return "Unknown"


def _choose_shot_type(dx, dy, movement, direction, height):
    horizontal = abs(dx)
    vertical = abs(dy)
    horizontal_ratio = horizontal / max(vertical, 1)

    if height == "Aerial" and movement >= MIN_POST_IMPACT_MOVEMENT_PX * 3:
        if direction == "Leg Side" and horizontal_ratio >= 1:
            return "Hook Shot", "aerial_leg_side_hook"
        return "Lofted Shot", "aerial_lofted"

    if direction == "Straight":
        return "Straight Drive", "straight_post_impact_movement"

    if direction == "Off Side":
        if horizontal_ratio >= SQUARE_SHOT_RATIO:
            return "Cut Shot", "off_side_square_cut"
        return "Cover Drive", "off_side_diagonal_cover_drive"

    if direction == "Leg Side":
        if horizontal_ratio >= SQUARE_SHOT_RATIO:
            return "Pull Shot", "leg_side_square_pull"
        if movement < MIN_POST_IMPACT_MOVEMENT_PX * 2.5:
            return "Flick", "compact_leg_side_flick"
        return "On Drive", "leg_side_forward_on_drive"

    return "Unknown", "unclear_movement"


def _estimate_shot_confidence(movement, point_count, impact_confidence, shot_type):
    if shot_type == "Unknown":
        return "Unknown"
    if point_count >= 3 and movement >= MIN_POST_IMPACT_MOVEMENT_PX * 2 and impact_confidence in {"High", "Medium"}:
        return "High"
    if point_count >= 2 and movement >= MIN_POST_IMPACT_MOVEMENT_PX:
        return "Medium"
    return "Low"


def _build_reason(shot_type, direction, height, movement, dx, dy, chosen_rule, handedness_note):
    return (
        f"Classified as {shot_type} because post-impact ball movement was "
        f"{movement:.1f}px ({dx:.1f}px lateral, {dy:.1f}px vertical), toward {direction.lower()}, "
        f"with {height.lower()} height. Rule: {chosen_rule}."
        f"{handedness_note}"
    )
