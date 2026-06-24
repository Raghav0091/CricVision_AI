"""Rule-based cricket shot outcome prediction."""

from math import hypot


POST_IMPACT_OUTCOME_LOOKAHEAD_FRAMES = 15
DOT_BALL_DISTANCE_PX = 40
SINGLE_DISTANCE_PX = 120
DOUBLE_DISTANCE_PX = 220
FOUR_DISTANCE_PX = 350
FAST_SHOT_SPEED_PX_PER_FRAME = 35
AERIAL_VERTICAL_THRESHOLD_PX = 40
FIELDER_NEAR_BALL_DISTANCE_PX = 80

SUPPORTED_OUTCOMES = {
    "Dot Ball",
    "Single",
    "Double",
    "Three",
    "Four",
    "Six",
    "Caught Chance",
    "Unknown",
}


def predict_shot_outcome(
    frame_detections,
    impact_result,
    shot_result,
    fps=None,
    batter_handedness=None,
):
    """Predict likely cricket outcome after bat-ball impact."""
    impact_result = impact_result or {}
    shot_result = shot_result or {}

    if not impact_result.get("impact_detected") and impact_result.get("impact_frame") is None:
        return _unknown("Outcome unavailable because bat-ball impact was not detected.")
    if not shot_result:
        return _unknown("Outcome prediction requires shot type detection.")

    frames = _normalize_frame_detections(frame_detections)
    if not frames:
        return _unknown("Outcome prediction requires impact frame and post-impact ball tracking.")

    impact_frame = impact_result.get("impact_frame")
    try:
        impact_frame = int(impact_frame)
    except (TypeError, ValueError):
        return _unknown("Outcome prediction requires a valid impact frame.")

    impact_point = _find_ball_center_at_or_before(frames, impact_frame)
    if impact_point is None:
        impact_point = impact_result.get("ball_center")
    if impact_point is None:
        return _unknown("Outcome prediction requires a ball position at impact.")

    post_points = _collect_post_impact_points(frames, impact_frame)
    if not post_points:
        return _unknown("Outcome prediction requires impact frame and post-impact ball tracking.")

    total_movement = _path_distance([impact_point, *post_points])
    final_point = post_points[-1]
    dx = float(final_point[0]) - float(impact_point[0])
    dy = float(final_point[1]) - float(impact_point[1])
    frame_span = max(1, _last_post_frame(frames, impact_frame) - impact_frame)
    average_speed = total_movement / frame_span
    shot_height = shot_result.get("shot_height", "Unknown")
    aerial = _is_aerial(shot_height, dy)
    tracking_disappears = _tracking_disappears_after_fast_shot(
        frames,
        impact_frame,
        average_speed,
    )
    fielder_near_ball = _fielder_near_ball(frames, final_point, impact_frame)

    predicted_outcome, run_estimate, dismissal_risk, boundary_chance, confidence, chosen_rule = _choose_outcome(
        total_movement,
        average_speed,
        aerial,
        tracking_disappears,
        fielder_near_ball,
        shot_result,
    )

    reason = _build_reason(
        predicted_outcome,
        total_movement,
        average_speed,
        aerial,
        tracking_disappears,
        fielder_near_ball,
        chosen_rule,
    )

    return {
        "predicted_outcome": predicted_outcome,
        "outcome_confidence": confidence,
        "run_estimate": run_estimate,
        "dismissal_risk": dismissal_risk,
        "boundary_chance": boundary_chance,
        "reason": reason,
        "outcome_reason": reason,
        "debug": {
            "impact_frame": impact_frame,
            "post_impact_ball_points": [[int(point[0]), int(point[1])] for point in post_points],
            "total_post_impact_movement_px": round(total_movement, 2),
            "average_speed_px_per_frame": round(average_speed, 2),
            "dx": round(dx, 2),
            "dy": round(dy, 2),
            "aerial": aerial,
            "tracking_disappears_after_fast_shot": tracking_disappears,
            "fielder_near_ball": fielder_near_ball,
            "chosen_rule": chosen_rule,
        },
    }


def _unknown(reason):
    return {
        "predicted_outcome": "Unknown",
        "outcome_confidence": "Unknown",
        "run_estimate": None,
        "dismissal_risk": "Unknown",
        "boundary_chance": "Unknown",
        "reason": reason,
        "outcome_reason": reason,
        "debug": {},
    }


def _choose_outcome(
    total_movement,
    average_speed,
    aerial,
    tracking_disappears,
    fielder_near_ball,
    shot_result,
):
    shot_confidence = shot_result.get("shot_confidence", "Unknown")

    if aerial and fielder_near_ball:
        return "Caught Chance", None, "High", "Low", "Medium", "aerial_fielder_near_ball"

    if aerial and (
        total_movement >= FOUR_DISTANCE_PX
        or average_speed >= FAST_SHOT_SPEED_PX_PER_FRAME
        or tracking_disappears
    ):
        confidence = "Medium" if shot_confidence in {"High", "Medium"} else "Low"
        return "Six", 6, "Medium", "High", confidence, "aerial_fast_or_far"

    if total_movement < DOT_BALL_DISTANCE_PX:
        return "Dot Ball", 0, "Low", "Low", "High", "minimal_post_impact_movement"

    if not aerial and (
        total_movement >= FOUR_DISTANCE_PX
        or average_speed >= FAST_SHOT_SPEED_PX_PER_FRAME
        or tracking_disappears
    ):
        confidence = "High" if shot_confidence in {"High", "Medium"} else "Medium"
        return "Four", 4, "Low", "High", confidence, "grounded_fast_boundary_chance"

    if not aerial and total_movement >= DOUBLE_DISTANCE_PX:
        if total_movement >= (DOUBLE_DISTANCE_PX + FOUR_DISTANCE_PX) / 2:
            return "Three", 3, "Low", "Medium", "Low", "grounded_large_non_boundary"
        return "Double", 2, "Low", "Medium", "Medium", "grounded_medium_large_movement"

    if not aerial and total_movement >= SINGLE_DISTANCE_PX:
        return "Double", 2, "Low", "Medium", "Medium", "grounded_medium_movement"

    if not aerial:
        return "Single", 1, "Low", "Low", "Medium", "grounded_small_movement"

    return "Unknown", None, "Unknown", "Unknown", "Low", "unclear_aerial_movement"


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


def _find_ball_center_at_or_before(frames, impact_frame):
    candidates = [
        item for item in frames if item["frame_index"] <= impact_frame and item["ball_detections"]
    ]
    if not candidates:
        return None
    return _best_center(candidates[-1]["ball_detections"])


def _collect_post_impact_points(frames, impact_frame):
    points = []
    max_frame = impact_frame + POST_IMPACT_OUTCOME_LOOKAHEAD_FRAMES
    for frame_item in frames:
        frame_index = frame_item["frame_index"]
        if frame_index <= impact_frame or frame_index > max_frame:
            continue
        center = _best_center(frame_item["ball_detections"])
        if center is not None:
            points.append(center)
    return points


def _last_post_frame(frames, impact_frame):
    max_frame = impact_frame
    for frame_item in frames:
        if impact_frame < frame_item["frame_index"] <= impact_frame + POST_IMPACT_OUTCOME_LOOKAHEAD_FRAMES:
            if frame_item["ball_detections"]:
                max_frame = max(max_frame, frame_item["frame_index"])
    return max_frame


def _best_center(detections):
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
        return [float(center[0]), float(center[1])]
    bbox = best.get("bbox") or best.get("box")
    try:
        if bbox is None or len(bbox) < 4:
            return None
        return [(float(bbox[0]) + float(bbox[2])) / 2, (float(bbox[1]) + float(bbox[3])) / 2]
    except (TypeError, ValueError):
        return None


def _path_distance(points):
    distance = 0
    for index in range(1, len(points)):
        distance += hypot(
            float(points[index][0]) - float(points[index - 1][0]),
            float(points[index][1]) - float(points[index - 1][1]),
        )
    return distance


def _is_aerial(shot_height, dy):
    if shot_height == "Aerial":
        return True
    if shot_height == "Grounded":
        return False
    return dy < -AERIAL_VERTICAL_THRESHOLD_PX


def _tracking_disappears_after_fast_shot(frames, impact_frame, average_speed):
    if average_speed < FAST_SHOT_SPEED_PX_PER_FRAME:
        return False
    max_expected_frame = impact_frame + POST_IMPACT_OUTCOME_LOOKAHEAD_FRAMES
    detected_frames = [
        item["frame_index"]
        for item in frames
        if impact_frame < item["frame_index"] <= max_expected_frame and item["ball_detections"]
    ]
    if not detected_frames:
        return False
    return max(detected_frames) < max_expected_frame - 3


def _fielder_near_ball(frames, ball_point, impact_frame):
    if ball_point is None:
        return False
    for frame_item in frames:
        if frame_item["frame_index"] < impact_frame:
            continue
        for fielder in frame_item.get("fielder_detections", []):
            center = _best_center([fielder])
            if center is None:
                continue
            if hypot(float(center[0]) - float(ball_point[0]), float(center[1]) - float(ball_point[1])) <= FIELDER_NEAR_BALL_DISTANCE_PX:
                return True
    return False


def _build_reason(
    predicted_outcome,
    total_movement,
    average_speed,
    aerial,
    tracking_disappears,
    fielder_near_ball,
    chosen_rule,
):
    path_text = "aerial" if aerial else "along the ground"
    extra = []
    if tracking_disappears:
        extra.append("tracking disappeared after a fast shot")
    if fielder_near_ball:
        extra.append("a fielder/player detection was close to the ball")
    extra_text = f" Additional signal: {', '.join(extra)}." if extra else ""
    return (
        f"Predicted {predicted_outcome} because the ball travelled {path_text} for "
        f"{total_movement:.1f}px after impact at {average_speed:.1f}px/frame. "
        f"Rule: {chosen_rule}.{extra_text}"
    )
