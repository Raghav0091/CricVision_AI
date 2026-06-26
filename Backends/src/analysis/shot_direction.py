"""Text-based cricket field zone estimation after bat-ball impact."""

from math import atan2, degrees, hypot

from Backends.src.analysis.frame_detection_utils import (
    best_detection_center,
    find_ball_center_at_or_before,
    normalize_frame_detections,
)

DEBUG_SHOT_DIRECTION = False

POST_IMPACT_DIRECTION_LOOKAHEAD_FRAMES = 12
MIN_DIRECTION_MOVEMENT_PX = 25
STRAIGHT_ZONE_THRESHOLD_DEGREES = 15
SQUARE_ZONE_THRESHOLD_DEGREES = 70

SUPPORTED_FIELD_ZONES = {
    "Straight",
    "Long Off",
    "Mid Off",
    "Extra Cover",
    "Cover",
    "Point",
    "Gully",
    "Third Man",
    "Long On",
    "Mid On",
    "Mid Wicket",
    "Deep Mid Wicket",
    "Square Leg",
    "Fine Leg",
    "Deep Fine Leg",
    "Deep Cover",
    "Deep Point",
    "Cow Corner",
    "Unknown",
}

RH_ZONE_RANGES = [
    (-180, -158, "Fine Leg"),
    (-158, -145, "Deep Fine Leg"),
    (-145, -122, "Square Leg"),
    (-122, -72, "Mid Wicket"),
    (-72, -55, "Deep Mid Wicket"),
    (-55, -38, "Cow Corner"),
    (-38, -28, "Mid On"),
    (-28, -8, "Long On"),
    (-8, 8, "Straight"),
    (8, 28, "Long Off"),
    (28, 38, "Mid Off"),
    (38, 55, "Extra Cover"),
    (55, 72, "Cover"),
    (72, 90, "Deep Cover"),
    (90, 100, "Point"),
    (100, 122, "Deep Point"),
    (122, 145, "Gully"),
    (145, 180, "Third Man"),
]

SHOT_TYPE_ZONE_BIAS = {
    "Straight Drive": ["Straight", "Long Off", "Long On"],
    "Cover Drive": ["Cover", "Extra Cover", "Mid Off"],
    "On Drive": ["Mid On", "Long On"],
    "Cut Shot": ["Point", "Gully", "Third Man", "Deep Point"],
    "Pull Shot": ["Square Leg", "Deep Mid Wicket", "Mid Wicket"],
    "Hook Shot": ["Fine Leg", "Square Leg", "Deep Fine Leg"],
    "Flick": ["Mid Wicket", "Square Leg"],
    "Sweep": ["Square Leg", "Fine Leg", "Deep Fine Leg"],
    "Lofted Shot": ["Long On", "Long Off", "Cow Corner"],
    "Defence": ["Straight"],
}


def estimate_shot_direction_zone(
    frame_detections,
    impact_result,
    shot_result=None,
    batter_handedness=None,
    fps=None,
):
    """Estimate text-based cricket field zone where the ball travelled after impact."""
    impact_result = impact_result or {}
    shot_result = shot_result or {}

    if not impact_result.get("impact_detected") and impact_result.get("impact_frame") is None:
        return _unknown("Field zone unavailable because bat-ball impact was not detected.")

    frames = normalize_frame_detections(frame_detections)
    if not frames:
        return _unknown("Field zone requires frame-level detections around impact.")

    impact_frame = impact_result.get("impact_frame")
    try:
        impact_frame = int(impact_frame)
    except (TypeError, ValueError):
        return _unknown("Field zone requires a valid impact frame.")

    impact_point = find_ball_center_at_or_before(frames, impact_frame)
    if impact_point is None:
        impact_point = impact_result.get("ball_center")
    if impact_point is None:
        return _unknown("Field zone requires a ball position at impact.")

    post_points = _collect_post_impact_points(frames, impact_frame)
    if not post_points:
        return _unknown("Field zone requires post-impact ball tracking.")

    post_point = post_points[-1]
    dx = float(post_point[0]) - float(impact_point[0])
    dy = float(post_point[1]) - float(impact_point[1])
    movement = hypot(dx, dy)
    if movement < MIN_DIRECTION_MOVEMENT_PX:
        return _unknown(
            f"Ball movement after impact was too small ({movement:.1f}px) to estimate a field zone.",
            dx=dx,
            dy=dy,
            movement=movement,
        )

    handedness = _normalize_handedness(batter_handedness)
    angle_deg = _movement_to_cricket_angle(dx, dy, handedness)
    coarse_direction = _coarse_direction(angle_deg, dx, handedness)
    field_zone = _angle_to_zone(angle_deg, handedness)
    field_zone = _apply_shot_type_bias(field_zone, shot_result.get("shot_type"), angle_deg, handedness)
    zone_confidence = _zone_confidence(
        movement,
        len(post_points),
        impact_result.get("impact_confidence"),
        shot_result.get("shot_confidence"),
        field_zone,
    )
    reason = _build_reason(
        field_zone,
        coarse_direction,
        angle_deg,
        movement,
        dx,
        dy,
        shot_result.get("shot_type"),
        handedness,
    )

    result = {
        "shot_direction": coarse_direction,
        "field_zone": field_zone,
        "zone_confidence": zone_confidence,
        "movement_dx": round(dx, 2),
        "movement_dy": round(dy, 2),
        "direction_angle_degrees": round(angle_deg, 2),
        "reason": reason,
    }
    if DEBUG_SHOT_DIRECTION:
        result["debug"] = {
            "impact_frame": impact_frame,
            "post_impact_points": [[int(x), int(y)] for x, y in post_points],
            "handedness": handedness,
            "movement_px": round(movement, 2),
        }
    return result


def _unknown(reason, dx=None, dy=None, movement=None):
    return {
        "shot_direction": "Unknown",
        "field_zone": "Unknown",
        "zone_confidence": "Unknown",
        "movement_dx": round(dx, 2) if dx is not None else None,
        "movement_dy": round(dy, 2) if dy is not None else None,
        "direction_angle_degrees": None,
        "reason": reason,
    }


def _collect_post_impact_points(frames, impact_frame):
    points = []
    max_frame = impact_frame + POST_IMPACT_DIRECTION_LOOKAHEAD_FRAMES
    for frame_item in frames:
        frame_index = frame_item["frame_index"]
        if frame_index <= impact_frame or frame_index > max_frame:
            continue
        center = best_detection_center(frame_item["ball_detections"])
        if center is not None:
            points.append([center[0], center[1]])
    return points


def _normalize_handedness(handedness):
    value = str(handedness or "right").strip().lower()
    return "left" if value.startswith("left") else "right"


def _movement_to_cricket_angle(dx, dy, handedness):
    display_angle = degrees(atan2(dx, -dy if dy != 0 else -1e-9))
    if display_angle > 180:
        display_angle -= 360
    elif display_angle <= -180:
        display_angle += 360
    if handedness == "left":
        display_angle = -display_angle
    return display_angle


def _coarse_direction(angle_deg, dx, handedness):
    if abs(angle_deg) <= STRAIGHT_ZONE_THRESHOLD_DEGREES:
        return "Straight"
    if abs(angle_deg) >= SQUARE_ZONE_THRESHOLD_DEGREES:
        if handedness == "left":
            return "Behind Square" if dx > 0 else "Behind Square"
        return "Behind Square"
    if handedness == "left":
        return "Leg Side" if dx > 0 else "Off Side"
    return "Off Side" if dx > 0 else "Leg Side"


def _angle_to_zone(angle_deg, handedness):
    angle = float(angle_deg)
    for start, end, zone in RH_ZONE_RANGES:
        if start <= angle < end:
            return zone
    return "Unknown"


def _apply_shot_type_bias(field_zone, shot_type, angle_deg, handedness):
    if field_zone == "Unknown" or not shot_type:
        return field_zone
    preferred = SHOT_TYPE_ZONE_BIAS.get(shot_type, [])
    if not preferred:
        return field_zone
    if field_zone in preferred:
        return field_zone
    # Nudge to nearest preferred zone by angle proximity if shot type is confident
    if shot_type in {"Cover Drive", "Cut Shot", "Pull Shot", "Hook Shot", "On Drive", "Straight Drive"}:
        return preferred[0]
    return field_zone


def _zone_confidence(movement, point_count, impact_confidence, shot_confidence, field_zone):
    if field_zone == "Unknown":
        return "Unknown"
    if point_count >= 4 and movement >= MIN_DIRECTION_MOVEMENT_PX * 2 and impact_confidence in {"High", "Medium"}:
        return "High"
    if point_count >= 2 and movement >= MIN_DIRECTION_MOVEMENT_PX:
        if shot_confidence in {"High", "Medium"}:
            return "Medium"
        return "Low"
    return "Low"


def _build_reason(field_zone, coarse_direction, angle_deg, movement, dx, dy, shot_type, handedness):
    handedness_text = "left-handed" if handedness == "left" else "right-handed"
    shot_text = f" Shot type hint: {shot_type}." if shot_type and shot_type != "Unknown" else ""
    return (
        f"Estimated {field_zone} for a {handedness_text} batter with ball movement "
        f"{movement:.1f}px ({dx:.1f}px lateral, {dy:.1f}px vertical) toward {coarse_direction.lower()} "
        f"at about {angle_deg:.1f}°.{shot_text}"
    )
