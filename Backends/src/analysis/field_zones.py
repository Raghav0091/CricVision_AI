"""Cricket wagon-wheel field zone utilities."""

import math


FIELD_ZONES = [
    "Third Man / Gully",
    "Point",
    "Cover",
    "Long Off",
    "Long On",
    "Mid Wicket",
    "Square Leg",
    "Fine Leg",
]


ANGLE_ZONE_RANGES = [
    (337.5, 360.0, "Long Off"),
    (0.0, 22.5, "Long Off"),
    (22.5, 67.5, "Cover"),
    (67.5, 112.5, "Point"),
    (112.5, 157.5, "Third Man / Gully"),
    (157.5, 202.5, "Fine Leg"),
    (202.5, 247.5, "Square Leg"),
    (247.5, 292.5, "Mid Wicket"),
    (292.5, 337.5, "Long On"),
]


def _valid_points(ball_trajectory):
    return [point for point in ball_trajectory or [] if point is not None]


def calculate_shot_angle(start_point, end_point):
    if start_point is None or end_point is None:
        return None

    start_x, start_y = start_point
    end_x, end_y = end_point
    dx = end_x - start_x
    dy = end_y - start_y

    if dx == 0 and dy == 0:
        return None

    return (math.degrees(math.atan2(dx, -dy)) + 360) % 360


def classify_shot_zone(angle):
    if angle is None:
        return "Unknown"

    normalized_angle = angle % 360

    for start_angle, end_angle, zone_name in ANGLE_ZONE_RANGES:
        if start_angle <= normalized_angle < end_angle:
            return zone_name

    return "Unknown"


def get_field_zone_label(angle):
    return classify_shot_zone(angle)


def _direction_change_index(points):
    if len(points) < 5:
        return None

    best_index = None
    best_turn = 0

    for index in range(2, len(points) - 2):
        prev_x, prev_y = points[index - 2]
        mid_x, mid_y = points[index]
        next_x, next_y = points[index + 2]
        vector_a = (mid_x - prev_x, mid_y - prev_y)
        vector_b = (next_x - mid_x, next_y - mid_y)
        mag_a = math.hypot(*vector_a)
        mag_b = math.hypot(*vector_b)

        if mag_a < 4 or mag_b < 4:
            continue

        dot = (vector_a[0] * vector_b[0]) + (vector_a[1] * vector_b[1])
        cosine = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
        turn_angle = math.degrees(math.acos(cosine))

        if turn_angle > best_turn:
            best_turn = turn_angle
            best_index = index

    if best_turn >= 28:
        return best_index

    return None


def _confidence_label(distance, point_count, used_contact_index):
    if distance >= 90 and point_count >= 8 and used_contact_index is not None:
        return "High"
    if distance >= 45 and point_count >= 5:
        return "Medium"
    return "Low"


def generate_wagon_wheel_data(
    ball_trajectory,
    mode="Use last part of trajectory",
    manual_contact_frame=None,
):
    points = _valid_points(ball_trajectory)

    if len(points) < 2:
        return {
            "success": False,
            "shot_angle": None,
            "estimated_zone": "Unknown",
            "nearest_zone": "Unknown",
            "confidence": "Low",
            "start_point": None,
            "end_point": None,
            "contact_index": None,
            "message": "Not enough trajectory points for shot direction.",
        }

    contact_index = None

    if mode == "Manually mark bat contact frame" and manual_contact_frame is not None:
        contact_index = max(0, min(int(manual_contact_frame), len(points) - 2))
    elif mode == "Use full trajectory":
        contact_index = 0
    else:
        contact_index = _direction_change_index(points)

        if contact_index is None:
            contact_index = max(0, len(points) - max(5, len(points) // 3))

    start_point = points[contact_index]
    end_point = points[-1]
    shot_angle = calculate_shot_angle(start_point, end_point)
    estimated_zone = classify_shot_zone(shot_angle)
    distance = math.hypot(end_point[0] - start_point[0], end_point[1] - start_point[1])
    confidence = _confidence_label(distance, len(points) - contact_index, contact_index)

    return {
        "success": shot_angle is not None,
        "shot_angle": shot_angle,
        "estimated_zone": estimated_zone,
        "nearest_zone": get_field_zone_label(shot_angle),
        "confidence": confidence,
        "start_point": start_point,
        "end_point": end_point,
        "contact_index": contact_index,
        "distance": distance,
        "message": "",
    }
