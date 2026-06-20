"""Cricket field setup and wagon-wheel utilities."""

import csv
import json
import math
from pathlib import Path


SIMPLE_FIELD_ZONES = [
    "Third Man / Gully",
    "Point",
    "Cover",
    "Long Off",
    "Long On",
    "Mid Wicket",
    "Square Leg",
    "Fine Leg",
]

DETAILED_FIELD_ZONES = [
    "Wicket Keeper",
    "First Slip",
    "Second Slip",
    "Third Slip",
    "Gully",
    "Point",
    "Backward Point",
    "Cover",
    "Extra Cover",
    "Mid Off",
    "Long Off",
    "Long On",
    "Mid On",
    "Mid Wicket",
    "Deep Mid Wicket",
    "Square Leg",
    "Deep Square Leg",
    "Fine Leg",
    "Third Man",
]

FIELD_PRESETS = [
    "Attacking Field",
    "Defensive Field",
    "Off-side Field",
    "Leg-side Field",
    "T20 Ring Field",
    "Custom",
]

DEPTH_OPTIONS = ["close", "inner ring", "deep"]
FIELD_SETUP_PATH = Path("outputs/field_setups/latest_field_setup.json")
FIELD_ANALYSIS_HISTORY_PATH = Path("outputs/video_analysis/field_analysis_history.csv")


SIMPLE_RIGHT_HAND_RANGES = [
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

DETAILED_RIGHT_HAND_RANGES = [
    (350, 360, "Long Off"),
    (0, 15, "Long Off"),
    (15, 35, "Extra Cover"),
    (35, 55, "Cover"),
    (55, 75, "Backward Point"),
    (75, 100, "Point"),
    (100, 118, "Gully"),
    (118, 132, "Third Slip"),
    (132, 145, "Second Slip"),
    (145, 158, "First Slip"),
    (158, 175, "Wicket Keeper"),
    (175, 202, "Fine Leg"),
    (202, 230, "Deep Square Leg"),
    (230, 255, "Square Leg"),
    (255, 282, "Deep Mid Wicket"),
    (282, 306, "Mid Wicket"),
    (306, 330, "Mid On"),
    (330, 350, "Long On"),
]

ZONE_ANGLES = {
    "Long Off": 0,
    "Extra Cover": 25,
    "Cover": 45,
    "Backward Point": 65,
    "Point": 90,
    "Gully": 112,
    "Third Man / Gully": 130,
    "Third Man": 132,
    "Third Slip": 128,
    "Second Slip": 140,
    "First Slip": 152,
    "Wicket Keeper": 180,
    "Fine Leg": 185,
    "Deep Square Leg": 218,
    "Square Leg": 245,
    "Deep Mid Wicket": 272,
    "Mid Wicket": 292,
    "Mid On": 318,
    "Long On": 340,
    "Mid Off": 20,
}


def normalize_handedness(batter_handedness):
    if str(batter_handedness).lower().startswith("left"):
        return "Left-hand batter"
    return "Right-hand batter"


def mirror_angle_for_handedness(angle, batter_handedness):
    if angle is None:
        return None
    if normalize_handedness(batter_handedness).startswith("Left"):
        return (360 - angle) % 360
    return angle % 360


def _zone_from_ranges(angle, ranges):
    if angle is None:
        return "Unknown"

    normalized_angle = angle % 360
    for start_angle, end_angle, zone_name in ranges:
        if start_angle <= normalized_angle < end_angle:
            return zone_name
    return "Unknown"


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


def get_simple_8_zone(angle, batter_handedness="Right-hand batter"):
    adjusted_angle = mirror_angle_for_handedness(angle, batter_handedness)
    return _zone_from_ranges(adjusted_angle, SIMPLE_RIGHT_HAND_RANGES)


def get_detailed_field_zone(angle, batter_handedness="Right-hand batter"):
    adjusted_angle = mirror_angle_for_handedness(angle, batter_handedness)
    return _zone_from_ranges(adjusted_angle, DETAILED_RIGHT_HAND_RANGES)


def classify_shot_zone(angle, batter_handedness="Right-hand batter"):
    return get_simple_8_zone(angle, batter_handedness)


def get_field_zone_label(angle, batter_handedness="Right-hand batter"):
    return get_simple_8_zone(angle, batter_handedness)


def _valid_points(ball_trajectory):
    return [point for point in ball_trajectory or [] if point is not None]


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
    batter_handedness="Right-hand batter",
    mode="Use last part of trajectory",
    manual_contact_frame=None,
):
    points = _valid_points(ball_trajectory)

    if len(points) < 2:
        return {
            "success": False,
            "shot_angle": None,
            "estimated_zone": "Unknown",
            "simple_zone": "Unknown",
            "detailed_zone": "Unknown",
            "nearest_zone": "Unknown",
            "confidence": "Low",
            "start_point": None,
            "end_point": None,
            "contact_index": None,
            "message": "Not enough trajectory points for shot direction.",
        }

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
    simple_zone = get_simple_8_zone(shot_angle, batter_handedness)
    detailed_zone = get_detailed_field_zone(shot_angle, batter_handedness)
    distance = math.hypot(end_point[0] - start_point[0], end_point[1] - start_point[1])
    confidence = _confidence_label(distance, len(points) - contact_index, contact_index)

    return {
        "success": shot_angle is not None,
        "shot_angle": shot_angle,
        "estimated_zone": simple_zone,
        "simple_zone": simple_zone,
        "detailed_zone": detailed_zone,
        "nearest_zone": detailed_zone,
        "confidence": confidence,
        "start_point": start_point,
        "end_point": end_point,
        "contact_index": contact_index,
        "distance": distance,
        "message": "",
    }


def _fielder(name, position, zone, depth):
    angle = ZONE_ANGLES.get(zone, 0)
    radius_by_depth = {"close": 0.25, "inner ring": 0.52, "deep": 0.82}
    radius = radius_by_depth.get(depth, 0.52)
    radians = math.radians(angle)
    return {
        "name": name,
        "position": position,
        "zone": zone,
        "depth": depth,
        "x": round(math.sin(radians) * radius, 3),
        "y": round(math.cos(radians) * radius, 3),
    }


def create_default_fielders(preset_name="Attacking Field", batter_handedness="Right-hand batter"):
    preset_name = preset_name or "Attacking Field"
    base = [
        ("Bowler", "Bowler", "Mid Off", "close"),
        ("Wicket Keeper", "Wicket Keeper", "Wicket Keeper", "close"),
        ("Slip", "Slip", "First Slip", "close"),
        ("Gully", "Gully", "Gully", "close"),
        ("Point", "Point", "Point", "inner ring"),
        ("Cover", "Cover", "Cover", "inner ring"),
        ("Mid Off", "Mid Off", "Mid Off", "inner ring"),
        ("Mid On", "Mid On", "Mid On", "inner ring"),
        ("Mid Wicket", "Mid Wicket", "Mid Wicket", "inner ring"),
        ("Square Leg", "Square Leg", "Square Leg", "inner ring"),
        ("Fine Leg", "Fine Leg", "Fine Leg", "deep"),
    ]

    if preset_name == "Defensive Field":
        base = [
            ("Bowler", "Bowler", "Mid Off", "close"),
            ("Wicket Keeper", "Wicket Keeper", "Wicket Keeper", "close"),
            ("Third Man", "Third Man", "Third Man", "deep"),
            ("Deep Point", "Point", "Point", "deep"),
            ("Deep Cover", "Cover", "Cover", "deep"),
            ("Long Off", "Long Off", "Long Off", "deep"),
            ("Long On", "Long On", "Long On", "deep"),
            ("Deep Mid Wicket", "Mid Wicket", "Deep Mid Wicket", "deep"),
            ("Deep Square", "Square Leg", "Deep Square Leg", "deep"),
            ("Fine Leg", "Fine Leg", "Fine Leg", "deep"),
            ("Mid Off", "Mid Off", "Mid Off", "inner ring"),
        ]
    elif preset_name == "Off-side Field":
        base[2:] = [
            ("Slip", "Slip", "First Slip", "close"),
            ("Gully", "Gully", "Gully", "close"),
            ("Backward Point", "Backward Point", "Backward Point", "inner ring"),
            ("Point", "Point", "Point", "inner ring"),
            ("Cover", "Cover", "Cover", "inner ring"),
            ("Extra Cover", "Extra Cover", "Extra Cover", "inner ring"),
            ("Mid Off", "Mid Off", "Mid Off", "inner ring"),
            ("Long Off", "Long Off", "Long Off", "deep"),
            ("Fine Leg", "Fine Leg", "Fine Leg", "deep"),
        ]
    elif preset_name == "Leg-side Field":
        base[2:] = [
            ("Fine Leg", "Fine Leg", "Fine Leg", "deep"),
            ("Square Leg", "Square Leg", "Square Leg", "inner ring"),
            ("Deep Square", "Square Leg", "Deep Square Leg", "deep"),
            ("Mid Wicket", "Mid Wicket", "Mid Wicket", "inner ring"),
            ("Deep Mid Wicket", "Mid Wicket", "Deep Mid Wicket", "deep"),
            ("Mid On", "Mid On", "Mid On", "inner ring"),
            ("Long On", "Long On", "Long On", "deep"),
            ("Point", "Point", "Point", "inner ring"),
            ("Cover", "Cover", "Cover", "inner ring"),
        ]
    elif preset_name == "T20 Ring Field":
        base[2:] = [
            ("Third Man", "Third Man", "Third Man", "deep"),
            ("Point", "Point", "Point", "inner ring"),
            ("Cover", "Cover", "Cover", "inner ring"),
            ("Long Off", "Long Off", "Long Off", "deep"),
            ("Long On", "Long On", "Long On", "deep"),
            ("Mid Wicket", "Mid Wicket", "Mid Wicket", "inner ring"),
            ("Deep Square", "Square Leg", "Deep Square Leg", "deep"),
            ("Fine Leg", "Fine Leg", "Fine Leg", "deep"),
            ("Extra Cover", "Extra Cover", "Extra Cover", "inner ring"),
        ]

    return [_fielder(*item) for item in base[:11]]


def find_nearest_fielder(shot_angle, fielders):
    if shot_angle is None or not fielders:
        return None

    target_x = math.sin(math.radians(shot_angle))
    target_y = math.cos(math.radians(shot_angle))
    best_fielder = None
    best_distance = None

    for fielder in fielders:
        try:
            fielder_x = float(fielder.get("x", 0))
            fielder_y = float(fielder.get("y", 0))
        except (TypeError, ValueError):
            continue

        fielder_length = math.hypot(fielder_x, fielder_y) or 1
        fielder_x /= fielder_length
        fielder_y /= fielder_length
        distance = math.hypot(target_x - fielder_x, target_y - fielder_y)

        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_fielder = fielder

    return best_fielder


def suggest_field_adjustment(wagon_wheel, nearest_fielder):
    detailed_zone = wagon_wheel.get("detailed_zone", "Unknown")

    if detailed_zone == "Unknown":
        return "Shot direction is uncertain; review manually before changing the field."

    if nearest_fielder is None:
        return f"Shot direction appears to be {detailed_zone}. Consider placing a fielder in that zone."

    name = nearest_fielder.get("name", "nearest fielder")
    return (
        f"Shot direction appears to be {wagon_wheel.get('simple_zone', 'Unknown')} / {detailed_zone}. "
        f"Nearest fielder is {name}. If this shot repeats, consider moving {name} slightly deeper or squarer."
    )


def save_field_setup(field_setup):
    FIELD_SETUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FIELD_SETUP_PATH, "w", encoding="utf-8") as setup_file:
        json.dump(field_setup, setup_file, indent=2)


def save_field_analysis_history(row):
    FIELD_ANALYSIS_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "source",
        "batter_handedness",
        "bowler_arm",
        "camera_view",
        "preset",
        "simple_zone",
        "detailed_zone",
        "shot_angle",
        "nearest_fielder",
        "confidence",
        "corrected_zone",
    ]
    write_header = not FIELD_ANALYSIS_HISTORY_PATH.exists()

    with open(FIELD_ANALYSIS_HISTORY_PATH, "a", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def create_field_map_figure(fielders, shot_direction=None):
    from Backends.src.ui.field_map import draw_field_map

    shot_angle = None if shot_direction is None else shot_direction.get("shot_angle")
    selected_zone = "Unknown" if shot_direction is None else shot_direction.get("detailed_zone", "Unknown")
    return draw_field_map(shot_angle=shot_angle, selected_zone=selected_zone, fielders=fielders)


FIELD_ZONES = SIMPLE_FIELD_ZONES
