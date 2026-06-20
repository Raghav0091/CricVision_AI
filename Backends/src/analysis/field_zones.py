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
    "Bowler",
    "First Slip",
    "Second Slip",
    "Third Slip",
    "Fourth Slip",
    "Fly Slip",
    "Leg Slip",
    "Gully",
    "Backward Point",
    "Point",
    "Forward Point",
    "Cover Point",
    "Cover",
    "Extra Cover",
    "Deep Extra Cover",
    "Mid Off",
    "Deep Mid Off",
    "Long Off",
    "Straight Hit / Long Stop",
    "Mid On",
    "Deep Mid On",
    "Long On",
    "Mid Wicket",
    "Deep Mid Wicket",
    "Cow Corner",
    "Square Leg",
    "Backward Square Leg",
    "Deep Square Leg",
    "Short Leg",
    "Silly Point",
    "Silly Mid Off",
    "Silly Mid On",
    "Short Fine Leg",
    "Fine Leg",
    "Deep Fine Leg",
    "Third Man",
    "Deep Third Man",
    "Long Leg",
    "Deep Backward Square",
    "Deep Forward Square",
]

FIELD_PRESETS = [
    "Attacking Test Field",
    "Defensive ODI Field",
    "T20 Ring Field",
    "Off Side Trap",
    "Leg Side Trap",
    "Spin Attacking Field",
    "Pace Slip Field",
    "Custom",
]

DEPTH_OPTIONS = ["close", "inner ring", "deep", "boundary"]
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
    (0, 14, "Long Off"),
    (14, 28, "Deep Extra Cover"),
    (28, 42, "Extra Cover"),
    (42, 56, "Cover"),
    (56, 72, "Cover Point"),
    (72, 90, "Point"),
    (90, 106, "Backward Point"),
    (106, 122, "Gully"),
    (122, 138, "Third Man"),
    (138, 150, "Fly Slip"),
    (150, 162, "First Slip"),
    (162, 178, "Wicket Keeper"),
    (178, 194, "Short Fine Leg"),
    (194, 212, "Fine Leg"),
    (212, 232, "Long Leg"),
    (232, 246, "Deep Backward Square"),
    (246, 264, "Square Leg"),
    (264, 282, "Deep Forward Square"),
    (282, 302, "Deep Mid Wicket"),
    (302, 318, "Cow Corner"),
    (318, 338, "Mid On"),
    (338, 350, "Long On"),
]

ZONE_ANGLES = {
    "Bowler": 0,
    "Long Off": 0,
    "Straight Hit / Long Stop": 0,
    "Deep Mid Off": 8,
    "Mid Off": 15,
    "Deep Extra Cover": 24,
    "Extra Cover": 32,
    "Cover": 45,
    "Cover Point": 58,
    "Forward Point": 72,
    "Point": 90,
    "Backward Point": 98,
    "Gully": 112,
    "Third Man / Gully": 125,
    "Fourth Slip": 124,
    "Third Slip": 132,
    "Deep Third Man": 134,
    "Third Man": 136,
    "Fly Slip": 145,
    "Second Slip": 144,
    "First Slip": 154,
    "Wicket Keeper": 180,
    "Leg Slip": 205,
    "Short Fine Leg": 195,
    "Fine Leg": 205,
    "Deep Fine Leg": 210,
    "Long Leg": 220,
    "Deep Backward Square": 235,
    "Backward Square Leg": 240,
    "Square Leg": 252,
    "Deep Square Leg": 252,
    "Short Leg": 258,
    "Deep Forward Square": 272,
    "Deep Mid Wicket": 286,
    "Mid Wicket": 296,
    "Cow Corner": 310,
    "Mid On": 326,
    "Deep Mid On": 340,
    "Long On": 344,
    "Silly Point": 95,
    "Silly Mid Off": 28,
    "Silly Mid On": 332,
}

POSITION_DEFAULTS = {
    "Wicket Keeper": {"x": 0.0, "y": -0.23, "zone": "Wicket Keeper", "depth": "close"},
    "Bowler": {"x": 0.0, "y": 0.28, "zone": "Bowler", "depth": "close"},
    "First Slip": {"x": 0.12, "y": -0.24, "zone": "First Slip", "depth": "close"},
    "Second Slip": {"x": 0.18, "y": -0.22, "zone": "Second Slip", "depth": "close"},
    "Third Slip": {"x": 0.25, "y": -0.20, "zone": "Third Slip", "depth": "close"},
    "Fourth Slip": {"x": 0.31, "y": -0.16, "zone": "Fourth Slip", "depth": "close"},
    "Fly Slip": {"x": 0.42, "y": -0.50, "zone": "Fly Slip", "depth": "deep"},
    "Leg Slip": {"x": -0.15, "y": -0.22, "zone": "Leg Slip", "depth": "close"},
    "Gully": {"x": 0.46, "y": -0.18, "zone": "Gully", "depth": "close"},
    "Backward Point": {"x": 0.55, "y": -0.08, "zone": "Backward Point", "depth": "inner ring"},
    "Point": {"x": 0.58, "y": 0.0, "zone": "Point", "depth": "inner ring"},
    "Forward Point": {"x": 0.52, "y": 0.14, "zone": "Forward Point", "depth": "inner ring"},
    "Cover Point": {"x": 0.55, "y": 0.25, "zone": "Cover Point", "depth": "inner ring"},
    "Cover": {"x": 0.45, "y": 0.42, "zone": "Cover", "depth": "inner ring"},
    "Extra Cover": {"x": 0.32, "y": 0.50, "zone": "Extra Cover", "depth": "inner ring"},
    "Deep Extra Cover": {"x": 0.44, "y": 0.75, "zone": "Deep Extra Cover", "depth": "deep"},
    "Mid Off": {"x": 0.16, "y": 0.55, "zone": "Mid Off", "depth": "inner ring"},
    "Deep Mid Off": {"x": 0.20, "y": 0.82, "zone": "Deep Mid Off", "depth": "deep"},
    "Long Off": {"x": 0.05, "y": 0.92, "zone": "Long Off", "depth": "boundary"},
    "Straight Hit / Long Stop": {"x": 0.0, "y": 0.95, "zone": "Straight Hit / Long Stop", "depth": "boundary"},
    "Mid On": {"x": -0.16, "y": 0.55, "zone": "Mid On", "depth": "inner ring"},
    "Deep Mid On": {"x": -0.20, "y": 0.82, "zone": "Deep Mid On", "depth": "deep"},
    "Long On": {"x": -0.05, "y": 0.92, "zone": "Long On", "depth": "boundary"},
    "Mid Wicket": {"x": -0.48, "y": 0.28, "zone": "Mid Wicket", "depth": "inner ring"},
    "Deep Mid Wicket": {"x": -0.72, "y": 0.32, "zone": "Deep Mid Wicket", "depth": "deep"},
    "Cow Corner": {"x": -0.75, "y": 0.50, "zone": "Cow Corner", "depth": "boundary"},
    "Square Leg": {"x": -0.55, "y": 0.0, "zone": "Square Leg", "depth": "inner ring"},
    "Backward Square Leg": {"x": -0.52, "y": -0.18, "zone": "Backward Square Leg", "depth": "inner ring"},
    "Deep Square Leg": {"x": -0.85, "y": 0.0, "zone": "Deep Square Leg", "depth": "deep"},
    "Short Leg": {"x": -0.20, "y": -0.06, "zone": "Short Leg", "depth": "close"},
    "Silly Point": {"x": 0.18, "y": 0.04, "zone": "Silly Point", "depth": "close"},
    "Silly Mid Off": {"x": 0.12, "y": 0.18, "zone": "Silly Mid Off", "depth": "close"},
    "Silly Mid On": {"x": -0.12, "y": 0.18, "zone": "Silly Mid On", "depth": "close"},
    "Short Fine Leg": {"x": -0.22, "y": -0.30, "zone": "Short Fine Leg", "depth": "close"},
    "Fine Leg": {"x": -0.38, "y": -0.55, "zone": "Fine Leg", "depth": "deep"},
    "Deep Fine Leg": {"x": -0.52, "y": -0.72, "zone": "Deep Fine Leg", "depth": "boundary"},
    "Third Man": {"x": 0.52, "y": -0.72, "zone": "Third Man", "depth": "boundary"},
    "Deep Third Man": {"x": 0.58, "y": -0.76, "zone": "Deep Third Man", "depth": "boundary"},
    "Long Leg": {"x": -0.72, "y": -0.50, "zone": "Long Leg", "depth": "boundary"},
    "Deep Backward Square": {"x": -0.78, "y": -0.28, "zone": "Deep Backward Square", "depth": "deep"},
    "Deep Forward Square": {"x": -0.78, "y": 0.20, "zone": "Deep Forward Square", "depth": "deep"},
}

DEPTH_SCALE = {"close": 0.45, "inner ring": 0.62, "deep": 0.82, "boundary": 0.95}


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


def get_position_defaults(position, depth=None):
    defaults = POSITION_DEFAULTS.get(position, POSITION_DEFAULTS["Cover"]).copy()
    selected_depth = depth or defaults["depth"]
    if selected_depth in DEPTH_SCALE and position not in {"Wicket Keeper", "Bowler"}:
        angle = math.radians(ZONE_ANGLES.get(position, ZONE_ANGLES.get(defaults["zone"], 45)))
        radius = DEPTH_SCALE[selected_depth]
        defaults["x"] = round(math.sin(angle) * radius, 3)
        defaults["y"] = round(math.cos(angle) * radius, 3)
        defaults["depth"] = selected_depth
    return defaults


def _fielder(name, position, depth=None):
    defaults = get_position_defaults(position, depth)
    return {
        "name": name,
        "position": position,
        "zone": defaults["zone"],
        "depth": defaults["depth"],
        "x": defaults["x"],
        "y": defaults["y"],
    }


def create_default_fielders(preset_name="Attacking Test Field", batter_handedness="Right-hand batter"):
    preset_name = preset_name or "Attacking Test Field"
    presets = {
        "Attacking Test Field": [
            ("Bowler", "Bowler", "close"), ("Wicket Keeper", "Wicket Keeper", "close"),
            ("First Slip", "First Slip", "close"), ("Second Slip", "Second Slip", "close"),
            ("Gully", "Gully", "close"), ("Point", "Point", "inner ring"),
            ("Cover", "Cover", "inner ring"), ("Mid Off", "Mid Off", "inner ring"),
            ("Mid On", "Mid On", "inner ring"), ("Mid Wicket", "Mid Wicket", "inner ring"),
            ("Fine Leg", "Fine Leg", "deep"),
        ],
        "Defensive ODI Field": [
            ("Bowler", "Bowler", "close"), ("Wicket Keeper", "Wicket Keeper", "close"),
            ("Third Man", "Third Man", "boundary"), ("Deep Point", "Point", "deep"),
            ("Deep Cover", "Deep Extra Cover", "deep"), ("Long Off", "Long Off", "boundary"),
            ("Long On", "Long On", "boundary"), ("Cow Corner", "Cow Corner", "boundary"),
            ("Deep Square", "Deep Square Leg", "deep"), ("Fine Leg", "Deep Fine Leg", "boundary"),
            ("Mid Off", "Mid Off", "inner ring"),
        ],
        "T20 Ring Field": [
            ("Bowler", "Bowler", "close"), ("Wicket Keeper", "Wicket Keeper", "close"),
            ("Backward Point", "Backward Point", "inner ring"), ("Cover", "Cover", "inner ring"),
            ("Extra Cover", "Extra Cover", "inner ring"), ("Mid Off", "Mid Off", "inner ring"),
            ("Mid On", "Mid On", "inner ring"), ("Mid Wicket", "Mid Wicket", "inner ring"),
            ("Square Leg", "Square Leg", "inner ring"), ("Third Man", "Third Man", "boundary"),
            ("Long On", "Long On", "boundary"),
        ],
        "Off Side Trap": [
            ("Bowler", "Bowler", "close"), ("Wicket Keeper", "Wicket Keeper", "close"),
            ("First Slip", "First Slip", "close"), ("Second Slip", "Second Slip", "close"),
            ("Gully", "Gully", "close"), ("Backward Point", "Backward Point", "inner ring"),
            ("Point", "Point", "inner ring"), ("Cover Point", "Cover Point", "inner ring"),
            ("Cover", "Cover", "inner ring"), ("Extra Cover", "Extra Cover", "inner ring"),
            ("Third Man", "Third Man", "boundary"),
        ],
        "Leg Side Trap": [
            ("Bowler", "Bowler", "close"), ("Wicket Keeper", "Wicket Keeper", "close"),
            ("Leg Slip", "Leg Slip", "close"), ("Short Leg", "Short Leg", "close"),
            ("Short Fine", "Short Fine Leg", "close"), ("Square Leg", "Square Leg", "inner ring"),
            ("Backward Square", "Backward Square Leg", "inner ring"), ("Mid Wicket", "Mid Wicket", "inner ring"),
            ("Deep Mid Wicket", "Deep Mid Wicket", "deep"), ("Fine Leg", "Fine Leg", "deep"),
            ("Long Leg", "Long Leg", "boundary"),
        ],
        "Spin Attacking Field": [
            ("Bowler", "Bowler", "close"), ("Wicket Keeper", "Wicket Keeper", "close"),
            ("Slip", "First Slip", "close"), ("Silly Point", "Silly Point", "close"),
            ("Short Leg", "Short Leg", "close"), ("Silly Mid Off", "Silly Mid Off", "close"),
            ("Silly Mid On", "Silly Mid On", "close"), ("Cover", "Cover", "inner ring"),
            ("Mid Wicket", "Mid Wicket", "inner ring"), ("Deep Square", "Deep Square Leg", "deep"),
            ("Long Off", "Long Off", "boundary"),
        ],
        "Pace Slip Field": [
            ("Bowler", "Bowler", "close"), ("Wicket Keeper", "Wicket Keeper", "close"),
            ("First Slip", "First Slip", "close"), ("Second Slip", "Second Slip", "close"),
            ("Third Slip", "Third Slip", "close"), ("Fourth Slip", "Fourth Slip", "close"),
            ("Gully", "Gully", "close"), ("Point", "Point", "inner ring"),
            ("Cover", "Cover", "inner ring"), ("Mid Off", "Mid Off", "inner ring"),
            ("Fine Leg", "Fine Leg", "deep"),
        ],
    }
    return [_fielder(*item) for item in presets.get(preset_name, presets["Attacking Test Field"])[:11]]


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


def save_field_setup(field_setup, path=FIELD_SETUP_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as setup_file:
        json.dump(field_setup, setup_file, indent=2)


def load_field_setup(path=FIELD_SETUP_PATH):
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as setup_file:
            return json.load(setup_file)
    except (OSError, json.JSONDecodeError):
        return None


def set_active_field_setup(field_setup):
    import streamlit as st
    st.session_state["current_field_setup"] = field_setup
    save_field_setup(field_setup)
    return field_setup


def get_active_field_setup():
    import streamlit as st
    if "current_field_setup" in st.session_state:
        return st.session_state["current_field_setup"]
    saved_setup = load_field_setup()
    if saved_setup is not None:
        st.session_state["current_field_setup"] = saved_setup
        return saved_setup
    default_setup = {
        "preset": "Attacking Test Field",
        "batter_handedness": "Right-hand batter",
        "bowler_arm": "Right-arm bowler",
        "camera_view": "Behind bowler",
        "fielders": create_default_fielders("Attacking Test Field", "Right-hand batter"),
        "is_default_setup": True,
    }
    st.session_state["current_field_setup"] = default_setup
    return default_setup


def save_field_analysis_history(row):
    FIELD_ANALYSIS_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp", "source", "batter_handedness", "bowler_arm", "camera_view",
        "preset", "simple_zone", "detailed_zone", "shot_angle", "nearest_fielder",
        "confidence", "corrected_zone",
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
