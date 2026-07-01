"""Cricket field setup and wagon-wheel utilities."""

import csv
import json
import math
from pathlib import Path


from Backends.src.analysis.field_geometry import (
    UMPIRE_POSITIONS_RH,
    angle_to_field_zone,
    display_angle_for_handedness,
    ensure_fielder_polar,
    ensure_umpire_polar,
    mirror_angle_for_handedness as mirror_signed_angle_for_handedness,
    normalize_handedness as normalize_handedness_value,
    polar_to_xy,
    umpire_from_name,
)
from Backends.src.config.paths import FIELD_ANALYSIS_HISTORY_PATH, FIELD_SETUP_PATH
from Backends.src.data.field_presets import PRESET_NAMES, create_preset_fielders

SIMPLE_FIELD_ZONES = [
    "Third Man",
    "Gully",
    "Point",
    "Cover",
    "Extra Cover",
    "Mid Off",
    "Long Off",
    "Long On",
    "Mid On",
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
FIELD_COORDINATE_SYSTEM_VERSION = 3


# Base coordinates are always stored from a right-handed batter's perspective.
# Negative Y is the bowler/straight-hit side; positive Y is the wicketkeeper side.
RIGHT_HANDED_FIELD_POSITIONS = {
    "Wicket Keeper": (0.0, 0.35),
    "First Slip": (0.18, 0.42),
    "Second Slip": (0.23, 0.38),
    "Third Slip": (0.28, 0.34),
    "Gully": (0.40, 0.25),
    "Point": (0.62, 0.00),
    "Cover": (0.48, -0.42),
    "Extra Cover": (0.38, -0.55),
    "Mid Off": (0.22, -0.58),
    "Long Off": (0.25, -0.88),
    "Mid On": (-0.22, -0.58),
    "Long On": (-0.25, -0.88),
    "Mid Wicket": (-0.62, 0.00),
    "Square Leg": (-0.62, 0.35),
    "Fine Leg": (-0.42, 0.78),
    "Deep Fine Leg": (-0.55, 0.82),
    "Deep Square Leg": (-0.78, 0.35),
    "Deep Mid Wicket": (-0.82, -0.05),
    "Third Man": (0.55, 0.72),
    "Deep Point": (0.85, 0.00),
    "Deep Cover": (0.78, -0.48),
    "Long Stop": (0.0, 0.88),
}

RIGHT_HANDED_UMPIRE_POSITIONS = {
    "Bowler's End Umpire": (0.0, -0.18),
    "Square Leg Umpire": (-0.72, 0.00),
}


SIMPLE_RIGHT_HAND_RANGES = [
    (337.5, 360.0, "Long Off"),
    (0.0, 16.0, "Long Off"),
    (16.0, 34.0, "Mid Off"),
    (34.0, 52.0, "Extra Cover"),
    (52.0, 72.0, "Cover"),
    (72.0, 100.0, "Point"),
    (100.0, 122.0, "Gully"),
    (122.0, 158.0, "Third Man"),
    (158.0, 205.0, "Fine Leg"),
    (205.0, 250.0, "Square Leg"),
    (250.0, 296.0, "Mid Wicket"),
    (296.0, 324.0, "Mid On"),
    (324.0, 337.5, "Long On"),
]

DETAILED_RIGHT_HAND_RANGES = [
    (337.5, 360.0, "Long Off"),
    (0.0, 16.0, "Long Off"),
    (16.0, 34.0, "Mid Off"),
    (34.0, 52.0, "Extra Cover"),
    (52.0, 72.0, "Cover"),
    (72.0, 100.0, "Point"),
    (100.0, 122.0, "Gully"),
    (122.0, 158.0, "Third Man"),
    (158.0, 205.0, "Fine Leg"),
    (205.0, 250.0, "Square Leg"),
    (250.0, 296.0, "Mid Wicket"),
    (296.0, 324.0, "Mid On"),
    (324.0, 337.5, "Long On"),
]

ZONE_ANGLES = {
    "Bowler": 0,
    "Straight": 0,
    "Long Off": 18,
    "Long On": -18,
    "Mid Off": 28,
    "Mid On": -28,
    "Extra Cover": 42,
    "Cover": 55,
    "Deep Cover": 55,
    "Point": 90,
    "Deep Point": 90,
    "Gully": 120,
    "Third Man": 145,
    "Wicket Keeper": 180,
    "First Slip": 155,
    "Second Slip": 150,
    "Third Slip": 145,
    "Fine Leg": -145,
    "Short Fine Leg": -140,
    "Square Leg": -90,
    "Deep Square Leg": -90,
    "Mid Wicket": -55,
    "Deep Mid Wicket": -55,
    "Cow Corner": -38,
    "Deep Extra Cover": 42,
    "Deep Mid Off": 28,
    "Deep Mid On": -28,
    "Straight Hit / Long Stop": 0,
    "Third Man / Gully": 125,
    "Fourth Slip": 124,
    "Deep Third Man": 134,
    "Fly Slip": 145,
    "Leg Slip": -205,
    "Backward Point": 98,
    "Forward Point": 72,
    "Cover Point": 58,
    "Backward Square Leg": -240,
    "Short Leg": -258,
    "Silly Point": 95,
    "Silly Mid Off": 28,
    "Silly Mid On": -28,
    "Deep Fine Leg": -210,
    "Long Leg": -220,
    "Deep Backward Square": -235,
    "Deep Forward Square": -272,
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

# The legacy table used the opposite Y direction. Convert all legacy-only
# positions once, then apply the canonical coordinates above.
for _position_defaults in POSITION_DEFAULTS.values():
    _position_defaults["y"] = -_position_defaults["y"]

for _position_name, (_position_x, _position_y) in RIGHT_HANDED_FIELD_POSITIONS.items():
    if _position_name in POSITION_DEFAULTS:
        POSITION_DEFAULTS[_position_name]["x"] = _position_x
        POSITION_DEFAULTS[_position_name]["y"] = _position_y

DEPTH_SCALE = {"close": 0.45, "inner ring": 0.62, "deep": 0.82, "boundary": 0.95}


def normalize_handedness(batter_handedness):
    return normalize_handedness_value(batter_handedness)


def mirror_angle_for_handedness(angle, batter_handedness):
    if angle is None:
        return None
    signed = float(angle) % 360
    if signed > 180:
        signed -= 360
    mirrored = mirror_signed_angle_for_handedness(signed, batter_handedness)
    return (mirrored + 360) % 360


def _signed_shot_angle(angle):
    if angle is None:
        return None
    signed = float(angle) % 360
    if signed > 180:
        signed -= 360
    return signed


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


def get_simple_8_zone(angle, batter_handedness="right"):
    return angle_to_field_zone(_signed_shot_angle(angle), batter_handedness)


def get_detailed_field_zone(angle, batter_handedness="right"):
    return angle_to_field_zone(_signed_shot_angle(angle), batter_handedness)


def classify_shot_zone(angle, batter_handedness="right"):
    return get_simple_8_zone(angle, batter_handedness)


def get_field_zone_label(angle, batter_handedness="right"):
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
    batter_handedness="right",
    mode="Use last part of trajectory",
    manual_contact_frame=None,
):
    """Internal trajectory context for field history; not rendered as a map in the UI."""
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
    selected_depth = depth or defaults.get("depth", "inner ring")
    angle = ZONE_ANGLES.get(position, ZONE_ANGLES.get(defaults.get("zone", "Cover"), 55))
    radius = DEPTH_SCALE.get(selected_depth, 0.62)
    defaults["angle"] = float(angle)
    defaults["radius"] = float(radius)
    defaults["x"], defaults["y"] = polar_to_xy(angle, radius)
    defaults["depth"] = selected_depth
    return defaults


def _fielder(name, position, depth=None):
    defaults = get_position_defaults(position, depth)
    return {
        "name": name,
        "position": position,
        "zone": defaults["zone"],
        "depth": defaults["depth"],
        "angle": defaults["angle"],
        "radius": defaults["radius"],
        "x": defaults["x"],
        "y": defaults["y"],
    }


LEGACY_PRESET_MAP = {
    "Attacking Test Field": "Attacking Pace",
    "Defensive ODI Field": "Defensive Pace",
    "T20 Ring Field": "T20 Death Overs",
    "Off Side Trap": "Attacking Pace",
    "Leg Side Trap": "Defensive Pace",
    "Spin Attacking Field": "Spin Attack",
    "Pace Slip Field": "Attacking Pace",
}


def create_default_fielders(preset_name="Balanced", batter_handedness="right"):
    preset_name = LEGACY_PRESET_MAP.get(preset_name, preset_name)
    if preset_name in PRESET_NAMES:
        return [ensure_fielder_polar(dict(item)) for item in create_preset_fielders(preset_name)]
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


def create_default_umpires():
    """Return separately stored RH angle/radius umpire positions."""
    return [ensure_umpire_polar(umpire_from_name(name)) for name in UMPIRE_POSITIONS_RH]


def upgrade_field_setup_coordinates(field_setup):
    """Upgrade legacy coordinates to angle/radius without flipping saved RH data."""
    if not isinstance(field_setup, dict):
        return field_setup
    field_setup["batter_handedness"] = normalize_handedness(
        field_setup.get("batter_handedness")
    )
    version = field_setup.get("coordinate_system_version", 1)
    if version < 2:
        for fielder in field_setup.get("fielders", []):
            try:
                fielder["y"] = -float(fielder.get("y", 0))
            except (TypeError, ValueError):
                continue
        field_setup["coordinate_system_version"] = 2
        version = 2
    for fielder in field_setup.get("fielders", []):
        ensure_fielder_polar(fielder)
    for umpire in field_setup.get("umpires", []):
        ensure_umpire_polar(umpire)
    if version < FIELD_COORDINATE_SYSTEM_VERSION:
        field_setup["coordinate_system_version"] = FIELD_COORDINATE_SYSTEM_VERSION
    field_setup.setdefault("umpires", create_default_umpires())
    return field_setup


def find_nearest_fielder(
    shot_angle,
    fielders,
    batter_handedness="right",
):
    if shot_angle is None or not fielders:
        return None
    target_signed = _signed_shot_angle(shot_angle)
    target_x = math.sin(math.radians(target_signed))
    target_y = -math.cos(math.radians(target_signed))
    best_fielder = None
    best_distance = None
    for fielder in fielders:
        ensure_fielder_polar(fielder)
        fielder_display_angle = display_angle_for_handedness(
            fielder["angle"],
            batter_handedness,
        )
        fielder_x, fielder_y = polar_to_xy(fielder_display_angle, fielder["radius"])
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
        return upgrade_field_setup_coordinates(st.session_state["current_field_setup"])
    saved_setup = load_field_setup()
    if saved_setup is not None:
        upgrade_field_setup_coordinates(saved_setup)
        st.session_state["current_field_setup"] = saved_setup
        return saved_setup
    default_setup = {
        "preset": "Balanced",
        "batter_handedness": "right",
        "bowler_arm": "Right-arm bowler",
        "camera_view": "Behind bowler",
        "fielders": create_default_fielders("Balanced", "right"),
        "umpires": create_default_umpires(),
        "coordinate_system_version": FIELD_COORDINATE_SYSTEM_VERSION,
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


FIELD_ZONES = SIMPLE_FIELD_ZONES
