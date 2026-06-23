"""Cricket field geometry: angle/radius source of truth with display rotation."""

import math

# Shared visual tilt for pitch, labels, fielders, and zones (0 = straight up).
PITCH_AXIS_DEGREES = 0
DEFAULT_VISUAL_ROTATION = PITCH_AXIS_DEGREES

FIELD_POSITION_ANGLES_RH = {
    "Straight": {"angle": 0, "radius": 0.88},
    "Long Off": {"angle": 18, "radius": 0.88},
    "Long On": {"angle": -18, "radius": 0.88},
    "Mid Off": {"angle": 28, "radius": 0.55},
    "Mid On": {"angle": -28, "radius": 0.55},
    "Extra Cover": {"angle": 42, "radius": 0.62},
    "Cover": {"angle": 55, "radius": 0.68},
    "Deep Cover": {"angle": 55, "radius": 0.90},
    "Point": {"angle": 90, "radius": 0.68},
    "Deep Point": {"angle": 90, "radius": 0.90},
    "Gully": {"angle": 120, "radius": 0.48},
    "Third Man": {"angle": 145, "radius": 0.88},
    "Wicket Keeper": {"angle": 180, "radius": 0.30},
    "First Slip": {"angle": 155, "radius": 0.36},
    "Second Slip": {"angle": 150, "radius": 0.40},
    "Third Slip": {"angle": 145, "radius": 0.44},
    "Fine Leg": {"angle": -145, "radius": 0.88},
    "Short Fine Leg": {"angle": -140, "radius": 0.48},
    "Square Leg": {"angle": -90, "radius": 0.65},
    "Deep Square Leg": {"angle": -90, "radius": 0.90},
    "Mid Wicket": {"angle": -55, "radius": 0.68},
    "Deep Mid Wicket": {"angle": -55, "radius": 0.90},
    "Cow Corner": {"angle": -38, "radius": 0.90},
    "Bowler": {"angle": 0, "radius": 0.22},
}

UMPIRE_POSITIONS_RH = {
    "Bowler's End Umpire": {"angle": 0, "radius": 0.22},
    "Square Leg Umpire": {"angle": -90, "radius": 0.58},
}

# Signed-angle zones for right-handed batters (positive = off side).
RH_ZONE_RANGES = [
    (-180, -158, "Fine Leg"),
    (-158, -122, "Square Leg"),
    (-122, -72, "Mid Wicket"),
    (-72, -38, "Mid On"),
    (-38, -8, "Long On"),
    (-8, 8, "Straight"),
    (8, 38, "Long Off"),
    (38, 72, "Mid Off"),
    (72, 100, "Cover"),
    (100, 122, "Point"),
    (122, 158, "Third Man"),
    (158, 180, "Fine Leg"),
]


def is_left_handed(handedness):
    return str(handedness or "").lower().startswith("left")


def normalize_handedness_label(handedness):
    return "Left-handed" if is_left_handed(handedness) else "Right-handed"


def mirror_angle_for_handedness(angle_deg, handedness):
    """Mirror signed cricket angle for display; stored data stays RH-relative."""
    return display_angle_for_handedness(angle_deg, handedness)


def display_angle_for_handedness(angle_deg, handedness):
    """RH cricket angle -> display angle before screen conversion."""
    if angle_deg is None:
        return None
    angle = float(angle_deg)
    if is_left_handed(handedness):
        return -angle
    return angle


def apply_visual_rotation(angle_deg, visual_rotation_deg=DEFAULT_VISUAL_ROTATION):
    """Tilt the rendered field without changing stored cricket meaning."""
    if angle_deg is None:
        return None
    return float(angle_deg) + float(visual_rotation_deg)


def polar_to_xy(angle_deg, radius):
    """Convert display angle/radius to normalized x/y.

    x = sin(angle) * radius  (positive x = off side for RH)
    y = -cos(angle) * radius (angle 0 points straight down the ground)
    """
    radians = math.radians(float(angle_deg))
    radius = float(radius)
    x = math.sin(radians) * radius
    y = -math.cos(radians) * radius
    return x, y


def xy_to_polar(x, y):
    """Convert normalized RH x/y to signed cricket angle and radius."""
    x = float(x)
    y = float(y)
    radius = math.hypot(x, y)
    if radius < 1e-9:
        return 0.0, 0.0
    angle = math.degrees(math.atan2(x, -y))
    return angle, radius


def polar_to_screen(angle_deg, radius, handedness="Right-handed", visual_rotation_deg=DEFAULT_VISUAL_ROTATION):
    """Display-only normalized coordinates in roughly -1..1.

    RH: display_angle = cricket_angle
    LH: display_angle = -cricket_angle
    Then: x = sin(display_angle) * radius, y = -cos(display_angle) * radius
    """
    display_angle = display_angle_for_handedness(angle_deg, handedness)
    if visual_rotation_deg:
        display_angle = apply_visual_rotation(display_angle, visual_rotation_deg)
    return polar_to_xy(display_angle, radius)


def screen_to_polar(x, y, handedness="Right-handed", visual_rotation_deg=DEFAULT_VISUAL_ROTATION):
    """Convert display x/y back to RH cricket angle/radius."""
    display_angle, radius = xy_to_polar(x, y)
    cricket_angle = display_angle
    if visual_rotation_deg:
        cricket_angle = display_angle - visual_rotation_deg
    cricket_angle = display_angle_for_handedness(cricket_angle, handedness)
    if cricket_angle > 180:
        cricket_angle -= 360
    elif cricket_angle <= -180:
        cricket_angle += 360
    return cricket_angle, min(max(radius, 0.0), 1.0)


def angle_to_field_zone(angle_deg, handedness="Right-handed"):
    """Map a signed RH cricket angle to a zone label."""
    if angle_deg is None:
        return "Unknown"
    angle = float(angle_deg)
    if is_left_handed(handedness):
        angle = -angle
    for start, end, zone in RH_ZONE_RANGES:
        if start <= angle < end:
            return zone
    return "Unknown"


def get_side_labels(handedness):
    if is_left_handed(handedness):
        return {"left": "Off side", "right": "Leg side"}
    return {"left": "Leg side", "right": "Off side"}


def get_position_spec(position_name):
    return FIELD_POSITION_ANGLES_RH.get(position_name, FIELD_POSITION_ANGLES_RH["Cover"])


def fielder_from_position(name, position, angle=None, radius=None):
    spec = get_position_spec(position)
    return {
        "name": name,
        "position": position,
        "zone": position,
        "angle": float(angle if angle is not None else spec["angle"]),
        "radius": float(radius if radius is not None else spec["radius"]),
    }


def umpire_from_name(name):
    spec = UMPIRE_POSITIONS_RH.get(name, UMPIRE_POSITIONS_RH["Bowler's End Umpire"])
    return {
        "name": name,
        "angle": float(spec["angle"]),
        "radius": float(spec["radius"]),
    }


def ensure_fielder_polar(fielder):
    """Ensure fielder dict has RH angle/radius; upgrade legacy x/y if needed."""
    if not isinstance(fielder, dict):
        return fielder
    if "angle" in fielder and "radius" in fielder:
        fielder["angle"] = float(fielder["angle"])
        fielder["radius"] = float(fielder["radius"])
    elif "x" in fielder and "y" in fielder:
        angle, radius = xy_to_polar(fielder["x"], fielder["y"])
        fielder["angle"] = angle
        fielder["radius"] = radius
    else:
        position = fielder.get("position") or fielder.get("zone") or "Cover"
        spec = get_position_spec(position)
        fielder["angle"] = float(spec["angle"])
        fielder["radius"] = float(spec["radius"])
    fielder["x"], fielder["y"] = polar_to_xy(fielder["angle"], fielder["radius"])
    return fielder


def ensure_umpire_polar(umpire):
    if not isinstance(umpire, dict):
        return umpire
    if "angle" in umpire and "radius" in umpire:
        umpire["angle"] = float(umpire["angle"])
        umpire["radius"] = float(umpire["radius"])
    elif "x" in umpire and "y" in umpire:
        angle, radius = xy_to_polar(umpire["x"], umpire["y"])
        umpire["angle"] = angle
        umpire["radius"] = radius
    else:
        spec = UMPIRE_POSITIONS_RH.get(umpire.get("name"), UMPIRE_POSITIONS_RH["Bowler's End Umpire"])
        umpire["angle"] = float(spec["angle"])
        umpire["radius"] = float(spec["radius"])
    umpire["x"], umpire["y"] = polar_to_xy(umpire["angle"], umpire["radius"])
    return umpire


def fielder_display_xy(fielder, handedness="Right-handed", visual_rotation_deg=DEFAULT_VISUAL_ROTATION):
    ensure_fielder_polar(fielder)
    return polar_to_screen(fielder["angle"], fielder["radius"], handedness, visual_rotation_deg)


def umpire_display_xy(umpire, handedness="Right-handed", visual_rotation_deg=DEFAULT_VISUAL_ROTATION):
    ensure_umpire_polar(umpire)
    return polar_to_screen(umpire["angle"], umpire["radius"], handedness, visual_rotation_deg)


def pitch_polygon_corners(
    handedness="Right-handed",
    visual_rotation_deg=DEFAULT_VISUAL_ROTATION,
    half_length=0.24,
    half_width=0.055,
):
    """Return four pitch corners aligned with the straight (bowler-to-batter) axis."""
    bowler_far = polar_to_screen(0, half_length, handedness, visual_rotation_deg)
    striker_far = polar_to_screen(180, half_length, handedness, visual_rotation_deg)
    perp_x, perp_y = polar_to_screen(90, half_width, handedness, visual_rotation_deg)
    return [
        (bowler_far[0] - perp_x, bowler_far[1] - perp_y),
        (bowler_far[0] + perp_x, bowler_far[1] + perp_y),
        (striker_far[0] + perp_x, striker_far[1] + perp_y),
        (striker_far[0] - perp_x, striker_far[1] - perp_y),
    ]


def striker_crease_xy(handedness="Right-handed", visual_rotation_deg=DEFAULT_VISUAL_ROTATION):
    return polar_to_screen(180, 0.11, handedness, visual_rotation_deg)


def bowler_end_xy(handedness="Right-handed", visual_rotation_deg=DEFAULT_VISUAL_ROTATION):
    return polar_to_screen(0, 0.08, handedness, visual_rotation_deg)
