"""Authoritative CricVision virtual cricket-pitch geometry in metres.

This module is the single backend source of truth for permanent pitch
dimensions. Legacy calibration helpers at the bottom retain their historical
longitudinal-X/lateral-Y ordering through explicit adapters.

Coordinate convention:
    origin: centre of the bowler-end wicket at ground level
    +X: lateral right when viewed from bowler end toward striker end
    +Y: from bowler end toward striker end
    +Z: upward from the pitch surface
"""

from __future__ import annotations

from dataclasses import dataclass


VIRTUAL_PITCH_MODEL_VERSION = "v1"
PITCH_LENGTH_M = 20.12
PITCH_WIDTH_M = 3.05
WICKET_WIDTH_M = 0.2286
STUMP_HEIGHT_M = 0.7112
STUMP_DIAMETER_MIN_M = 0.0350
STUMP_DIAMETER_MAX_M = 0.0381
BOWLING_CREASE_LENGTH_M = 2.64
POPPING_CREASE_OFFSET_M = 1.22
RETURN_CREASE_OFFSET_M = 1.32
CREASE_LINE_WIDTH_M = 0.0254
DISPLAY_DECIMAL_PLACES = 3

# Compatibility aliases for the established Calibration V2 API.
DEFAULT_PITCH_LENGTH_M = PITCH_LENGTH_M
DEFAULT_WICKET_WIDTH_M = WICKET_WIDTH_M
DEFAULT_WICKET_HEIGHT_M = STUMP_HEIGHT_M
# Regulation wicket width is measured across the outside edges of the outer
# stumps. A 1.5 inch stump diameter therefore puts the outer stump centres
# 95.25 mm from the wicket centre, not at +/- wicket_width / 2.
DEFAULT_STUMP_DIAMETER_M = STUMP_DIAMETER_MAX_M
DEFAULT_PITCH_WIDTH_M = PITCH_WIDTH_M
DEFAULT_POPPING_CREASE_DISTANCE_M = POPPING_CREASE_OFFSET_M

LEFT_RIGHT_CONVENTION = (
    "Legacy Calibration V2 ordering: world left is negative Y and world "
    "right is positive Y when viewed toward the striker."
)
CANONICAL_LEFT_RIGHT_CONVENTION = (
    "World left/right is viewed from the bowler-end wicket toward the "
    "striker-end wicket: left is negative X and right is positive X."
)
CANONICAL_COORDINATE_SYSTEM = (
    "Right-handed metres: origin at the bowler-end middle-stump base; "
    "+x is camera-neutral right across the pitch when looking toward the "
    "striker; +y points toward the striker; +z is upward."
)


@dataclass(frozen=True)
class CricketPitchDimensions:
    pitch_length_m: float = DEFAULT_PITCH_LENGTH_M
    wicket_width_m: float = DEFAULT_WICKET_WIDTH_M
    wicket_height_m: float = DEFAULT_WICKET_HEIGHT_M
    stump_diameter_m: float = DEFAULT_STUMP_DIAMETER_M
    pitch_width_m: float = DEFAULT_PITCH_WIDTH_M
    popping_crease_distance_m: float = DEFAULT_POPPING_CREASE_DISTANCE_M
    bowling_crease_length_m: float = BOWLING_CREASE_LENGTH_M
    return_crease_offset_m: float = RETURN_CREASE_OFFSET_M

    def validate(self) -> None:
        if self.pitch_length_m <= 0:
            raise ValueError("Pitch length must be positive.")
        if self.wicket_width_m <= 0:
            raise ValueError("Wicket width must be positive.")
        if self.wicket_height_m <= 0:
            raise ValueError("Wicket height must be positive.")
        if not 0 < self.stump_diameter_m < self.wicket_width_m / 2:
            raise ValueError("Stump diameter is invalid for the wicket width.")
        if self.pitch_width_m < self.wicket_width_m:
            raise ValueError("Pitch width must exceed wicket width.")
        if not 0 < self.popping_crease_distance_m < self.pitch_length_m / 2:
            raise ValueError("Popping crease distance is invalid.")
        if self.bowling_crease_length_m <= 0:
            raise ValueError("Bowling crease length must be positive.")
        if self.return_crease_offset_m <= 0:
            raise ValueError("Return crease offset must be positive.")


def canonical_to_calibration_world(
    x_m: float,
    y_m: float,
    z_m: float,
) -> tuple[float, float, float]:
    """Adapt canonical lateral/longitudinal/up into Calibration V2 ordering."""
    return (y_m, x_m, z_m)


def calibration_to_canonical_world(
    longitudinal_x_m: float,
    lateral_y_m: float,
    z_m: float,
) -> tuple[float, float, float]:
    """Adapt Calibration V2 longitudinal/lateral/up into canonical ordering."""
    return (lateral_y_m, longitudinal_x_m, z_m)


def stump_base_world_points(
    geometry: CricketPitchDimensions,
) -> dict[str, tuple[float, float, float]]:
    geometry.validate()
    left, middle, right = stump_lateral_positions_m(geometry)
    return {
        # Legacy Calibration V2 ordering: longitudinal X, lateral Y, up Z.
        "bowler_left_stump_base": (0.0, left, 0.0),
        "bowler_middle_stump_base": (0.0, middle, 0.0),
        "bowler_right_stump_base": (0.0, right, 0.0),
        "striker_left_stump_base": (
            geometry.pitch_length_m,
            left,
            0.0,
        ),
        "striker_middle_stump_base": (
            geometry.pitch_length_m,
            0.0,
            0.0,
        ),
        "striker_right_stump_base": (
            geometry.pitch_length_m,
            right,
            0.0,
        ),
    }


def canonical_stump_base_world_points(
    geometry: CricketPitchDimensions,
) -> dict[str, tuple[float, float, float]]:
    """Return six stump bases in canonical lateral-X/longitudinal-Y order."""
    return {
        landmark_id: calibration_to_canonical_world(*point)
        for landmark_id, point in stump_base_world_points(geometry).items()
    }


def stump_lateral_positions_m(
    geometry: CricketPitchDimensions,
) -> tuple[float, float, float]:
    """Return left/middle/right stump centre positions in world Y.

    ``wicket_width_m`` is the outside-to-outside wicket width. Subtracting one
    stump diameter gives the distance between the two outer stump centres.
    The three stump centres are treated as equally spaced.
    """
    geometry.validate()
    outer_centre = (geometry.wicket_width_m - geometry.stump_diameter_m) / 2
    return (-outer_centre, 0.0, outer_centre)


def wicket_landmark_world_points(
    geometry: CricketPitchDimensions,
) -> dict[str, tuple[float, float, float]]:
    """Return explicit 3D correspondences for six bases and six stump tops.

    A top is the top of the stump body at ``wicket_height_m``. Bail height is
    deliberately not included.
    """
    bases = stump_base_world_points(geometry)
    points = dict(bases)
    for landmark_id, (x_m, y_m, _) in bases.items():
        points[landmark_id.replace("_base", "_top")] = (
            x_m,
            y_m,
            geometry.wicket_height_m,
        )
    return points


def canonical_wicket_landmark_world_points(
    geometry: CricketPitchDimensions,
) -> dict[str, tuple[float, float, float]]:
    """Return six bases and six stump tops in canonical coordinates."""
    return {
        landmark_id: calibration_to_canonical_world(*point)
        for landmark_id, point in wicket_landmark_world_points(geometry).items()
    }


def standard_ground_reference_world_points(
    geometry: CricketPitchDimensions,
) -> dict[str, tuple[float, float, float]]:
    """Known crease/pitch-edge intersections for optional manual placement.

    These world coordinates are metric facts from the configured pitch
    geometry. Their matching image pixels must still be observed and placed by
    the user; this function does not invent image correspondences.
    """
    geometry.validate()
    half_pitch = geometry.pitch_width_m / 2
    bowler_crease_x = geometry.popping_crease_distance_m
    striker_crease_x = (
        geometry.pitch_length_m - geometry.popping_crease_distance_m
    )
    return {
        "bowler_end_left_ground_reference": (
            bowler_crease_x,
            -half_pitch,
            0.0,
        ),
        "bowler_end_right_ground_reference": (
            bowler_crease_x,
            half_pitch,
            0.0,
        ),
        "striker_end_left_ground_reference": (
            striker_crease_x,
            -half_pitch,
            0.0,
        ),
        "striker_end_right_ground_reference": (
            striker_crease_x,
            half_pitch,
            0.0,
        ),
    }


def virtual_pitch_ground_lines(
    geometry: CricketPitchDimensions,
) -> dict[str, tuple[tuple[float, float], ...]]:
    geometry.validate()
    left_stump, _, right_stump = stump_lateral_positions_m(geometry)
    half_pitch = geometry.pitch_width_m / 2
    striker_x = geometry.pitch_length_m
    return {
        "pitch_outline": (
            (0.0, -half_pitch),
            (striker_x, -half_pitch),
            (striker_x, half_pitch),
            (0.0, half_pitch),
            (0.0, -half_pitch),
        ),
        "pitch_centreline": ((0.0, 0.0), (striker_x, 0.0)),
        "bowler_wicket_width": (
            (0.0, left_stump),
            (0.0, right_stump),
        ),
        "striker_wicket_width": (
            (striker_x, left_stump),
            (striker_x, right_stump),
        ),
        "bowler_popping_crease": (
            (geometry.popping_crease_distance_m, -half_pitch),
            (geometry.popping_crease_distance_m, half_pitch),
        ),
        "striker_popping_crease": (
            (
                striker_x - geometry.popping_crease_distance_m,
                -half_pitch,
            ),
            (
                striker_x - geometry.popping_crease_distance_m,
                half_pitch,
            ),
        ),
    }
