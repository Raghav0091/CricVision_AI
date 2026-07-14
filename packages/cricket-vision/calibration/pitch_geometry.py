from math import hypot

from .environment_context import CalibrationQuality, EnvironmentContext


Point = tuple[float, float]


def build_pitch_axis(non_striker_center: Point, striker_center: Point) -> tuple[Point, Point]:
    if non_striker_center == striker_center:
        raise ValueError("Stump centers must be distinct.")
    return non_striker_center, striker_center


def build_pitch_corridor(axis: tuple[Point, Point], width_px: float) -> tuple[Point, ...]:
    if width_px <= 0:
        raise ValueError("Pitch corridor width must be positive.")
    (x1, y1), (x2, y2) = axis
    length = hypot(x2 - x1, y2 - y1)
    if length == 0:
        raise ValueError("Pitch axis must have length.")
    offset_x = -(y2 - y1) / length * width_px / 2
    offset_y = (x2 - x1) / length * width_px / 2
    return (
        (x1 + offset_x, y1 + offset_y),
        (x2 + offset_x, y2 + offset_y),
        (x2 - offset_x, y2 - offset_y),
        (x1 - offset_x, y1 - offset_y),
    )


def build_environment_context(
    frame_width: int,
    frame_height: int,
    non_striker_center: Point,
    striker_center: Point,
    corridor_width_px: float,
    quality: CalibrationQuality,
) -> EnvironmentContext:
    axis = build_pitch_axis(non_striker_center, striker_center)
    corridor = build_pitch_corridor(axis, corridor_width_px)
    return EnvironmentContext(
        frame_width=frame_width,
        frame_height=frame_height,
        striker_stump_center=striker_center,
        non_striker_stump_center=non_striker_center,
        pitch_axis=axis,
        pitch_corridor=corridor,
        quality=quality,
    )
