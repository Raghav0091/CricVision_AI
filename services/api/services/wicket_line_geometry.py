"""Small, dependency-light geometry helpers for wicket landmark extraction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

    def as_mapping(self) -> dict[str, float]:
        return {"x": float(self.x), "y": float(self.y)}


@dataclass(frozen=True)
class LineSegment:
    start: Point2D
    end: Point2D

    @property
    def length(self) -> float:
        return math.hypot(self.end.x - self.start.x, self.end.y - self.start.y)

    @property
    def angle_degrees(self) -> float:
        angle = math.degrees(
            math.atan2(self.end.y - self.start.y, self.end.x - self.start.x)
        )
        return angle % 180.0

    @property
    def midpoint(self) -> Point2D:
        return Point2D(
            (self.start.x + self.end.x) / 2.0,
            (self.start.y + self.end.y) / 2.0,
        )

    def ordered_by_y(self) -> "LineSegment":
        return self if self.start.y <= self.end.y else LineSegment(self.end, self.start)

    def ordered_by_x(self) -> "LineSegment":
        return self if self.start.x <= self.end.x else LineSegment(self.end, self.start)

    def as_mapping(self) -> dict[str, object]:
        return {"start": self.start.as_mapping(), "end": self.end.as_mapping()}


def angular_distance_degrees(first: float, second: float) -> float:
    """Return unsigned orientation difference for unoriented 2D lines."""

    delta = abs((first - second) % 180.0)
    return min(delta, 180.0 - delta)


def is_vertical(line: LineSegment, tolerance_degrees: float = 14.0) -> bool:
    return angular_distance_degrees(line.angle_degrees, 90.0) <= tolerance_degrees


def is_horizontal(line: LineSegment, tolerance_degrees: float = 12.0) -> bool:
    return angular_distance_degrees(line.angle_degrees, 0.0) <= tolerance_degrees


def normalized_line_equation(line: LineSegment) -> tuple[float, float, float]:
    """Return ``ax + by + c = 0`` with unit normal and stable sign."""

    dx = line.end.x - line.start.x
    dy = line.end.y - line.start.y
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        raise ValueError("A line segment must have non-zero length.")
    a, b = dy / norm, -dx / norm
    c = -(a * line.start.x + b * line.start.y)
    if a < 0 or (abs(a) <= 1e-12 and b < 0):
        a, b, c = -a, -b, -c
    return float(a), float(b), float(c)


def line_intersection(first: LineSegment, second: LineSegment) -> Point2D | None:
    a1, b1, c1 = normalized_line_equation(first)
    a2, b2, c2 = normalized_line_equation(second)
    determinant = a1 * b2 - a2 * b1
    if abs(determinant) <= 1e-8:
        return None
    return Point2D(
        (b1 * c2 - b2 * c1) / determinant,
        (c1 * a2 - c2 * a1) / determinant,
    )


def point_line_distance(point: Point2D, line: LineSegment) -> float:
    a, b, c = normalized_line_equation(line)
    return abs(a * point.x + b * point.y + c)


def translate_line(line: LineSegment, dx: float, dy: float) -> LineSegment:
    return LineSegment(
        Point2D(line.start.x + dx, line.start.y + dy),
        Point2D(line.end.x + dx, line.end.y + dy),
    )


def robust_location(values: Sequence[float]) -> tuple[float, float]:
    """Return median and robust sigma (scaled MAD)."""

    if not values:
        raise ValueError("At least one value is required.")
    data = np.asarray(values, dtype=np.float64)
    median = float(np.median(data))
    sigma = float(1.4826 * np.median(np.abs(data - median)))
    return median, sigma


def inlier_indices(values: Sequence[float], *, minimum_tolerance: float = 1.5) -> list[int]:
    if not values:
        return []
    median, sigma = robust_location(values)
    tolerance = max(minimum_tolerance, 3.0 * sigma)
    return [index for index, value in enumerate(values) if abs(value - median) <= tolerance]


def merge_collinear_segments(
    lines: Iterable[LineSegment],
    *,
    orientation: str,
    position_tolerance_px: float,
) -> list[LineSegment]:
    """Merge fragmented near-collinear Hough segments deterministically."""

    candidates = [
        line.ordered_by_y() if orientation == "vertical" else line.ordered_by_x()
        for line in lines
        if (is_vertical(line) if orientation == "vertical" else is_horizontal(line))
    ]
    candidates.sort(key=lambda item: item.midpoint.x if orientation == "vertical" else item.midpoint.y)
    groups: list[list[LineSegment]] = []
    for line in candidates:
        position = line.midpoint.x if orientation == "vertical" else line.midpoint.y
        if not groups:
            groups.append([line])
            continue
        previous = groups[-1]
        previous_position = float(np.median([
            item.midpoint.x if orientation == "vertical" else item.midpoint.y
            for item in previous
        ]))
        if abs(position - previous_position) <= position_tolerance_px:
            previous.append(line)
        else:
            groups.append([line])

    merged: list[LineSegment] = []
    for group in groups:
        if orientation == "vertical":
            x = float(np.median([item.midpoint.x for item in group]))
            merged.append(LineSegment(Point2D(x, min(item.start.y for item in group)), Point2D(x, max(item.end.y for item in group))))
        else:
            y = float(np.median([item.midpoint.y for item in group]))
            merged.append(LineSegment(Point2D(min(item.start.x for item in group), y), Point2D(max(item.end.x for item in group), y)))
    return merged
