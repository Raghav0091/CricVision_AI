"""Synthetic trajectory replay visualization (OpenCV/NumPy only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from Backends.src.utils.cv2_loader import cv2

MIN_REPLAY_POINTS = 3

OUTFIELD_GREEN = (42, 110, 42)
PITCH_STRIP = (168, 196, 132)
PITCH_LANE = (188, 210, 150)
STUMP_WOOD = (210, 210, 210)
STUMP_OUTLINE = (40, 40, 40)
TRAJECTORY_COLOR = (0, 0, 220)
SAMPLE_DOT_COLOR = (255, 255, 255)
BOUNCE_COLOR = (0, 165, 255)
CARD_BG = (28, 28, 28)
CARD_BORDER = (70, 70, 70)
CARD_TEXT = (245, 245, 245)
CARD_MUTED = (170, 170, 170)


def _coerce_point(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            try:
                return int(value["x"]), int(value["y"])
            except (TypeError, ValueError):
                return None
        center = value.get("center")
        if center is not None:
            return _coerce_point(center)
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _valid_trajectory_points(trajectory_points: Any) -> list[tuple[int, int]]:
    if not isinstance(trajectory_points, (list, tuple)):
        return []
    points: list[tuple[int, int]] = []
    for item in trajectory_points:
        point = _coerce_point(item)
        if point is not None:
            points.append(point)
    return points


def _format_rate(value: Any) -> str:
    if value is None:
        return "Unknown"
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return "Unknown"
    if rate <= 1.0:
        rate *= 100.0
    return f"{rate:.1f}%"


def _format_int(value: Any) -> str:
    try:
        if value is None:
            return "0"
        return str(int(value))
    except (TypeError, ValueError):
        return "0"


def _pitch_lane_rect(width: int, height: int) -> tuple[int, int, int, int]:
    lane_left = int(width * 0.34)
    lane_right = int(width * 0.66)
    lane_top = int(height * 0.10)
    lane_bottom = int(height * 0.82)
    return lane_left, lane_top, lane_right, lane_bottom


def _draw_pitch_base(canvas: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    canvas[:] = OUTFIELD_GREEN
    lane_left, lane_top, lane_right, lane_bottom = _pitch_lane_rect(width, height)
    cv2.rectangle(canvas, (lane_left, lane_top), (lane_right, lane_bottom), PITCH_STRIP, -1)

    overlay = canvas.copy()
    inner_left = int(width * 0.42)
    inner_right = int(width * 0.58)
    cv2.rectangle(overlay, (inner_left, lane_top), (inner_right, lane_bottom), PITCH_LANE, -1)
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

    crease_y_bat = lane_bottom - int((lane_bottom - lane_top) * 0.08)
    crease_y_bowl = lane_top + int((lane_bottom - lane_top) * 0.08)
    for crease_y in (crease_y_bat, crease_y_bowl):
        cv2.line(canvas, (lane_left, crease_y), (lane_right, crease_y), (230, 230, 230), 1)

    return lane_left, lane_top, lane_right, lane_bottom


def _draw_stumps(canvas: np.ndarray, center_x: int, base_y: int, *, scale: float = 1.0) -> None:
    stump_width = max(3, int(5 * scale))
    stump_height = max(18, int(28 * scale))
    gap = max(4, int(7 * scale))
    offsets = (-gap, 0, gap)
    for offset in offsets:
        x1 = center_x + offset - stump_width // 2
        x2 = x1 + stump_width
        y1 = base_y - stump_height
        y2 = base_y
        cv2.rectangle(canvas, (x1, y1), (x2, y2), STUMP_OUTLINE, -1)
        cv2.rectangle(canvas, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), STUMP_WOOD, -1)


def _normalize_points_to_lane(
    points: list[tuple[int, int]],
    lane_rect: tuple[int, int, int, int],
    *,
    extra_point: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    lane_left, lane_top, lane_right, lane_bottom = lane_rect
    lane_width = max(lane_right - lane_left, 1)
    lane_height = max(lane_bottom - lane_top, 1)
    margin_x = int(lane_width * 0.12)
    margin_y = int(lane_height * 0.08)
    target_left = lane_left + margin_x
    target_right = lane_right - margin_x
    target_top = lane_top + margin_y
    target_bottom = lane_bottom - margin_y
    target_width = max(target_right - target_left, 1)
    target_height = max(target_bottom - target_top, 1)

    source_points = list(points)
    if extra_point is not None:
        source_points.append(extra_point)

    xs = [point[0] for point in source_points]
    ys = [point[1] for point in source_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1)
    span_y = max(max_y - min_y, 1)

    mapped: list[tuple[int, int]] = []
    for x, y in points:
        norm_x = target_left + int(((x - min_x) / span_x) * target_width)
        norm_y = target_top + int(((y - min_y) / span_y) * target_height)
        mapped.append((norm_x, norm_y))
    return mapped


def _normalize_single_point_to_lane(
    point: tuple[int, int],
    source_points: list[tuple[int, int]],
    lane_rect: tuple[int, int, int, int],
) -> tuple[int, int]:
    mapped = _normalize_points_to_lane(source_points, lane_rect, extra_point=point)
    if not mapped:
        return point
    source_index = len(source_points)
    if source_index < len(mapped):
        return mapped[source_index]
    return mapped[-1]


def _metric_cards(health: dict[str, Any] | None) -> list[tuple[str, str]]:
    health = health or {}
    return [
        ("Tracking Quality", str(health.get("overall_tracking_quality") or "Unknown")),
        ("Detection Rate", _format_rate(health.get("ball_detection_rate"))),
        ("Tracking Rate", _format_rate(health.get("ball_tracking_rate"))),
        ("Raw Ball Detections", _format_int(health.get("raw_ball_detections"))),
        ("Selected Ball Points", _format_int(health.get("selected_ball_points"))),
        ("Speed", "Not calibrated"),
        ("Swing", "Unknown"),
        ("Spin", "Unknown"),
    ]


def _draw_metric_cards(canvas: np.ndarray, health: dict[str, Any] | None, width: int, height: int) -> None:
    cards = _metric_cards(health)

    card_top = int(height * 0.84)
    card_height = int(height * 0.015)
    row_gap = int(height * 0.006)
    col_gap = int(width * 0.02)
    card_width = int((width - (3 * col_gap)) / 2)
    card_pad_x = int(width * 0.03)
    card_pad_y = int(height * 0.008)

    for index, (label, value) in enumerate(cards):
        row = index // 2
        col = index % 2
        x1 = col_gap + col * (card_width + col_gap)
        y1 = card_top + row * (card_height + row_gap + card_pad_y * 3)
        x2 = x1 + card_width
        y2 = y1 + card_height + card_pad_y * 3
        cv2.rectangle(canvas, (x1, y1), (x2, y2), CARD_BG, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), CARD_BORDER, 1)
        cv2.putText(
            canvas,
            label,
            (x1 + card_pad_x, y1 + card_pad_y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            CARD_MUTED,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(value),
            (x1 + card_pad_x, y2 - card_pad_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            CARD_TEXT,
            1,
            cv2.LINE_AA,
        )


def build_trajectory_replay_image(
    trajectory_points: Any,
    bounce_point: Any = None,
    health: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
    width: int = 720,
    height: int = 1280,
) -> np.ndarray | None:
    """Render a synthetic vertical-pitch trajectory replay image."""
    points = _valid_trajectory_points(trajectory_points)
    if len(points) < MIN_REPLAY_POINTS:
        return None

    width = max(int(width), 1)
    height = max(int(height), 1)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    lane_rect = _draw_pitch_base(canvas, width, height)
    lane_left, lane_top, lane_right, lane_bottom = lane_rect
    center_x = (lane_left + lane_right) // 2

    _draw_stumps(canvas, center_x, lane_bottom - 6, scale=1.0)
    _draw_stumps(canvas, center_x, lane_top + 34, scale=0.85)

    mapped_points = _normalize_points_to_lane(points, lane_rect)
    bounce = _coerce_point(bounce_point)
    mapped_bounce = None
    if bounce is not None:
        mapped_bounce = _normalize_single_point_to_lane(bounce, points, lane_rect)

    if len(mapped_points) >= 2:
        cv2.polylines(
            canvas,
            [np.array(mapped_points, dtype=np.int32)],
            False,
            TRAJECTORY_COLOR,
            6,
            cv2.LINE_AA,
        )

    sample_stride = max(1, len(mapped_points) // 8)
    for index, point in enumerate(mapped_points):
        if index % sample_stride == 0 or index == len(mapped_points) - 1:
            cv2.circle(canvas, point, 4, SAMPLE_DOT_COLOR, -1, lineType=cv2.LINE_AA)

    if mapped_bounce is not None:
        cv2.circle(canvas, mapped_bounce, 10, BOUNCE_COLOR, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, mapped_bounce, 14, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    cv2.putText(
        canvas,
        "CricVision Trajectory Replay v1",
        (int(width * 0.08), int(height * 0.05)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Approximate image-space replay (not calibrated)",
        (int(width * 0.08), int(height * 0.075)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        CARD_MUTED,
        1,
        cv2.LINE_AA,
    )

    _draw_metric_cards(canvas, health, width, height)

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), canvas)

    return canvas
