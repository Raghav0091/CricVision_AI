"""Conservative post-tracking trajectory fitting for delivery overlays."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from statistics import median

import numpy as np


_MAX_MAIN_SEGMENT_GAP = 3
_STATIC_MOVE_PX = 1.5
_MAX_EXTRAPOLATION_FRAMES = 3
_PARTIAL_MIN_POINTS = 5
_GOOD_MIN_POINTS = 18
_MEANINGFUL_MOVEMENT_PX = 25.0
_EDGE_MARGIN_RATIO = 0.02
_EDGE_MARGIN_MIN_PX = 5


@dataclass(slots=True)
class TrajectoryFitResult:
    fitted_trajectory_points: list[tuple[int, int]]
    observed_trajectory_points: list[tuple[int, int]]
    trajectory_fit_quality: str | None
    trajectory_fit_reason: str
    start_frame: int | None
    end_frame: int | None
    best_segment_start_frame: int | None
    best_segment_end_frame: int | None
    best_segment_point_count: int
    best_segment_duration_sec: float | None
    selected_segment_score: float
    selected_segment_reason: str
    observed_point_count: int
    predicted_point_count: int
    extrapolation_used: bool
    trajectory_visualization_mode: str

    def to_dict(self) -> dict:
        return {
            "fitted_trajectory_points": self.fitted_trajectory_points,
            "observed_trajectory_points": self.observed_trajectory_points,
            "trajectory_fit_quality": self.trajectory_fit_quality,
            "trajectory_fit_reason": self.trajectory_fit_reason,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "best_segment_start_frame": self.best_segment_start_frame,
            "best_segment_end_frame": self.best_segment_end_frame,
            "best_segment_point_count": self.best_segment_point_count,
            "best_segment_duration_sec": self.best_segment_duration_sec,
            "selected_segment_score": self.selected_segment_score,
            "selected_segment_reason": self.selected_segment_reason,
            "observed_point_count": self.observed_point_count,
            "predicted_point_count": self.predicted_point_count,
            "extrapolation_used": self.extrapolation_used,
            "trajectory_visualization_mode": self.trajectory_visualization_mode,
        }


def fit_delivery_trajectory(
    accepted_points,
    *,
    frame_size: tuple[int, int],
    fps: float | None = None,
    calibration_context=None,
    delivery_track_terminated_frame: int | None = None,
) -> dict:
    """Fit a smooth, conservative path from accepted trajectory points."""
    frame_width = max(int(frame_size[0] or 0), 1)
    frame_height = max(int(frame_size[1] or 0), 1)

    cleaned = _clean_points(
        accepted_points or [],
        frame_width=frame_width,
        frame_height=frame_height,
        terminated_frame=delivery_track_terminated_frame,
    )
    if len(cleaned) < 3:
        return TrajectoryFitResult(
            fitted_trajectory_points=[],
            observed_trajectory_points=[(int(p["x"]), int(p["y"])) for p in cleaned],
            trajectory_fit_quality="Poor" if cleaned else None,
            trajectory_fit_reason="too_few_clean_points" if cleaned else "no_points",
            start_frame=cleaned[0]["frame_index"] if cleaned else None,
            end_frame=cleaned[-1]["frame_index"] if cleaned else None,
            best_segment_start_frame=cleaned[0]["frame_index"] if cleaned else None,
            best_segment_end_frame=cleaned[-1]["frame_index"] if cleaned else None,
            best_segment_point_count=len(cleaned),
            best_segment_duration_sec=_segment_duration(cleaned, fps),
            selected_segment_score=0.0,
            selected_segment_reason="too_few_clean_points" if cleaned else "no_points",
            observed_point_count=len(cleaned),
            predicted_point_count=0,
            extrapolation_used=False,
            trajectory_visualization_mode="hidden",
        ).to_dict()

    segment, segment_score, segment_reason = _best_delivery_segment(
        cleaned,
        frame_width=frame_width,
        frame_height=frame_height,
        calibration_context=calibration_context,
    )
    if len(segment) < 3:
        return TrajectoryFitResult(
            fitted_trajectory_points=[],
            observed_trajectory_points=[(int(p["x"]), int(p["y"])) for p in segment],
            trajectory_fit_quality="Poor",
            trajectory_fit_reason=segment_reason or "no_continuous_segment",
            start_frame=segment[0]["frame_index"] if segment else None,
            end_frame=segment[-1]["frame_index"] if segment else None,
            best_segment_start_frame=segment[0]["frame_index"] if segment else None,
            best_segment_end_frame=segment[-1]["frame_index"] if segment else None,
            best_segment_point_count=len(segment),
            best_segment_duration_sec=_segment_duration(segment, fps),
            selected_segment_score=round(segment_score, 3),
            selected_segment_reason=segment_reason,
            observed_point_count=len(segment),
            predicted_point_count=0,
            extrapolation_used=False,
            trajectory_visualization_mode="hidden",
        ).to_dict()

    fit_points, extrapolation_used = _fit_segment(
        segment,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    if len(fit_points) < 2:
        return TrajectoryFitResult(
            fitted_trajectory_points=[],
            observed_trajectory_points=[(int(p["x"]), int(p["y"])) for p in segment],
            trajectory_fit_quality="Poor",
            trajectory_fit_reason="fit_failed",
            start_frame=segment[0]["frame_index"],
            end_frame=segment[-1]["frame_index"],
            best_segment_start_frame=segment[0]["frame_index"],
            best_segment_end_frame=segment[-1]["frame_index"],
            best_segment_point_count=len(segment),
            best_segment_duration_sec=_segment_duration(segment, fps),
            selected_segment_score=round(segment_score, 3),
            selected_segment_reason=segment_reason,
            observed_point_count=len(segment),
            predicted_point_count=0,
            extrapolation_used=False,
            trajectory_visualization_mode="hidden",
        ).to_dict()

    quality, reason, mode = _quality_for_segment(segment, len(fit_points))
    observed_count = len(segment)
    predicted_count = max(0, len(fit_points) - observed_count)
    return TrajectoryFitResult(
        fitted_trajectory_points=fit_points,
        observed_trajectory_points=[(int(p["x"]), int(p["y"])) for p in segment],
        trajectory_fit_quality=quality,
        trajectory_fit_reason=reason,
        start_frame=segment[0]["frame_index"],
        end_frame=segment[-1]["frame_index"],
        best_segment_start_frame=segment[0]["frame_index"],
        best_segment_end_frame=segment[-1]["frame_index"],
        best_segment_point_count=len(segment),
        best_segment_duration_sec=_segment_duration(segment, fps),
        selected_segment_score=round(segment_score, 3),
        selected_segment_reason=segment_reason,
        observed_point_count=observed_count,
        predicted_point_count=predicted_count,
        extrapolation_used=extrapolation_used,
        trajectory_visualization_mode=mode,
    ).to_dict()


def _normalize_point(point):
    if isinstance(point, dict):
        frame_index = point.get("frame_index")
        x = point.get("x")
        y = point.get("y")
        confidence = point.get("confidence")
    elif isinstance(point, (tuple, list)) and len(point) >= 3:
        frame_index, x, y = point[:3]
        confidence = point[3] if len(point) > 3 else None
    else:
        return None
    if frame_index is None or x is None or y is None:
        return None
    try:
        return {
            "frame_index": int(frame_index),
            "x": float(x),
            "y": float(y),
            "confidence": confidence,
        }
    except (TypeError, ValueError):
        return None


def _clean_points(points, *, frame_width, frame_height, terminated_frame):
    normalized = []
    for raw in points:
        point = _normalize_point(raw)
        if point is None:
            continue
        if terminated_frame is not None and point["frame_index"] > terminated_frame:
            continue
        if (
            point["x"] < 0
            or point["x"] > frame_width - 1
            or point["y"] < 0
            or point["y"] > frame_height - 1
        ):
            continue
        normalized.append(point)
    normalized.sort(key=lambda item: item["frame_index"])
    if not normalized:
        return []

    de_static = [normalized[0]]
    for point in normalized[1:]:
        prev = de_static[-1]
        if point["frame_index"] == prev["frame_index"]:
            continue
        dx = point["x"] - prev["x"]
        dy = point["y"] - prev["y"]
        if (dx * dx + dy * dy) ** 0.5 <= _STATIC_MOVE_PX:
            continue
        de_static.append(point)
    if len(de_static) < 4:
        return de_static
    return _remove_outliers(de_static)


def _remove_outliers(points):
    steps = []
    for index in range(1, len(points)):
        dx = points[index]["x"] - points[index - 1]["x"]
        dy = points[index]["y"] - points[index - 1]["y"]
        steps.append((dx * dx + dy * dy) ** 0.5)
    if not steps:
        return points
    typical_step = max(median(steps), 1.0)
    jump_limit = max(24.0, typical_step * 3.5)

    filtered = [points[0]]
    for index in range(1, len(points) - 1):
        prev = filtered[-1]
        current = points[index]
        nxt = points[index + 1]
        jump_prev = ((current["x"] - prev["x"]) ** 2 + (current["y"] - prev["y"]) ** 2) ** 0.5
        jump_next = ((nxt["x"] - current["x"]) ** 2 + (nxt["y"] - current["y"]) ** 2) ** 0.5
        bridge = ((nxt["x"] - prev["x"]) ** 2 + (nxt["y"] - prev["y"]) ** 2) ** 0.5
        if jump_prev > jump_limit and jump_next > jump_limit and bridge <= jump_limit:
            continue
        filtered.append(current)
    filtered.append(points[-1])
    return filtered


def _segments(points):
    if not points:
        return []
    segments = []
    current = [points[0]]
    for point in points[1:]:
        gap = point["frame_index"] - current[-1]["frame_index"]
        if 1 <= gap <= _MAX_MAIN_SEGMENT_GAP:
            current.append(point)
        else:
            segments.append(current)
            current = [point]
    segments.append(current)
    return segments


def _best_delivery_segment(
    points,
    *,
    frame_width,
    frame_height,
    calibration_context=None,
):
    candidates = _segments(points)
    if not candidates:
        return [], 0.0, "no_segments"

    scored = [
        (
            _segment_score(
                segment,
                frame_width=frame_width,
                frame_height=frame_height,
                calibration_context=calibration_context,
            ),
            segment,
        )
        for segment in candidates
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    score, segment = scored[0]
    return segment, score, _segment_reason(segment, score)


def _segment_score(
    segment,
    *,
    frame_width,
    frame_height,
    calibration_context=None,
) -> float:
    if not segment:
        return float("-inf")

    movement = _segment_displacement(segment)
    path_length = _segment_path_length(segment)
    span = max(1, segment[-1]["frame_index"] - segment[0]["frame_index"] + 1)
    point_count = len(segment)
    score = point_count * 12.0
    score += min(movement, 240.0) * 0.35
    score += min(path_length, 360.0) * 0.08
    score += min(span, 80) * 0.18
    score += _smoothness_score(segment) * 8.0
    if movement < _MEANINGFUL_MOVEMENT_PX:
        score -= 80.0
    score -= _edge_fraction(segment, frame_width, frame_height) * 50.0
    score += _pitch_corridor_fraction(segment, calibration_context) * 10.0
    return score


def _segment_reason(segment, score):
    if not segment:
        return "no_segment"
    movement = _segment_displacement(segment)
    if len(segment) < _PARTIAL_MIN_POINTS:
        return "short_best_segment"
    if movement < _MEANINGFUL_MOVEMENT_PX:
        return "near_static_segment_penalized"
    if len(segment) >= _GOOD_MIN_POINTS:
        return "long_smooth_moving_segment"
    return "best_smooth_moving_segment"


def _segment_displacement(segment):
    if len(segment) < 2:
        return 0.0
    return hypot(
        segment[-1]["x"] - segment[0]["x"],
        segment[-1]["y"] - segment[0]["y"],
    )


def _segment_path_length(segment):
    total = 0.0
    for index in range(1, len(segment)):
        total += hypot(
            segment[index]["x"] - segment[index - 1]["x"],
            segment[index]["y"] - segment[index - 1]["y"],
        )
    return total


def _smoothness_score(segment):
    if len(segment) < 3:
        return 0.0
    aligned = 0
    possible = 0
    for index in range(2, len(segment)):
        ax, ay = segment[index - 2]["x"], segment[index - 2]["y"]
        bx, by = segment[index - 1]["x"], segment[index - 1]["y"]
        cx, cy = segment[index]["x"], segment[index]["y"]
        v1 = (bx - ax, by - ay)
        v2 = (cx - bx, cy - by)
        if hypot(*v1) <= _STATIC_MOVE_PX or hypot(*v2) <= _STATIC_MOVE_PX:
            continue
        possible += 1
        if v1[0] * v2[0] + v1[1] * v2[1] > 0:
            aligned += 1
    return 0.0 if possible == 0 else aligned / possible


def _edge_fraction(segment, frame_width, frame_height):
    if not segment:
        return 0.0
    margin = max(
        _EDGE_MARGIN_MIN_PX,
        min(frame_width, frame_height) * _EDGE_MARGIN_RATIO,
    )
    edge_count = sum(
        1
        for point in segment
        if (
            point["x"] <= margin
            or point["y"] <= margin
            or point["x"] >= frame_width - margin
            or point["y"] >= frame_height - margin
        )
    )
    return edge_count / len(segment)


def _pitch_corridor_fraction(segment, calibration_context):
    corridor = (calibration_context or {}).get("pitch_corridor") or {}
    box = corridor.get("bbox")
    if not box or not segment:
        return 0.0
    x1, y1, x2, y2 = box
    inside = sum(
        1
        for point in segment
        if x1 <= point["x"] <= x2 and y1 <= point["y"] <= y2
    )
    return inside / len(segment)


def _segment_duration(segment, fps):
    if not segment or not fps or fps <= 0:
        return None
    span = segment[-1]["frame_index"] - segment[0]["frame_index"] + 1
    return round(span / float(fps), 3)


def _fit_segment(segment, *, frame_width, frame_height):
    frames = np.array([point["frame_index"] for point in segment], dtype=float)
    x_values = np.array([point["x"] for point in segment], dtype=float)
    y_values = np.array([point["y"] for point in segment], dtype=float)

    degree = 2 if len(segment) >= 5 else 1
    x_poly = np.poly1d(np.polyfit(frames, x_values, degree))
    y_poly = np.poly1d(np.polyfit(frames, y_values, degree))

    observed_start = int(segment[0]["frame_index"])
    observed_end = int(segment[-1]["frame_index"])
    frame_span = max(1, observed_end - observed_start)
    extra_frames = min(_MAX_EXTRAPOLATION_FRAMES, max(0, frame_span // 5))
    final_end = observed_end + extra_frames

    fit_points = []
    extrapolation_used = extra_frames > 0
    for frame_index in range(observed_start, final_end + 1):
        x_fit = int(round(float(x_poly(frame_index))))
        y_fit = int(round(float(y_poly(frame_index))))
        x_fit = min(max(0, x_fit), frame_width - 1)
        y_fit = min(max(0, y_fit), frame_height - 1)
        if fit_points and fit_points[-1] == (x_fit, y_fit):
            continue
        fit_points.append((x_fit, y_fit))
    return fit_points, extrapolation_used


def _quality_for_segment(segment, fitted_count):
    observed_count = len(segment)
    if observed_count < _PARTIAL_MIN_POINTS or fitted_count < 3:
        return "Poor", "insufficient_observed_segment", "hidden"
    if observed_count < _GOOD_MIN_POINTS:
        return "Partial", "short_segment_smoothed", "partial_fit"
    return "Good", "stable_segment_smoothed", "full_fit"
