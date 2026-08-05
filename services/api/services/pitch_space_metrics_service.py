"""Robust, explicitly estimated delivery metrics in canonical pitch space."""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable, Mapping, Sequence

from .virtual_pitch_service import build_virtual_pitch_specification


class _Record(dict):
    __getattr__ = dict.__getitem__


def _value(item: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(item, Mapping) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return default


def _result(model_names: Sequence[str], values: dict[str, Any]) -> Any:
    try:
        from ..schemas import pitch_space_analysis as schema

        for name in model_names:
            model = getattr(schema, name, None)
            if model is not None:
                return model.model_validate(values)
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    return _Record(values)


def _confidence(score: float) -> str:
    return "HIGH" if score >= 0.75 else "MEDIUM" if score >= 0.5 else "LOW"


def _usable_points(track: Iterable[Any], bounce_frame: int | None = None) -> list[Any]:
    result = []
    for point in track:
        try:
            frame = int(_value(point, "frame_index"))
            timestamp = float(_value(point, "timestamp_seconds"))
            x = float(_value(point, "pitch_x_m"))
            y = float(_value(point, "pitch_y_m"))
            confidence = float(_value(point, "combined_confidence", default=0.0))
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (timestamp, x, y, confidence)):
            continue
        if bounce_frame is not None and frame > bounce_frame:
            continue
        if confidence < 0.15 or str(_value(point, "provenance", default="")).upper() == "PROJECTED":
            continue
        result.append(point)
    return sorted(result, key=lambda item: (float(_value(item, "timestamp_seconds")), int(_value(item, "frame_index"))))


def _theil_sen(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    slopes = [
        (ys[j] - ys[i]) / (xs[j] - xs[i])
        for i in range(len(xs))
        for j in range(i + 1, len(xs))
        if abs(xs[j] - xs[i]) > 1e-9
    ]
    if not slopes:
        raise ValueError("No distinct samples")
    slope = statistics.median(slopes)
    intercept = statistics.median([y - slope * x for x, y in zip(xs, ys)])
    return slope, intercept


def _mad(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    median = statistics.median(values)
    return statistics.median([abs(value - median) for value in values])


def calculate_line_and_length(bounce: Any | None) -> Any:
    specification = build_virtual_pitch_specification()
    unavailable = {
        "status": "UNAVAILABLE", "line": "UNAVAILABLE", "length": "UNAVAILABLE",
        "lateral_offset_from_middle_m": None, "distance_from_striker_wicket_m": None,
        "distance_from_striker_popping_crease_m": None, "confidence": 0.0,
        "warnings": ["A reliable metric bounce point is unavailable."],
    }
    if bounce is None or str(_value(bounce, "status", default="UNAVAILABLE")).upper() == "UNAVAILABLE":
        return _result(("PitchSpaceLineLengthResult", "LineLengthResult"), unavailable)
    try:
        x = float(_value(bounce, "pitch_x_m", "world_x_m"))
        y = float(_value(bounce, "pitch_y_m", "world_y_m"))
        confidence = max(0.0, min(1.0, float(_value(bounce, "confidence", "confidence_score", default=0.0))))
    except (TypeError, ValueError):
        return _result(("PitchSpaceLineLengthResult", "LineLengthResult"), unavailable)
    distance = specification.dimensions.pitch_length_m - y
    crease_distance = distance - specification.dimensions.popping_crease_offset_m
    line = "MIDDLE" if abs(x) <= 0.12 else "PITCH_LEFT" if x < 0 else "PITCH_RIGHT"
    if distance < 2.0:
        length = "YORKER"
    elif distance < 5.0:
        length = "FULL"
    elif distance < 8.0:
        length = "GOOD_LENGTH"
    elif distance < 11.0:
        length = "BACK_OF_A_LENGTH"
    else:
        length = "SHORT"
    return _result(("PitchSpaceLineLengthResult", "LineLengthResult"), {
        "status": "AVAILABLE", "line": line, "length": length,
        "lateral_offset_from_middle_m": round(x, 6),
        "distance_from_striker_wicket_m": round(distance, 6),
        "distance_from_striker_popping_crease_m": round(crease_distance, 6),
        "confidence": confidence, "warnings": [],
    })


def estimate_planar_speed(pitch_space_track: Iterable[Any], *, bounce_frame: int | None = None) -> Any:
    points = _usable_points(pitch_space_track, bounce_frame)
    unavailable = {
        "status": "UNAVAILABLE", "label": "ESTIMATED_PLANAR_SPEED", "speed_mps": None,
        "speed_kmh": None, "frames_used": [], "interval_seconds": None,
        "method": "ROBUST_LONGITUDINAL_REGRESSION", "confidence": "INSUFFICIENT_EVIDENCE",
        "confidence_score": 0.0, "estimated_range_kmh": None,
        "warnings": ["At least four stable, non-projected pre-bounce points are required."],
    }
    if len(points) < 4:
        return _result(("EstimatedPlanarSpeedResult", "PitchSpaceSpeedResult"), unavailable)
    times = [float(_value(point, "timestamp_seconds")) for point in points]
    ys = [float(_value(point, "pitch_y_m")) for point in points]
    if times[-1] - times[0] < 0.08:
        unavailable["warnings"] = ["The observed interval is too short for a robust speed estimate."]
        return _result(("EstimatedPlanarSpeedResult", "PitchSpaceSpeedResult"), unavailable)
    slope, intercept = _theil_sen(times, ys)
    residuals = [y - (slope * t + intercept) for t, y in zip(times, ys)]
    median_residual = statistics.median(residuals)
    scale = max(0.02, 1.4826 * _mad(residuals))
    inliers = [index for index, value in enumerate(residuals) if abs(value - median_residual) <= 3.5 * scale]
    if len(inliers) >= 4 and len(inliers) < len(points):
        times = [times[index] for index in inliers]
        ys = [ys[index] for index in inliers]
        points = [points[index] for index in inliers]
        slope, intercept = _theil_sen(times, ys)
        residuals = [y - (slope * t + intercept) for t, y in zip(times, ys)]
    speed_mps = abs(slope)
    if not 1.0 <= speed_mps <= 60.0:
        unavailable["warnings"] = ["Estimated planar speed lies outside conservative plausible bounds."]
        return _result(("EstimatedPlanarSpeedResult", "PitchSpaceSpeedResult"), unavailable)
    pair_slopes = [abs((ys[j] - ys[i]) / (times[j] - times[i])) for i in range(len(times)) for j in range(i + 1, len(times)) if times[j] - times[i] > 1e-9]
    slope_mad = 1.4826 * _mad(pair_slopes)
    confidence_score = min(0.9, 0.3 + 0.08 * len(points) + 0.25 * min(1.0, (times[-1] - times[0]) / 0.3))
    if any(str(_value(point, "provenance", default="")).upper() != "OBSERVED" for point in points):
        confidence_score *= 0.85
    spread_kmh = max(2.0, slope_mad * 3.6)
    speed_kmh = speed_mps * 3.6
    warnings = ["Planar ground projection is not true airborne 3D ball speed."]
    return _result(("EstimatedPlanarSpeedResult", "PitchSpaceSpeedResult"), {
        "status": "AVAILABLE", "label": "ESTIMATED_PLANAR_SPEED",
        "speed_mps": round(speed_mps, 6), "speed_kmh": round(speed_kmh, 2),
        "frames_used": [int(_value(point, "frame_index")) for point in points],
        "interval_seconds": round(times[-1] - times[0], 6),
        "method": "ROBUST_LONGITUDINAL_REGRESSION", "confidence": _confidence(confidence_score),
        "confidence_score": round(confidence_score, 6),
        "estimated_range_kmh": [round(max(0.0, speed_kmh - spread_kmh), 2), round(speed_kmh + spread_kmh, 2)],
        "warnings": warnings,
    })


def estimate_lateral_movement(pitch_space_track: Iterable[Any], *, bounce_frame: int | None = None) -> Any:
    points = _usable_points(pitch_space_track, bounce_frame)
    unavailable = {
        "status": "UNAVAILABLE", "label": "ESTIMATED_LATERAL_MOVEMENT", "direction": "UNAVAILABLE",
        "movement_m": None, "maximum_movement_m": None, "frames_used": [],
        "confidence": "INSUFFICIENT_EVIDENCE", "confidence_score": 0.0,
        "warnings": ["At least six stable, non-projected pre-bounce points are required."],
    }
    if len(points) < 6:
        return _result(("EstimatedLateralMovementResult", "PitchSpaceMovementResult"), unavailable)
    ys = [float(_value(point, "pitch_y_m")) for point in points]
    xs = [float(_value(point, "pitch_x_m")) for point in points]
    if max(ys) - min(ys) < 1.0:
        unavailable["warnings"] = ["Longitudinal span is too small for lateral movement estimation."]
        return _result(("EstimatedLateralMovementResult", "PitchSpaceMovementResult"), unavailable)
    early_count = max(3, int(math.ceil(len(points) * 0.45)))
    slope, intercept = _theil_sen(ys[:early_count], xs[:early_count])
    residuals = [x - (slope * y + intercept) for x, y in zip(xs, ys)]
    scale = max(0.015, 1.4826 * _mad(residuals[:early_count]))
    # A delivery may curve progressively; keep coherent late movement while
    # rejecting only deviations too large to be credible for this planar lab.
    inliers = [index for index, value in enumerate(residuals) if abs(value - statistics.median(residuals)) <= max(0.25, 4.0 * scale)]
    later = [residuals[index] for index in inliers if index >= early_count]
    if len(later) < 2:
        unavailable["warnings"] = ["Later track points are too inconsistent for movement estimation."]
        return _result(("EstimatedLateralMovementResult", "PitchSpaceMovementResult"), unavailable)
    final = statistics.median(later[-min(3, len(later)):])
    maximum = max(later, key=abs)
    direction = "STRAIGHT" if abs(final) < 0.03 else "PITCH_RIGHT" if final > 0 else "PITCH_LEFT"
    confidence_score = min(0.85, 0.25 + 0.07 * len(inliers) + 0.2 * min(1.0, (max(ys) - min(ys)) / 8.0))
    if any(str(_value(points[index], "provenance", default="")).upper() != "OBSERVED" for index in inliers):
        confidence_score *= 0.85
    return _result(("EstimatedLateralMovementResult", "PitchSpaceMovementResult"), {
        "status": "AVAILABLE", "label": "ESTIMATED_LATERAL_MOVEMENT", "direction": direction,
        "movement_m": round(abs(final), 6), "signed_movement_m": round(final, 6),
        "maximum_movement_m": round(abs(maximum), 6),
        "frames_used": [int(_value(points[index], "frame_index")) for index in inliers],
        "confidence": _confidence(confidence_score), "confidence_score": round(confidence_score, 6),
        "warnings": ["Movement is projected lateral deviation; aerodynamic causation is not inferred."],
    })


def calculate_pitch_space_metrics(pitch_space_track: Iterable[Any], bounce: Any | None) -> dict[str, Any]:
    track = list(pitch_space_track)
    bounce_frame = _value(bounce, "bounce_frame", "frame_index") if bounce is not None else None
    return {
        "line_and_length": calculate_line_and_length(bounce),
        "estimated_planar_speed": estimate_planar_speed(track, bounce_frame=bounce_frame),
        "estimated_lateral_movement": estimate_lateral_movement(track, bounce_frame=bounce_frame),
    }
