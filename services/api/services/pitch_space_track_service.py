"""Pure image-track to canonical pitch-space conversion."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from .virtual_pitch_service import build_virtual_pitch_specification


AIRBORNE_WARNING = (
    "Ground-plane projection does not recover the airborne ball's true 3D position."
)


class _Record(dict):
    """Mapping fallback used while the shared schema is integrated."""

    __getattr__ = dict.__getitem__


def _value(item: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(item, Mapping) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _pitch_space_point(values: dict[str, Any]) -> Any:
    try:
        from ..schemas.pitch_space_analysis import PitchSpaceTrackPoint

        return PitchSpaceTrackPoint.model_validate(values)
    except (ImportError, AttributeError, TypeError, ValueError):
        return _Record(values)


def project_image_point(
    image_x_px: float,
    image_y_px: float,
    image_to_pitch_homography: Sequence[Sequence[float]],
) -> tuple[float, float] | None:
    """Apply a supplied 3x3 homography without mutating either input."""
    try:
        matrix = [[float(value) for value in row] for row in image_to_pitch_homography]
    except (TypeError, ValueError):
        return None
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        return None
    if not all(math.isfinite(value) for row in matrix for value in row):
        return None
    denominator = matrix[2][0] * image_x_px + matrix[2][1] * image_y_px + matrix[2][2]
    if not math.isfinite(denominator) or abs(denominator) < 1e-10:
        return None
    pitch_x = (
        matrix[0][0] * image_x_px + matrix[0][1] * image_y_px + matrix[0][2]
    ) / denominator
    pitch_y = (
        matrix[1][0] * image_x_px + matrix[1][1] * image_y_px + matrix[1][2]
    ) / denominator
    if not math.isfinite(pitch_x) or not math.isfinite(pitch_y):
        return None
    return float(pitch_x), float(pitch_y)


def convert_track_to_pitch_space(
    tracked_points: Iterable[Any],
    image_to_pitch_homography: Sequence[Sequence[float]],
    *,
    pitch_fit_confidence: float,
    bounce_frame: int | None = None,
    camera_stability: str = "FIXED_CAMERA",
    unstable_after_frame: int | None = None,
) -> list[Any]:
    """Map valid existing track points while preserving image-space evidence."""
    dimensions = build_virtual_pitch_specification().dimensions
    fit_confidence = _clamp(float(pitch_fit_confidence))
    results: list[Any] = []
    for point in tracked_points:
        if _value(point, "track_valid", "valid", default=True) is False:
            continue
        try:
            frame_index = int(_value(point, "frame_index"))
            timestamp = float(_value(point, "timestamp_seconds"))
            image_x = float(_value(point, "image_x_px", "x", "pixel_x"))
            image_y = float(_value(point, "image_y_px", "y", "pixel_y"))
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (timestamp, image_x, image_y)):
            continue
        projected = project_image_point(image_x, image_y, image_to_pitch_homography)
        if projected is None:
            continue

        detector_confidence = _clamp(
            float(_value(point, "detection_confidence", "detector_confidence", "confidence", default=0.0))
        )
        point_fit_confidence = fit_confidence
        warnings = [AIRBORNE_WARNING]
        stability = camera_stability.upper()
        affected_by_instability = (
            stability == "UNSTABLE_CAMERA"
            and (unstable_after_frame is None or frame_index >= unstable_after_frame)
        )
        if stability == "MINOR_DRIFT":
            point_fit_confidence *= 0.8
            warnings.append("Minor camera drift reduces pitch-space confidence.")
        elif affected_by_instability:
            point_fit_confidence = 0.0
            warnings.append("Camera instability makes this metric projection unreliable.")

        pitch_x, pitch_y = projected
        in_pitch = (
            -dimensions.pitch_width_m / 2 <= pitch_x <= dimensions.pitch_width_m / 2
            and 0.0 <= pitch_y <= dimensions.pitch_length_m
        )
        if not in_pitch:
            warnings.append("Projected point is outside canonical pitch bounds.")
        provenance = str(_value(point, "provenance", "source", default="UNKNOWN")).upper()
        if provenance != "OBSERVED":
            warnings.append(f"Point provenance is {provenance}; it was not directly detected.")
        phase = "UNKNOWN"
        if bounce_frame is not None:
            phase = "PRE_BOUNCE" if frame_index < bounce_frame else "POST_BOUNCE"
            if frame_index == bounce_frame:
                phase = "BOUNCE"
        combined = math.sqrt(detector_confidence * point_fit_confidence)
        if not in_pitch:
            combined *= 0.75
        values = {
            "frame_index": frame_index,
            "timestamp_seconds": timestamp,
            "image_x_px": image_x,
            "image_y_px": image_y,
            "pitch_x_m": pitch_x,
            "pitch_y_m": pitch_y,
            "detection_confidence": detector_confidence,
            "pitch_fit_confidence": _clamp(point_fit_confidence),
            "combined_confidence": round(_clamp(combined), 6),
            "provenance": provenance,
            "in_pitch_bounds": in_pitch,
            "bounce_phase": phase,
            "warnings": warnings,
        }
        results.append(_pitch_space_point(values))
    return sorted(results, key=lambda item: (_value(item, "frame_index"), _value(item, "timestamp_seconds")))


# Integration-friendly spelling used by some callers.
transform_track_to_pitch_space = convert_track_to_pitch_space

