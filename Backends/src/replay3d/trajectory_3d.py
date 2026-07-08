"""Estimate a coarse 3D trajectory from image-space tracking and calibration."""

from __future__ import annotations

from typing import Any

MIN_3D_POINTS = 5

UNKNOWN_METRICS = {
    "speed_kmh": "Not calibrated",
    "swing": "Unknown",
    "spin": "Unknown",
    "lbw": "Not available",
}


def build_estimated_3d_trajectory(
    trajectory_points: Any,
    calibration_context: dict[str, Any] | None,
    bounce_point: Any = None,
    impact_point: Any = None,
) -> dict[str, Any]:
    """Map tracked image points into coarse pitch-foot coordinates."""
    calibration_context = calibration_context or {}
    points_2d = _valid_points(trajectory_points)
    notes = list(calibration_context.get("notes") or [])
    notes.extend(
        [
            "3D positions are illustrative estimates, not measured ball physics.",
            f"Speed: {UNKNOWN_METRICS['speed_kmh']}.",
            f"Swing: {UNKNOWN_METRICS['swing']}.",
            f"Spin: {UNKNOWN_METRICS['spin']}.",
            f"LBW: {UNKNOWN_METRICS['lbw']}.",
        ]
    )

    if len(points_2d) < MIN_3D_POINTS:
        return {
            "available": False,
            "trajectory_quality": "Unavailable",
            "points_3d": [],
            "release_3d": None,
            "bounce_3d": None,
            "impact_3d": None,
            "metrics": dict(UNKNOWN_METRICS),
            "notes": notes
            + [f"Need at least {MIN_3D_POINTS} tracked points; got {len(points_2d)}."],
        }

    if calibration_context.get("calibration_quality") == "Disabled":
        return {
            "available": False,
            "trajectory_quality": "Unavailable",
            "points_3d": [],
            "release_3d": None,
            "bounce_3d": None,
            "impact_3d": None,
            "metrics": dict(UNKNOWN_METRICS),
            "notes": notes + ["Calibration disabled; cannot estimate 3D replay."],
        }

    mapper = _PitchMapper(calibration_context)
    points_3d = [mapper.image_to_pitch(point) for point in points_2d]
    release_3d = points_3d[0] if points_3d else None

    bounce_2d = _coerce_point(bounce_point)
    bounce_3d = None
    if bounce_2d is not None:
        bounce_3d = mapper.image_to_pitch(bounce_2d)
        bounce_3d["z_ft"] = 0.0
        bounce_3d["z_source"] = "bounce_plane"
    else:
        notes.append("Bounce point unknown; bounce marker omitted.")

    impact_2d = _coerce_point(impact_point)
    impact_3d = mapper.image_to_pitch(impact_2d) if impact_2d else None
    if impact_2d is None:
        notes.append("Impact point unknown; impact marker omitted.")

    quality = _trajectory_quality(
        len(points_3d),
        calibration_context.get("calibration_quality"),
        bounce_3d is not None,
        impact_3d is not None,
    )

    return {
        "available": True,
        "trajectory_quality": quality,
        "points_3d": points_3d,
        "release_3d": release_3d,
        "bounce_3d": bounce_3d,
        "impact_3d": impact_3d,
        "metrics": dict(UNKNOWN_METRICS),
        "notes": list(dict.fromkeys(notes)),
    }


class _PitchMapper:
    """Map image pixels to coarse pitch X/Y/Z using stump-calibrated ROI."""

    def __init__(self, calibration_context: dict[str, Any]) -> None:
        self.context = calibration_context
        centerline = calibration_context.get("pitch_centerline") or {}
        bowler = centerline.get("bowler_end_image") or [640.0, 200.0]
        batter = centerline.get("batter_end_image") or [640.0, 520.0]
        self.bowler_y = float(bowler[1])
        self.batter_y = float(batter[1])
        self.center_x = float((bowler[0] + batter[0]) / 2.0)
        pitch_roi = calibration_context.get("pitch_roi") or {}
        bbox = pitch_roi.get("bbox") or [0.0, self.bowler_y, 1280.0, self.batter_y]
        self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2 = [float(v) for v in bbox]
        self.pitch_length_ft = float(calibration_context.get("pitch_length_ft") or 66.0)
        self.pitch_width_ft = float(calibration_context.get("pitch_width_ft") or 10.0)
        self.camera_height_ft = float(calibration_context.get("camera_height_ft") or 8.0)
        self.length_span = max(abs(self.batter_y - self.bowler_y), 1.0)
        self.width_span = max(self.roi_x2 - self.roi_x1, 1.0)

    def image_to_pitch(self, point: tuple[int, int]) -> dict[str, float]:
        x_px, y_px = float(point[0]), float(point[1])
        length_t = (y_px - self.bowler_y) / self.length_span
        length_t = min(max(length_t, 0.0), 1.0)
        lateral_t = (x_px - self.center_x) / (self.width_span / 2.0)
        lateral_t = min(max(lateral_t, -1.0), 1.0)

        y_ft = length_t * self.pitch_length_ft
        x_ft = lateral_t * (self.pitch_width_ft / 2.0)
        # ponytail: height from image offset above local pitch line; not true ball tracking height.
        line_y = self.bowler_y + length_t * (self.batter_y - self.bowler_y)
        vertical_offset = max(0.0, line_y - y_px)
        z_ft = min(
            self.camera_height_ft * 0.35,
            (vertical_offset / max(self.length_span, 1.0)) * self.camera_height_ft,
        )
        return {
            "x_ft": round(x_ft, 3),
            "y_ft": round(y_ft, 3),
            "z_ft": round(z_ft, 3),
            "image_x": round(x_px, 2),
            "image_y": round(y_px, 2),
        }


def _trajectory_quality(
    point_count: int,
    calibration_quality: Any,
    has_bounce: bool,
    has_impact: bool,
) -> str:
    if point_count < MIN_3D_POINTS:
        return "Unavailable"
    if calibration_quality == "Good" and point_count >= 8:
        return "Good"
    if calibration_quality in {"Good", "Partial"} and point_count >= 5:
        if has_bounce or has_impact:
            return "Partial"
        return "Low"
    if calibration_quality == "Low":
        return "Low"
    return "Partial" if point_count >= 6 else "Low"


def _valid_points(trajectory_points: Any) -> list[tuple[int, int]]:
    if not isinstance(trajectory_points, (list, tuple)):
        return []
    points: list[tuple[int, int]] = []
    for item in trajectory_points:
        point = _coerce_point(item)
        if point is not None:
            points.append(point)
    return points


def _coerce_point(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            try:
                return int(value["x"]), int(value["y"])
            except (TypeError, ValueError):
                return None
        if "x_ft" in value and "y_ft" in value:
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
