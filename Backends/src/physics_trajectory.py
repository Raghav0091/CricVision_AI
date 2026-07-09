"""Physics-assisted trajectory fitter v1 for cricket delivery paths.

Pure Python/NumPy helpers. No Streamlit, YOLO, OpenCV, or model loading.
Overlay/3D/UI only — fits a conservative delivery path from trusted
pre-impact ball points and never fakes official review-system verdicts.
"""

from __future__ import annotations

from math import hypot, isfinite
from typing import Any

import numpy as np

from Backends.src.cricket_path_validity import (
    _coerce_pitch_roi_bbox as coerce_pitch_roi_bbox,
    extract_impact_frame,
    parse_frame_size,
)

QUALITY_UNAVAILABLE = "Unavailable"
QUALITY_POOR = "Poor"
QUALITY_PARTIAL = "Partial"
QUALITY_GOOD = "Good"

CONFIDENCE_UNKNOWN = "Unknown"
CONFIDENCE_LOW = "Low"
CONFIDENCE_MEDIUM = "Medium"

PROJECTED_PATH_NOTE = "Projected path — estimated, not observed after bat contact."
SAFETY_NOTE = "Physics layer is an estimated visual aid, not an official review system."

# ponytail: heuristic thresholds tuned for smartphone cricket clips; mirrors cricket_path_validity.
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720
MIN_FIT_POINTS = 5
MAX_JUMP_DIAG_RATIO = 0.22
MIN_MAX_JUMP_PX = 25.0
SIDEWAYS_RATIO = 1.7
SIDEWAYS_MIN_DX = 6.0
REVERSAL_MIN_DY = 12.0
ROI_MARGIN_RATIO = 0.15
FIT_RESIDUAL_DIAG_RATIO = 0.03
MIN_FIT_RESIDUAL_PX = 20.0
GOOD_MIN_POINTS = 10
GOOD_MAX_REJECT_RATIO = 0.2
POOR_REJECT_RATIO = 0.6
MIN_PROJECTION_INPUT_POINTS = 6
MAX_PROJECTION_STEPS = 20
MAX_PROJECTION_TRAVEL_DIAG_RATIO = 0.3
MIN_BOUNCE_POINTS = 5
MIN_BOUNCE_TURN_PX = 2.0
CLEAR_BOUNCE_TURN_PX = 4.0


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if isfinite(result) else default


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_point(item: Any) -> dict[str, Any] | None:
    if isinstance(item, dict):
        x = _safe_float(item.get("x"))
        y = _safe_float(item.get("y"))
        if x is None or y is None:
            for key in ("center", "centroid"):
                pair = item.get(key)
                if isinstance(pair, dict):
                    x = _safe_float(pair.get("x"))
                    y = _safe_float(pair.get("y"))
                elif isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    x = _safe_float(pair[0])
                    y = _safe_float(pair[1])
                if x is not None and y is not None:
                    break
        if x is None or y is None:
            bbox = item.get("bbox") or item.get("box")
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                corners = [_safe_float(value) for value in bbox[:4]]
                if None not in corners:
                    x = (corners[0] + corners[2]) / 2.0
                    y = (corners[1] + corners[3]) / 2.0
        if x is None or y is None:
            return None
        return {
            "frame_index": _safe_int(item.get("frame_index")),
            "x": float(x),
            "y": float(y),
            "confidence": _safe_float(item.get("confidence")),
            "source": str(item.get("source")) if item.get("source") else "observed",
        }

    if isinstance(item, (list, tuple)):
        if len(item) >= 3:
            frame_index = _safe_int(item[0])
            x = _safe_float(item[1])
            y = _safe_float(item[2])
            confidence = _safe_float(item[3]) if len(item) > 3 else None
        elif len(item) == 2:
            frame_index = None
            x = _safe_float(item[0])
            y = _safe_float(item[1])
            confidence = None
        else:
            return None
        if x is None or y is None:
            return None
        return {
            "frame_index": frame_index,
            "x": float(x),
            "y": float(y),
            "confidence": confidence,
            "source": "observed",
        }

    return None


def normalize_trajectory_points(points: Any) -> list[dict[str, Any]]:
    """Normalize mixed point formats into {frame_index, x, y, confidence, source} dicts."""
    if points is None or not isinstance(points, (list, tuple)):
        return []

    parsed: list[dict[str, Any]] = []
    for order, item in enumerate(points):
        point = _parse_point(item)
        if point is not None:
            point["_order"] = order
            parsed.append(point)

    if parsed and all(point["frame_index"] is None for point in parsed):
        # ponytail: no frame info anywhere; list order is the best frame proxy we have.
        for order_index, point in enumerate(parsed):
            point["frame_index"] = order_index
    if parsed and all(point["frame_index"] is not None for point in parsed):
        parsed.sort(key=lambda point: (point["frame_index"], point["_order"]))

    for point in parsed:
        point.pop("_order", None)
    return parsed


def split_pre_impact_path(
    points: Any,
    impact_frame: Any = None,
    impact_point: Any = None,
) -> dict[str, Any]:
    """Split normalized points into pre-impact (trusted) and post-impact observed shot path."""
    parsed = normalize_trajectory_points(points)
    frame = extract_impact_frame(impact_frame)
    point = _parse_point(impact_point) if impact_point is not None else None

    if frame is None:
        return {
            "pre_impact_points": parsed,
            "post_impact_observed_points": [],
            "impact_detected": point is not None,
            "impact_frame": None,
            "impact_point": point,
        }

    pre: list[dict[str, Any]] = []
    post: list[dict[str, Any]] = []
    for item in parsed:
        item_frame = item.get("frame_index")
        if item_frame is None or item_frame <= frame:
            pre.append(item)
        else:
            post.append(item)
    return {
        "pre_impact_points": pre,
        "post_impact_observed_points": post,
        "impact_detected": True,
        "impact_frame": frame,
        "impact_point": point,
    }


def _frame_bounds(frame_size: Any) -> tuple[int, int, float]:
    size = parse_frame_size(frame_size)
    width, height = size if size is not None else (DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT)
    return width, height, hypot(width, height)


def estimate_bounce_point(
    pre_impact_points: Any,
    frame_size: Any = None,
    pitch_roi: Any = None,
) -> dict[str, Any]:
    """Estimate a likely bounce point from the pre-impact path without overclaiming."""
    notes: list[str] = []
    result: dict[str, Any] = {
        "bounce_detected": False,
        "bounce_point": None,
        "bounce_frame": None,
        "confidence": CONFIDENCE_UNKNOWN,
        "notes": notes,
    }
    try:
        points = normalize_trajectory_points(pre_impact_points)
    except Exception as exc:
        notes.append(f"Bounce estimate failed safely: {exc}")
        return result

    if len(points) < MIN_BOUNCE_POINTS:
        notes.append(
            f"Need at least {MIN_BOUNCE_POINTS} pre-impact points for a bounce estimate; got {len(points)}."
        )
        result["confidence"] = CONFIDENCE_LOW if len(points) >= 3 else CONFIDENCE_UNKNOWN
        return result

    ys = [point["y"] for point in points]
    overall_dy = ys[-1] - ys[0]
    if abs(overall_dy) < 8.0:
        notes.append("No clear vertical ball motion; bounce not estimated.")
        return result

    sign = 1.0 if overall_dy > 0 else -1.0
    directional = [sign * value for value in ys]
    turn_index = int(np.argmax(directional))
    if turn_index in (0, len(points) - 1):
        notes.append("No local vertical turning point inside the path; bounce not estimated.")
        return result

    turn_before = directional[turn_index] - directional[max(0, turn_index - 2)]
    turn_after = directional[turn_index] - directional[min(len(points) - 1, turn_index + 2)]
    if turn_before < MIN_BOUNCE_TURN_PX or turn_after < MIN_BOUNCE_TURN_PX:
        notes.append("Vertical turning is too weak to call a bounce.")
        result["confidence"] = CONFIDENCE_LOW
        return result

    width, height, _diag = _frame_bounds(frame_size)
    candidate = points[turn_index]
    bounce_x = min(max(candidate["x"], 0.0), float(width - 1))
    bounce_y = min(max(candidate["y"], 0.0), float(height - 1))

    confidence = (
        CONFIDENCE_MEDIUM
        if len(points) >= 8 and turn_before >= CLEAR_BOUNCE_TURN_PX and turn_after >= CLEAR_BOUNCE_TURN_PX
        else CONFIDENCE_LOW
    )
    roi_bbox = coerce_pitch_roi_bbox(pitch_roi)
    if roi_bbox is not None:
        x1, y1, x2, y2 = roi_bbox
        margin_x = max(10.0, (x2 - x1) * ROI_MARGIN_RATIO)
        margin_y = max(10.0, (y2 - y1) * ROI_MARGIN_RATIO)
        if not (x1 - margin_x <= bounce_x <= x2 + margin_x and y1 - margin_y <= bounce_y <= y2 + margin_y):
            confidence = CONFIDENCE_LOW
            notes.append("Bounce candidate sits outside the pitch corridor; confidence lowered.")

    result.update(
        {
            "bounce_detected": True,
            "bounce_point": {"x": round(bounce_x, 3), "y": round(bounce_y, 3)},
            "bounce_frame": candidate.get("frame_index"),
            "confidence": confidence,
        }
    )
    notes.append("Bounce estimated from a local vertical turning point; image-space heuristic only.")
    return result


def _reject_payload(point: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "frame_index": point.get("frame_index"),
        "x": round(float(point["x"]), 3),
        "y": round(float(point["y"]), 3),
        "reason": reason,
    }


def _filter_pre_impact_points(
    points: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    diag: float,
    roi_bbox: tuple[float, float, float, float] | None,
    bounce_frame: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jump_limit = max(MIN_MAX_JUMP_PX, diag * MAX_JUMP_DIAG_RATIO)
    ys = [point["y"] for point in points]
    sign = 1.0 if (ys[-1] - ys[0]) >= 0 else -1.0

    used: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for point in points:
        x, y = point["x"], point["y"]
        reason = None
        if x < 0 or y < 0 or x > width or y > height:
            reason = "outside_frame"
        elif roi_bbox is not None:
            x1, y1, x2, y2 = roi_bbox
            margin_x = max(10.0, (x2 - x1) * ROI_MARGIN_RATIO)
            margin_y = max(10.0, (y2 - y1) * ROI_MARGIN_RATIO)
            if not (x1 - margin_x <= x <= x2 + margin_x and y1 - margin_y <= y <= y2 + margin_y):
                reason = "outside_pitch_corridor"

        if reason is None and used:
            prev = used[-1]
            dx = x - prev["x"]
            dy = y - prev["y"]
            near_bounce = (
                bounce_frame is not None
                and point.get("frame_index") is not None
                and abs(point["frame_index"] - bounce_frame) <= 2
            )
            if hypot(dx, dy) > jump_limit:
                reason = "huge_jump"
            elif abs(dx) > abs(dy) * SIDEWAYS_RATIO and abs(dx) > SIDEWAYS_MIN_DX:
                reason = "impossible_sideways"
            elif (
                not near_bounce
                and dy * sign < 0
                and abs(dy) > max(REVERSAL_MIN_DY, abs(dx))
            ):
                reason = "direction_reversal"

        if reason is not None:
            rejected.append(_reject_payload(point, reason))
        else:
            used.append(point)
    return used, rejected


def _poly_fit_path(
    sub_points: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    diag: float,
) -> list[dict[str, Any]] | None:
    """Fit low-degree x(t)/y(t) polynomials; return None when the fit is unstable."""
    frames = [point["frame_index"] for point in sub_points]
    if any(value is None for value in frames):
        t = np.arange(len(sub_points), dtype=float)
    else:
        t = np.array(frames, dtype=float)
        if float(np.ptp(t)) < 1e-6:
            t = np.arange(len(sub_points), dtype=float)
    xs = np.array([point["x"] for point in sub_points], dtype=float)
    ys = np.array([point["y"] for point in sub_points], dtype=float)

    degree = 2 if len(sub_points) >= 6 else 1
    try:
        coef_x = np.polyfit(t, xs, degree)
        coef_y = np.polyfit(t, ys, degree)
    except Exception:
        return None

    residual = max(
        float(np.max(np.abs(np.polyval(coef_x, t) - xs))),
        float(np.max(np.abs(np.polyval(coef_y, t) - ys))),
    )
    if residual > max(MIN_FIT_RESIDUAL_PX, diag * FIT_RESIDUAL_DIAG_RATIO):
        return None

    # No extrapolation: sample only over the observed frame span.
    t0, t1 = float(t[0]), float(t[-1])
    sample_count = int(min(max(t1 - t0 + 1, 2), max(2 * len(sub_points), 24)))
    fitted: list[dict[str, Any]] = []
    clamped = 0
    for tv in np.linspace(t0, t1, num=sample_count):
        x = float(np.polyval(coef_x, tv))
        y = float(np.polyval(coef_y, tv))
        cx = min(max(x, 0.0), float(width - 1))
        cy = min(max(y, 0.0), float(height - 1))
        if cx != x or cy != y:
            clamped += 1
        fitted.append(
            {
                "frame_index": int(round(tv)),
                "x": round(cx, 3),
                "y": round(cy, 3),
                "source": "physics_fit",
            }
        )
    if clamped > len(fitted) * 0.3:
        return None
    return fitted


def _raw_fallback_path(used: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "frame_index": point.get("frame_index"),
            "x": round(float(point["x"]), 3),
            "y": round(float(point["y"]), 3),
            "source": "physics_raw",
        }
        for point in used
    ]


def fit_physics_trajectory(
    pre_impact_points: Any,
    frame_size: Any = None,
    pitch_roi: Any = None,
    bounce_context: Any = None,
) -> dict[str, Any]:
    """Fit a conservative cricket delivery path from trusted pre-impact points."""
    try:
        return _fit_physics_trajectory_impl(
            pre_impact_points,
            frame_size=frame_size,
            pitch_roi=pitch_roi,
            bounce_context=bounce_context,
        )
    except Exception as exc:
        return {
            "fitted_path": [],
            "raw_points_used": [],
            "rejected_points": [],
            "bounce": {
                "bounce_detected": False,
                "bounce_point": None,
                "bounce_frame": None,
                "confidence": CONFIDENCE_UNKNOWN,
                "notes": [],
            },
            "physics_quality": QUALITY_UNAVAILABLE,
            "path_score": 0.0,
            "notes": [f"Physics fit failed safely: {exc}"],
        }


def _fit_physics_trajectory_impl(
    pre_impact_points: Any,
    *,
    frame_size: Any = None,
    pitch_roi: Any = None,
    bounce_context: Any = None,
) -> dict[str, Any]:
    points = normalize_trajectory_points(pre_impact_points)
    width, height, diag = _frame_bounds(frame_size)
    roi_bbox = coerce_pitch_roi_bbox(pitch_roi)
    notes: list[str] = []

    if len(points) < MIN_FIT_POINTS:
        quality = QUALITY_UNAVAILABLE if not points else QUALITY_POOR
        notes.append(f"Need at least {MIN_FIT_POINTS} pre-impact points; got {len(points)}.")
        bounce = bounce_context if isinstance(bounce_context, dict) else estimate_bounce_point(
            points, frame_size=frame_size, pitch_roi=pitch_roi
        )
        return {
            "fitted_path": [],
            "raw_points_used": [],
            "rejected_points": [_reject_payload(point, "too_few_points") for point in points],
            "bounce": bounce,
            "physics_quality": quality,
            "path_score": 0.0,
            "notes": notes,
        }

    bounce = bounce_context if isinstance(bounce_context, dict) else estimate_bounce_point(
        points, frame_size=frame_size, pitch_roi=pitch_roi
    )
    bounce_frame = _safe_int(bounce.get("bounce_frame")) if bounce.get("bounce_detected") else None

    used, rejected = _filter_pre_impact_points(
        points,
        width=width,
        height=height,
        diag=diag,
        roi_bbox=roi_bbox,
        bounce_frame=bounce_frame,
    )
    if rejected:
        notes.append(f"Rejected {len(rejected)} implausible points before fitting.")

    if len(used) < MIN_FIT_POINTS:
        notes.append(f"Only {len(used)} points survived filtering; not enough to fit.")
        return {
            "fitted_path": [],
            "raw_points_used": used,
            "rejected_points": rejected,
            "bounce": bounce,
            "physics_quality": QUALITY_POOR,
            "path_score": 0.0,
            "notes": notes,
        }

    fitted: list[dict[str, Any]] | None = None
    if bounce_frame is not None:
        # ponytail: one quadratic cannot follow a V-shaped bounce; fit each arc separately.
        pre_bounce = [p for p in used if p.get("frame_index") is not None and p["frame_index"] <= bounce_frame]
        post_bounce = [p for p in used if p.get("frame_index") is not None and p["frame_index"] > bounce_frame]
        if len(pre_bounce) >= 4 and len(post_bounce) >= 4:
            first = _poly_fit_path(pre_bounce, width=width, height=height, diag=diag)
            second = _poly_fit_path(post_bounce, width=width, height=height, diag=diag)
            if first and second:
                fitted = first + second
                notes.append("Fitted delivery in two arcs around the estimated bounce.")
    if fitted is None:
        fitted = _poly_fit_path(used, width=width, height=height, diag=diag)

    fallback = False
    if fitted is None:
        fallback = True
        fitted = _raw_fallback_path(used)
        notes.append("Curve fit unstable; using raw filtered pre-impact points.")
    elif roi_bbox is not None:
        x1, y1, x2, y2 = roi_bbox
        margin_x = max(10.0, (x2 - x1) * ROI_MARGIN_RATIO)
        margin_y = max(10.0, (y2 - y1) * ROI_MARGIN_RATIO)
        inside = sum(
            1
            for point in fitted
            if x1 - margin_x <= point["x"] <= x2 + margin_x and y1 - margin_y <= point["y"] <= y2 + margin_y
        )
        if inside < len(fitted) * 0.7:
            fallback = True
            fitted = _raw_fallback_path(used)
            notes.append("Fitted curve strayed from the pitch corridor; using raw filtered points.")

    reject_ratio = len(rejected) / max(1, len(used) + len(rejected))
    if reject_ratio > POOR_REJECT_RATIO:
        quality = QUALITY_POOR
        notes.append("Most points were rejected; path is not trustworthy.")
    elif fallback:
        quality = QUALITY_PARTIAL
    elif len(used) >= GOOD_MIN_POINTS and reject_ratio <= GOOD_MAX_REJECT_RATIO:
        quality = QUALITY_GOOD
        notes.append("Smooth forward pre-impact path fitted successfully.")
    else:
        quality = QUALITY_PARTIAL

    path_score = round(max(0.0, min(1.0, (len(used) / 12.0) * (1.0 - reject_ratio))), 3)
    return {
        "fitted_path": fitted if quality != QUALITY_POOR else [],
        "raw_points_used": used,
        "rejected_points": rejected,
        "bounce": bounce,
        "physics_quality": quality,
        "path_score": path_score,
        "notes": notes,
    }


def project_path_after_impact(
    fitted_path: Any,
    impact_frame: Any = None,
    frame_size: Any = None,
    pitch_roi: Any = None,
    max_future_points: int = MAX_PROJECTION_STEPS,
) -> dict[str, Any]:
    """Project a short, conservative estimated continuation after impact."""
    try:
        return _project_path_after_impact_impl(
            fitted_path,
            impact_frame=impact_frame,
            frame_size=frame_size,
            pitch_roi=pitch_roi,
            max_future_points=max_future_points,
        )
    except Exception as exc:
        return {
            "projected_path": [],
            "projection_quality": QUALITY_UNAVAILABLE,
            "notes": [f"Projection failed safely: {exc}"],
        }


def _project_path_after_impact_impl(
    fitted_path: Any,
    *,
    impact_frame: Any = None,
    frame_size: Any = None,
    pitch_roi: Any = None,
    max_future_points: int = MAX_PROJECTION_STEPS,
) -> dict[str, Any]:
    points = normalize_trajectory_points(fitted_path)
    notes: list[str] = []
    if len(points) < MIN_PROJECTION_INPUT_POINTS:
        return {
            "projected_path": [],
            "projection_quality": QUALITY_UNAVAILABLE,
            "notes": ["Not enough fitted points to project a continuation."],
        }

    steps = max(0, min(_safe_int(max_future_points, MAX_PROJECTION_STEPS) or 0, MAX_PROJECTION_STEPS))
    if steps == 0:
        return {
            "projected_path": [],
            "projection_quality": QUALITY_UNAVAILABLE,
            "notes": ["Projection disabled."],
        }

    width, height, diag = _frame_bounds(frame_size)
    roi_bbox = coerce_pitch_roi_bbox(pitch_roi)

    tail = points[-min(5, len(points)) :]
    frames = [point["frame_index"] for point in tail]
    if any(value is None for value in frames):
        t = np.arange(len(tail), dtype=float)
    else:
        t = np.array(frames, dtype=float)
        if float(np.ptp(t)) < 1e-6:
            t = np.arange(len(tail), dtype=float)
    xs = np.array([point["x"] for point in tail], dtype=float)
    ys = np.array([point["y"] for point in tail], dtype=float)
    vx = float(np.polyfit(t, xs, 1)[0])
    vy = float(np.polyfit(t, ys, 1)[0])
    step_px = hypot(vx, vy)
    if step_px < 0.5:
        return {
            "projected_path": [],
            "projection_quality": QUALITY_UNAVAILABLE,
            "notes": ["Near-zero motion at the end of the fitted path; projection skipped."],
        }

    last = points[-1]
    last_frame = _safe_int(last.get("frame_index"))
    impact = extract_impact_frame(impact_frame)
    start_frame = max(value for value in (last_frame, impact, 0) if value is not None)

    max_travel = diag * MAX_PROJECTION_TRAVEL_DIAG_RATIO
    projected: list[dict[str, Any]] = []
    traveled = 0.0
    for step in range(1, steps + 1):
        x_next = float(last["x"]) + vx * step
        y_next = float(last["y"]) + vy * step
        if not (0 <= x_next <= width - 1 and 0 <= y_next <= height - 1):
            notes.append("Projection stopped at the frame boundary.")
            break
        if roi_bbox is not None:
            x1, y1, x2, y2 = roi_bbox
            margin_x = max(10.0, (x2 - x1) * ROI_MARGIN_RATIO)
            margin_y = max(10.0, (y2 - y1) * ROI_MARGIN_RATIO)
            if not (x1 - margin_x <= x_next <= x2 + margin_x and y1 - margin_y <= y_next <= y2 + margin_y):
                notes.append("Projection stopped at the pitch corridor edge.")
                break
        traveled += step_px
        if traveled > max_travel:
            notes.append("Projection stopped at the conservative travel limit.")
            break
        projected.append(
            {
                "frame_index": start_frame + step,
                "x": round(x_next, 3),
                "y": round(y_next, 3),
                "source": "projected",
            }
        )

    if not projected:
        quality = QUALITY_UNAVAILABLE
        notes.append("No safe projected points were produced.")
    else:
        notes.append(PROJECTED_PATH_NOTE)
        if len(projected) < 3:
            quality = QUALITY_POOR
        elif len(projected) >= 5 and len(points) >= GOOD_MIN_POINTS:
            quality = QUALITY_GOOD
        else:
            quality = QUALITY_PARTIAL
    return {
        "projected_path": projected,
        "projection_quality": quality,
        "notes": notes,
    }


def build_physics_trajectory_report(
    points: Any,
    impact_frame: Any = None,
    impact_point: Any = None,
    frame_size: Any = None,
    pitch_roi: Any = None,
) -> dict[str, Any]:
    """Normalize, split at impact, estimate bounce, fit, and conservatively project."""
    try:
        return _build_physics_trajectory_report_impl(
            points,
            impact_frame=impact_frame,
            impact_point=impact_point,
            frame_size=frame_size,
            pitch_roi=pitch_roi,
        )
    except Exception as exc:
        return {
            "pre_impact_path": [],
            "post_impact_observed_points": [],
            "fitted_delivery_path": [],
            "projected_path": [],
            "bounce": {
                "bounce_detected": False,
                "bounce_point": None,
                "bounce_frame": None,
                "confidence": CONFIDENCE_UNKNOWN,
                "notes": [],
            },
            "impact": {"impact_detected": False, "impact_frame": None, "impact_point": None},
            "physics_quality": QUALITY_UNAVAILABLE,
            "projection_quality": QUALITY_UNAVAILABLE,
            "path_score": 0.0,
            "notes": [f"Physics trajectory report failed safely: {exc}"],
        }


def _build_physics_trajectory_report_impl(
    points: Any,
    *,
    impact_frame: Any = None,
    impact_point: Any = None,
    frame_size: Any = None,
    pitch_roi: Any = None,
) -> dict[str, Any]:
    split = split_pre_impact_path(points, impact_frame=impact_frame, impact_point=impact_point)
    pre = split["pre_impact_points"]
    post = split["post_impact_observed_points"]

    bounce = estimate_bounce_point(pre, frame_size=frame_size, pitch_roi=pitch_roi)
    fit = fit_physics_trajectory(
        pre,
        frame_size=frame_size,
        pitch_roi=pitch_roi,
        bounce_context=bounce,
    )

    if split["impact_detected"] and fit["physics_quality"] in {QUALITY_GOOD, QUALITY_PARTIAL}:
        projection = project_path_after_impact(
            fit["fitted_path"],
            impact_frame=split["impact_frame"],
            frame_size=frame_size,
            pitch_roi=pitch_roi,
        )
    else:
        reason = (
            "No impact detected; full observed path is pre-impact and nothing is projected."
            if not split["impact_detected"]
            else "Physics fit too weak to project after impact."
        )
        projection = {
            "projected_path": [],
            "projection_quality": QUALITY_UNAVAILABLE,
            "notes": [reason],
        }

    notes = [SAFETY_NOTE]
    if post:
        notes.append(
            f"Excluded {len(post)} post-impact observed points from delivery fitting."
        )
    notes.extend(fit.get("notes") or [])
    notes.extend(bounce.get("notes") or [])
    notes.extend(projection.get("notes") or [])

    return {
        "pre_impact_path": pre,
        "post_impact_observed_points": post,
        "fitted_delivery_path": fit["fitted_path"],
        "projected_path": projection["projected_path"],
        "bounce": fit["bounce"],
        "impact": {
            "impact_detected": split["impact_detected"],
            "impact_frame": split["impact_frame"],
            "impact_point": split["impact_point"],
        },
        "physics_quality": fit["physics_quality"],
        "projection_quality": projection["projection_quality"],
        "path_score": fit["path_score"],
        "notes": notes,
    }
