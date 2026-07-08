"""Cricket delivery path validity filter for safe trajectory drawing.

Pure Python/NumPy helpers. No Streamlit, YOLO, or model loading.
Drawing/3D/UI only — does not alter detector or tracker selection.
"""

from __future__ import annotations

from typing import Any

import numpy as np

QUALITY_UNAVAILABLE = "Unavailable"
QUALITY_POOR = "Poor"
QUALITY_PARTIAL = "Partial"
QUALITY_GOOD = "Good"

# ponytail: heuristic thresholds tuned for smartphone cricket clips; refine with labeled paths later.
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720
MIN_VALID_POINTS = 5
SIDEWAYS_RATIO = 1.7
SIDEWAYS_MIN_DX = 4.0
ROI_MARGIN_RATIO = 0.12
MAX_JUMP_DIAG_RATIO = 0.22
MIN_MAX_JUMP_PX = 25.0
SEGMENT_GAP_DIAG_RATIO = 0.18
MIN_SEGMENT_GAP_PX = 20.0
REVERSAL_MIN_PROGRESS = 8.0
PROJECTED_SOURCE = "projected_no_contact"
OBSERVED_SOURCE = "observed"
DEFAULT_PROJECT_FRAMES = 12
MIN_FIT_POINTS = 3
MAX_PROJECT_DISTANCE_DIAG_RATIO = 0.35
PROJECT_COLOR_NOTE = "Projected continuation assumes no bat contact."


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_points(points: Any) -> list[dict[str, Any]]:
    """Normalize mixed point formats to {x, y, frame_index?, source_index} dicts."""
    if points is None:
        return []
    if not isinstance(points, (list, tuple)):
        return []

    normalized: list[dict[str, Any]] = []
    for source_index, item in enumerate(points):
        parsed = _parse_point(item, source_index)
        if parsed is not None:
            normalized.append(parsed)
    return normalized


def _parse_point(item: Any, source_index: int) -> dict[str, Any] | None:
    if item is None:
        return None

    if isinstance(item, dict):
        if "center" in item and not ("x" in item and "y" in item):
            return _parse_point(item.get("center"), source_index)
        x = _safe_float(item.get("x"))
        y = _safe_float(item.get("y"))
        if x is None or y is None:
            center = item.get("center")
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                x = _safe_float(center[0])
                y = _safe_float(center[1])
        if x is None or y is None or np.isnan(x) or np.isnan(y):
            return None
        point = {
            "x": float(x),
            "y": float(y),
            "source_index": source_index,
            "source": str(item.get("source") or OBSERVED_SOURCE),
        }
        frame_index = _safe_int(item.get("frame_index"))
        if frame_index is not None:
            point["frame_index"] = frame_index
        return point

    if isinstance(item, (list, tuple)) and len(item) >= 2:
        x = _safe_float(item[0])
        y = _safe_float(item[1])
        if x is None or y is None or np.isnan(x) or np.isnan(y):
            return None
        return {
            "x": float(x),
            "y": float(y),
            "source_index": source_index,
            "source": OBSERVED_SOURCE,
        }

    return None


def extract_impact_frame(impact_info: Any) -> int | None:
    """Read impact frame index from impact_info-like payloads."""
    if impact_info is None:
        return None
    if isinstance(impact_info, (int, float)) and not isinstance(impact_info, bool):
        try:
            return int(impact_info)
        except (TypeError, ValueError):
            return None
    if isinstance(impact_info, dict):
        if impact_info.get("impact_detected") is False and impact_info.get("impact_frame") is None:
            return None
        for key in ("impact_frame", "frame_index", "possible_impact_frame"):
            frame = _safe_int(impact_info.get(key))
            if frame is not None:
                return frame
    return None


def truncate_to_pre_impact(points: Any, impact_frame: Any) -> dict[str, Any]:
    """Keep only points at or before impact_frame; drop real post-impact detections."""
    parsed = normalize_points(points)
    frame = extract_impact_frame(impact_frame)
    if frame is None:
        return {
            "points": parsed,
            "impact_frame": None,
            "truncated": False,
            "dropped_post_impact": [],
            "notes": ["No impact frame; using full observed path for validity."],
        }

    pre: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    has_frame_index = any("frame_index" in item for item in parsed)

    if has_frame_index:
        for item in parsed:
            item_frame = _safe_int(item.get("frame_index"))
            if item_frame is None:
                # Keep unlabeled mid-stream points only if surrounded by pre-impact context.
                pre.append(item)
                continue
            if item_frame <= frame:
                pre.append(item)
            else:
                dropped.append(_reject_payload(item, "post_impact_observation"))
    else:
        # ponytail: without frame_index, treat impact_frame as inclusive index into the track.
        cut = min(max(frame + 1, 0), len(parsed))
        pre = parsed[:cut]
        for item in parsed[cut:]:
            dropped.append(_reject_payload(item, "post_impact_observation"))

    notes = [
        f"Truncated path to pre-contact points at impact_frame={frame}.",
        f"Dropped {len(dropped)} post-impact observed points from drawing path.",
    ]
    return {
        "points": pre,
        "impact_frame": frame,
        "truncated": True,
        "dropped_post_impact": dropped,
        "notes": notes,
    }


def project_no_contact_continuation(
    pre_impact_points: Any,
    *,
    impact_frame: Any = None,
    frame_size: Any = None,
    stump_context: Any = None,
    project_frames: int = DEFAULT_PROJECT_FRAMES,
) -> dict[str, Any]:
    """Project a short no-contact delivery continuation from pre-impact dynamics."""
    try:
        return _project_no_contact_continuation_impl(
            pre_impact_points,
            impact_frame=impact_frame,
            frame_size=frame_size,
            stump_context=stump_context,
            project_frames=project_frames,
        )
    except Exception as exc:
        return {
            "projected_points": [],
            "used": False,
            "notes": [f"Projection failed safely: {exc}"],
        }


def _project_no_contact_continuation_impl(
    pre_impact_points: Any,
    *,
    impact_frame: Any = None,
    frame_size: Any = None,
    stump_context: Any = None,
    project_frames: int = DEFAULT_PROJECT_FRAMES,
) -> dict[str, Any]:
    points = normalize_points(pre_impact_points)
    notes: list[str] = []
    if len(points) < MIN_FIT_POINTS:
        return {
            "projected_points": [],
            "used": False,
            "notes": ["Not enough pre-impact points to project continuation."],
        }

    size = parse_frame_size(frame_size)
    width, height = size if size is not None else (DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT)
    direction_sign = _infer_direction_sign(stump_context, points)
    fit_window = points[-min(6, len(points)) :]

    # Build per-step times from frame_index when available; else unit steps.
    times: list[float] = []
    for index, item in enumerate(fit_window):
        frame_idx = _safe_int(item.get("frame_index"))
        times.append(float(frame_idx if frame_idx is not None else index))
    t0 = times[0]
    t = np.array([value - t0 for value in times], dtype=float)
    if float(np.ptp(t)) < 1e-6:
        t = np.arange(len(fit_window), dtype=float)
    xs = np.array([item["x"] for item in fit_window], dtype=float)
    ys = np.array([item["y"] for item in fit_window], dtype=float)

    # Linear velocity fit; quadratic fallback on y only when clearly curved & stable.
    vx, x_intercept = np.polyfit(t, xs, 1)
    vy, y_intercept = np.polyfit(t, ys, 1)
    if len(fit_window) >= 4:
        try:
            coef_y = np.polyfit(t, ys, 2)
            pred_y = np.polyval(coef_y, t)
            linear_y = vy * t + y_intercept
            if float(np.mean((pred_y - ys) ** 2)) < float(np.mean((linear_y - ys) ** 2)) * 0.85:
                # Use quadratic y with linear x for mild pitch arc.
                use_quad_y = True
            else:
                use_quad_y = False
                coef_y = None
        except Exception:
            use_quad_y = False
            coef_y = None
    else:
        use_quad_y = False
        coef_y = None

    # Prefer delivery direction: if fitted vy fights stump/pitch direction, flip to mean step.
    mean_dy = float(ys[-1] - ys[0]) / max(1.0, float(t[-1] - t[0]))
    if vy * direction_sign < 0 and mean_dy * direction_sign > 0:
        vy = mean_dy
        use_quad_y = False
        notes.append("Projection velocity aligned to pitch delivery direction.")

    last = points[-1]
    last_frame = _safe_int(last.get("frame_index"))
    if last_frame is None:
        impact = extract_impact_frame(impact_frame)
        last_frame = impact if impact is not None else int(last.get("source_index") or 0)

    n_project = max(0, min(int(project_frames or 0), DEFAULT_PROJECT_FRAMES * 2))
    if n_project <= 0:
        return {"projected_points": [], "used": False, "notes": ["Projection disabled."]}

    last_t = float(t[-1])

    # Anchor projection to last observed point to avoid discontinuous jump.
    x_at_last = float(vx * last_t + x_intercept)
    if use_quad_y and coef_y is not None:
        y_at_last = float(np.polyval(coef_y, last_t))
    else:
        y_at_last = float(vy * last_t + y_intercept)
    x_offset = float(last["x"]) - x_at_last
    y_offset = float(last["y"]) - y_at_last

    max_distance = max(40.0, ((width**2 + height**2) ** 0.5) * MAX_PROJECT_DISTANCE_DIAG_RATIO)
    traveled = 0.0
    projected: list[dict[str, Any]] = []
    prev_x, prev_y = float(last["x"]), float(last["y"])

    for step in range(1, n_project + 1):
        t_next = last_t + step
        x_next = float(vx * t_next + x_intercept) + x_offset
        if use_quad_y and coef_y is not None:
            y_next = float(np.polyval(coef_y, t_next)) + y_offset
        else:
            y_next = float(vy * t_next + y_intercept) + y_offset

        if x_next < 0 or y_next < 0 or x_next > width or y_next > height:
            notes.append("Stopped projection at frame boundary.")
            break
        step_dist = float(((x_next - prev_x) ** 2 + (y_next - prev_y) ** 2) ** 0.5)
        traveled += step_dist
        if traveled > max_distance:
            notes.append("Stopped projection at max travel distance.")
            break
        if step_dist < 0.5:
            notes.append("Stopped projection: near-zero motion.")
            break

        projected.append(
            {
                "x": round(x_next, 3),
                "y": round(y_next, 3),
                "frame_index": int(last_frame) + step,
                "source": PROJECTED_SOURCE,
                "source_index": None,
            }
        )
        prev_x, prev_y = x_next, y_next

    if projected:
        notes.append(PROJECT_COLOR_NOTE)
        notes.append(f"Projected {len(projected)} no-contact continuation points.")
    return {
        "projected_points": projected,
        "used": bool(projected),
        "notes": notes,
        "velocity": {"vx": float(vx), "vy": float(vy)},
    }


def points_as_xy(points: Any) -> list[tuple[int, int]]:
    """Return integer (x, y) tuples for drawing."""
    result: list[tuple[int, int]] = []
    for item in normalize_points(points):
        try:
            result.append((int(round(item["x"])), int(round(item["y"]))))
        except (TypeError, ValueError, KeyError):
            continue
    return result


def parse_frame_size(frame_size: Any) -> tuple[int, int] | None:
    if frame_size is None:
        return None
    if isinstance(frame_size, dict):
        width = _safe_int(frame_size.get("width") or frame_size.get("frame_width"))
        height = _safe_int(frame_size.get("height") or frame_size.get("frame_height"))
        if width and height and width > 0 and height > 0:
            return int(width), int(height)
        return None
    if isinstance(frame_size, (list, tuple)) and len(frame_size) >= 2:
        width = _safe_int(frame_size[0])
        height = _safe_int(frame_size[1])
        if width and height and width > 0 and height > 0:
            return int(width), int(height)
    return None


def _coerce_pitch_roi_bbox(pitch_roi: Any) -> tuple[float, float, float, float] | None:
    if pitch_roi is None:
        return None
    if isinstance(pitch_roi, dict):
        bbox = pitch_roi.get("bbox") or pitch_roi.get("roi_box")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return _clean_bbox(bbox)
        polygon = pitch_roi.get("polygon")
        if isinstance(polygon, (list, tuple)) and polygon:
            xs = []
            ys = []
            for vertex in polygon:
                if isinstance(vertex, (list, tuple)) and len(vertex) >= 2:
                    x = _safe_float(vertex[0])
                    y = _safe_float(vertex[1])
                    if x is not None and y is not None:
                        xs.append(x)
                        ys.append(y)
            if xs and ys:
                return min(xs), min(ys), max(xs), max(ys)
        return None
    if isinstance(pitch_roi, (list, tuple)) and len(pitch_roi) == 4:
        if all(isinstance(item, (int, float)) for item in pitch_roi):
            return _clean_bbox(pitch_roi)
    return None


def _clean_bbox(bbox: Any) -> tuple[float, float, float, float] | None:
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _infer_direction_sign(stump_context: Any, points: list[dict[str, Any]]) -> float:
    """Return +1 if delivery progresses toward increasing y, else -1."""
    if isinstance(stump_context, dict):
        batter_y = None
        bowler_y = None
        stumps = stump_context.get("stumps") or {}
        if isinstance(stumps, dict):
            batter = stumps.get("batter_end") or {}
            bowler = stumps.get("bowler_end") or {}
            if isinstance(batter, dict) and batter.get("center"):
                batter_y = _safe_float(batter["center"][1] if len(batter["center"]) > 1 else None)
            if isinstance(bowler, dict) and bowler.get("center"):
                bowler_y = _safe_float(bowler["center"][1] if len(bowler["center"]) > 1 else None)
        for key in ("batter_stumps", "bowler_stumps"):
            items = stump_context.get(key)
            if isinstance(items, list) and items:
                centers = [
                    _safe_float(item.get("center", [None, None])[1])
                    for item in items
                    if isinstance(item, dict) and item.get("center")
                ]
                centers = [c for c in centers if c is not None]
                if centers and key == "batter_stumps":
                    batter_y = float(np.mean(centers))
                if centers and key == "bowler_stumps":
                    bowler_y = float(np.mean(centers))
        if batter_y is not None and bowler_y is not None and abs(batter_y - bowler_y) > 1.0:
            return 1.0 if batter_y > bowler_y else -1.0

    if len(points) >= 2:
        dy = points[-1]["y"] - points[0]["y"]
        if abs(dy) > 1.0:
            return 1.0 if dy > 0 else -1.0
    return 1.0


def _default_max_gap_px(frame_size: tuple[int, int] | None) -> float:
    if frame_size is None:
        width, height = DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT
    else:
        width, height = frame_size
    diag = float((width**2 + height**2) ** 0.5)
    return max(MIN_MAX_JUMP_PX, diag * MAX_JUMP_DIAG_RATIO)


def _reject_payload(point: dict[str, Any], reason: str) -> dict[str, Any]:
    payload = {
        "x": round(float(point["x"]), 3),
        "y": round(float(point["y"]), 3),
        "reason": reason,
        "source_index": point.get("source_index"),
    }
    if "frame_index" in point:
        payload["frame_index"] = point["frame_index"]
    return payload


def validate_cricket_path(
    points: Any,
    frame_size: Any = None,
    pitch_roi: Any = None,
    stump_context: Any = None,
    max_gap_px: float | None = None,
) -> dict[str, Any]:
    """Filter tracked points that look implausible for a cricket delivery path."""
    try:
        return _validate_cricket_path_impl(
            points,
            frame_size=frame_size,
            pitch_roi=pitch_roi,
            stump_context=stump_context,
            max_gap_px=max_gap_px,
        )
    except Exception as exc:  # never crash drawing/UI consumers
        return {
            "is_valid": False,
            "quality": QUALITY_UNAVAILABLE,
            "valid_points": [],
            "rejected_points": [],
            "reason_summary": {"internal_error": 1},
            "notes": [f"Path validity check failed safely: {exc}"],
            "main_rejection_reason": "internal_error",
        }


def _validate_cricket_path_impl(
    points: Any,
    frame_size: Any = None,
    pitch_roi: Any = None,
    stump_context: Any = None,
    max_gap_px: float | None = None,
) -> dict[str, Any]:
    parsed = normalize_points(points)
    size = parse_frame_size(frame_size)
    roi_bbox = _coerce_pitch_roi_bbox(pitch_roi)
    notes: list[str] = []
    reason_summary: dict[str, int] = {}

    if len(parsed) < MIN_VALID_POINTS:
        quality = QUALITY_UNAVAILABLE if len(parsed) == 0 else QUALITY_POOR
        reason = "too_few_points"
        reason_summary[reason] = len(parsed)
        notes.append(f"Need at least {MIN_VALID_POINTS} points; got {len(parsed)}.")
        return {
            "is_valid": False,
            "quality": quality,
            "valid_points": [],
            "rejected_points": [_reject_payload(item, reason) for item in parsed],
            "reason_summary": reason_summary,
            "notes": notes,
            "main_rejection_reason": reason,
        }

    gap_limit = float(max_gap_px) if max_gap_px is not None else _default_max_gap_px(size)
    direction_sign = _infer_direction_sign(stump_context, parsed)
    if size is not None:
        width, height = size
    else:
        width, height = DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT
        notes.append("Frame size missing; using default bounds heuristics.")

    valid_points: list[dict[str, Any]] = []
    rejected_points: list[dict[str, Any]] = []

    for point in parsed:
        reason = None
        x, y = point["x"], point["y"]

        if size is not None and (x < 0 or y < 0 or x > width or y > height):
            reason = "out_of_bounds"
        elif roi_bbox is not None:
            x1, y1, x2, y2 = roi_bbox
            margin_x = max(8.0, (x2 - x1) * ROI_MARGIN_RATIO)
            margin_y = max(8.0, (y2 - y1) * ROI_MARGIN_RATIO)
            if not (
                x1 - margin_x <= x <= x2 + margin_x
                and y1 - margin_y <= y <= y2 + margin_y
            ):
                reason = "outside_pitch_corridor"

        if reason is None and valid_points:
            prev = valid_points[-1]
            dx = x - prev["x"]
            dy = y - prev["y"]
            dist = float((dx * dx + dy * dy) ** 0.5)
            if dist > gap_limit:
                reason = "huge_jump"
            elif abs(dx) > abs(dy) * SIDEWAYS_RATIO and abs(dx) > SIDEWAYS_MIN_DX:
                reason = "extreme_sideways"
            elif (
                dist >= REVERSAL_MIN_PROGRESS
                and dy * direction_sign < -1.0
                and abs(dy) > abs(dx) * 0.35
            ):
                reason = "sudden_reversal"

        if reason is not None:
            rejected_points.append(_reject_payload(point, reason))
            reason_summary[reason] = reason_summary.get(reason, 0) + 1
            continue

        kept = {
            "x": round(float(x), 3),
            "y": round(float(y), 3),
            "source_index": point.get("source_index"),
            "source": point.get("source") or OBSERVED_SOURCE,
        }
        if "frame_index" in point:
            kept["frame_index"] = point["frame_index"]
        valid_points.append(kept)

    if len(valid_points) < MIN_VALID_POINTS:
        quality = QUALITY_POOR if valid_points else QUALITY_UNAVAILABLE
        notes.append(
            f"Only {len(valid_points)} points survived cricket-validity checks."
        )
        main_reason = _main_rejection_reason(reason_summary) or "too_few_valid_points"
        if "too_few_valid_points" not in reason_summary:
            reason_summary["too_few_valid_points"] = len(valid_points)
        return {
            "is_valid": False,
            "quality": quality,
            "valid_points": valid_points,
            "rejected_points": rejected_points,
            "reason_summary": reason_summary,
            "notes": notes,
            "main_rejection_reason": main_reason,
        }

    reject_ratio = len(rejected_points) / max(1, len(parsed))
    if reject_ratio == 0 and len(valid_points) >= 8:
        quality = QUALITY_GOOD
        notes.append("Path looks like a coherent cricket delivery track.")
    elif reject_ratio <= 0.35 and len(valid_points) >= MIN_VALID_POINTS:
        quality = QUALITY_PARTIAL if reject_ratio > 0 else QUALITY_GOOD
        if reject_ratio > 0:
            notes.append("Some outliers rejected; remaining path is partially trusted.")
        else:
            notes.append("Short but coherent delivery path.")
    else:
        quality = QUALITY_PARTIAL
        notes.append("Multiple outliers rejected; use safe segments only.")

    return {
        "is_valid": True,
        "quality": quality,
        "valid_points": valid_points,
        "rejected_points": rejected_points,
        "reason_summary": reason_summary,
        "notes": notes,
        "main_rejection_reason": _main_rejection_reason(reason_summary),
    }


def _main_rejection_reason(reason_summary: dict[str, int]) -> str | None:
    if not reason_summary:
        return None
    return max(reason_summary.items(), key=lambda item: item[1])[0]


def build_safe_trajectory_segments(
    valid_points: Any,
    frame_size: Any = None,
    max_segment_gap_px: float | None = None,
) -> dict[str, Any]:
    """Split valid points into drawable segments without bridging huge gaps."""
    try:
        return _build_safe_trajectory_segments_impl(
            valid_points,
            frame_size=frame_size,
            max_segment_gap_px=max_segment_gap_px,
        )
    except Exception as exc:
        return {
            "segments": [],
            "draw_allowed": False,
            "fit_quality": QUALITY_UNAVAILABLE,
            "notes": [f"Safe segment build failed safely: {exc}"],
        }


def _build_safe_trajectory_segments_impl(
    valid_points: Any,
    frame_size: Any = None,
    max_segment_gap_px: float | None = None,
) -> dict[str, Any]:
    points = normalize_points(valid_points)
    notes: list[str] = []
    if len(points) < 2:
        return {
            "segments": [],
            "draw_allowed": False,
            "fit_quality": QUALITY_UNAVAILABLE if not points else QUALITY_POOR,
            "notes": ["Not enough valid points to draw a trajectory."],
        }

    size = parse_frame_size(frame_size)
    if max_segment_gap_px is not None:
        gap_limit = float(max_segment_gap_px)
    else:
        if size is None:
            width, height = DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT
        else:
            width, height = size
        diag = float((width**2 + height**2) ** 0.5)
        gap_limit = max(MIN_SEGMENT_GAP_PX, diag * SEGMENT_GAP_DIAG_RATIO)

    raw_segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = [points[0]]
    for point in points[1:]:
        prev = current[-1]
        dist = float(
            ((point["x"] - prev["x"]) ** 2 + (point["y"] - prev["y"]) ** 2) ** 0.5
        )
        if dist > gap_limit:
            if len(current) >= 2:
                raw_segments.append(current)
            elif len(current) == 1:
                notes.append("Dropped isolated valid point between gaps.")
            current = [point]
        else:
            current.append(point)
    if len(current) >= 2:
        raw_segments.append(current)
    elif len(current) == 1:
        notes.append("Trailing isolated point was not drawn.")

    if not raw_segments:
        return {
            "segments": [],
            "draw_allowed": False,
            "fit_quality": QUALITY_POOR,
            "notes": notes + ["No contiguous segment long enough to draw."],
        }

    segments = [_maybe_smooth_segment(segment) for segment in raw_segments]
    # Fallback if smoothing produced loops / instability
    if any(_segment_looks_unstable(raw, smooth) for raw, smooth in zip(raw_segments, segments)):
        segments = raw_segments
        notes.append("Mild smoothing looked unstable; using raw valid segments.")
    else:
        notes.append("Built safe segments without bridging large gaps.")

    total_points = sum(len(segment) for segment in segments)
    if len(segments) == 1 and total_points >= 8:
        fit_quality = QUALITY_GOOD
    elif total_points >= MIN_VALID_POINTS:
        fit_quality = QUALITY_PARTIAL if len(segments) > 1 else QUALITY_GOOD
    else:
        fit_quality = QUALITY_PARTIAL

    return {
        "segments": segments,
        "draw_allowed": True,
        "fit_quality": fit_quality,
        "notes": notes,
    }


def _maybe_smooth_segment(segment: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(segment) < 3:
        return segment
    xs = np.array([item["x"] for item in segment], dtype=float)
    ys = np.array([item["y"] for item in segment], dtype=float)
    kernel = np.array([0.25, 0.5, 0.25], dtype=float)
    smooth_x = np.convolve(xs, kernel, mode="same")
    smooth_y = np.convolve(ys, kernel, mode="same")
    smooth_x[0], smooth_x[-1] = xs[0], xs[-1]
    smooth_y[0], smooth_y[-1] = ys[0], ys[-1]

    smoothed: list[dict[str, Any]] = []
    for index, item in enumerate(segment):
        point = {
            "x": round(float(smooth_x[index]), 3),
            "y": round(float(smooth_y[index]), 3),
            "source_index": item.get("source_index"),
            "source": item.get("source") or OBSERVED_SOURCE,
        }
        if "frame_index" in item:
            point["frame_index"] = item["frame_index"]
        smoothed.append(point)
    return smoothed


def _segment_looks_unstable(
    raw: list[dict[str, Any]],
    smooth: list[dict[str, Any]],
) -> bool:
    if len(raw) != len(smooth) or len(raw) < 3:
        return False
    for index in range(len(raw)):
        dx = smooth[index]["x"] - raw[index]["x"]
        dy = smooth[index]["y"] - raw[index]["y"]
        if (dx * dx + dy * dy) ** 0.5 > 18.0:
            return True
    # Detect simple loop: later point nearly coincides with earlier non-adjacent point
    for i in range(len(smooth)):
        for j in range(i + 3, len(smooth)):
            dx = smooth[i]["x"] - smooth[j]["x"]
            dy = smooth[i]["y"] - smooth[j]["y"]
            if (dx * dx + dy * dy) ** 0.5 < 2.0:
                return True
    return False


def prepare_safe_trajectory_for_draw(
    points: Any,
    frame_size: Any = None,
    pitch_roi: Any = None,
    stump_context: Any = None,
    max_gap_px: float | None = None,
    max_segment_gap_px: float | None = None,
    impact_info: Any = None,
    project_frames: int = DEFAULT_PROJECT_FRAMES,
) -> dict[str, Any]:
    """Run validity + optional pre-impact truncate/project + segment build for draw/UI/3D."""
    truncate = truncate_to_pre_impact(points, impact_info)
    working_points = truncate.get("points") or []
    impact_frame = truncate.get("impact_frame")

    validity = validate_cricket_path(
        working_points,
        frame_size=frame_size,
        pitch_roi=pitch_roi,
        stump_context=stump_context,
        max_gap_px=max_gap_px,
    )

    # Fold post-impact drops into rejection summary for honest UI metrics.
    dropped_post = truncate.get("dropped_post_impact") or []
    if dropped_post:
        rejected = list(validity.get("rejected_points") or [])
        rejected.extend(dropped_post)
        reason_summary = dict(validity.get("reason_summary") or {})
        reason_summary["post_impact_observation"] = reason_summary.get(
            "post_impact_observation", 0
        ) + len(dropped_post)
        notes = list(validity.get("notes") or []) + list(truncate.get("notes") or [])
        validity = {
            **validity,
            "rejected_points": rejected,
            "reason_summary": reason_summary,
            "notes": notes,
            "main_rejection_reason": validity.get("main_rejection_reason")
            or "post_impact_observation",
        }

    projection = {
        "projected_points": [],
        "used": False,
        "notes": ["No impact detected; projection not applied."],
    }
    observed_valid = [
        point
        for point in (validity.get("valid_points") or [])
        if point.get("source") != PROJECTED_SOURCE
    ]
    if impact_frame is not None and validity.get("is_valid"):
        projection = project_no_contact_continuation(
            observed_valid,
            impact_frame=impact_frame,
            frame_size=frame_size,
            stump_context=stump_context,
            project_frames=project_frames,
        )

    combined_points = list(observed_valid)
    projected_points = list(projection.get("projected_points") or [])
    if projected_points:
        combined_points.extend(projected_points)
        validity = {
            **validity,
            "notes": list(validity.get("notes") or []) + list(projection.get("notes") or []),
        }

    # Segments for observed pre-contact only; projected is a separate drawable segment.
    segments_result = build_safe_trajectory_segments(
        observed_valid,
        frame_size=frame_size,
        max_segment_gap_px=max_segment_gap_px,
    )
    if validity.get("quality") in {QUALITY_UNAVAILABLE, QUALITY_POOR}:
        segments_result = {
            **segments_result,
            "draw_allowed": False,
            "fit_quality": validity.get("quality") or QUALITY_UNAVAILABLE,
            "segments": [],
            "notes": list(segments_result.get("notes") or [])
            + ["Drawing disabled because path validity is Poor/Unavailable."],
        }
        projected_points = []

    observed_draw_segments = [
        points_as_xy(segment) for segment in (segments_result.get("segments") or [])
    ]
    observed_draw_segments = [segment for segment in observed_draw_segments if len(segment) >= 2]
    projected_xy = points_as_xy(projected_points)
    projected_draw_segments = [projected_xy] if len(projected_xy) >= 2 else []

    draw_allowed = bool(segments_result.get("draw_allowed")) and bool(observed_draw_segments)
    labels = []
    if draw_allowed:
        labels.append("Pre-contact delivery path")
    if projected_draw_segments:
        labels.append("Projected continuation (no bat contact)")

    return {
        "validity": validity,
        "segments_result": segments_result,
        "draw_allowed": draw_allowed,
        "draw_segments": observed_draw_segments,
        "projected_draw_segments": projected_draw_segments,
        "projected_points": projected_points,
        "projection_used": bool(projected_draw_segments),
        "impact_frame": impact_frame,
        "valid_xy": points_as_xy(combined_points if projected_points else observed_valid),
        "observed_valid_xy": points_as_xy(observed_valid),
        "rejected_count": len(validity.get("rejected_points") or []),
        "quality": validity.get("quality") or QUALITY_UNAVAILABLE,
        "main_rejection_reason": validity.get("main_rejection_reason"),
        "labels": labels,
        "ui_summary": {
            "path_validity": validity.get("quality") or QUALITY_UNAVAILABLE,
            "is_valid": bool(validity.get("is_valid")),
            "valid_points": len(observed_valid),
            "rejected_points": len(validity.get("rejected_points") or []),
            "draw_allowed": draw_allowed,
            "main_rejection_reason": validity.get("main_rejection_reason"),
            "reason_summary": validity.get("reason_summary") or {},
            "impact_frame": impact_frame,
            "projection_used": bool(projected_draw_segments),
            "projected_points": len(projected_points),
            "path_mode": (
                "pre_contact_plus_projection"
                if projected_draw_segments
                else ("pre_contact" if impact_frame is not None else "full_path")
            ),
            "labels": labels,
        },
    }
