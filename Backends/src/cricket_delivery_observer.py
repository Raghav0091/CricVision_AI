"""Experimental observer layer for cricket delivery path selection.

Display/debug only: this module does not modify detector, tracker, or report contracts.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _extract_bbox(detection: dict[str, Any]) -> tuple[float, float, float, float] | None:
    box = detection.get("bbox") or detection.get("box") or detection.get("xyxy")
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        x1, y1, x2, y2 = [_safe_float(item, np.nan) for item in box[:4]]
    elif all(key in detection for key in ("x1", "y1", "x2", "y2")):
        x1 = _safe_float(detection.get("x1"), np.nan)
        y1 = _safe_float(detection.get("y1"), np.nan)
        x2 = _safe_float(detection.get("x2"), np.nan)
        y2 = _safe_float(detection.get("y2"), np.nan)
    else:
        return None
    if any(np.isnan([x1, y1, x2, y2])):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _extract_center(detection: dict[str, Any]) -> tuple[float, float] | None:
    center = detection.get("center")
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        x = _safe_float(center[0], np.nan)
        y = _safe_float(center[1], np.nan)
        if not np.isnan(x) and not np.isnan(y):
            return x, y
    bbox = _extract_bbox(detection)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _coerce_pitch_roi_bbox(pitch_roi: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(pitch_roi, dict):
        return None
    bbox = pitch_roi.get("bbox") or pitch_roi.get("roi_box")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x1, y1, x2, y2 = [_safe_float(item, np.nan) for item in bbox[:4]]
        if not any(np.isnan([x1, y1, x2, y2])):
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            return x1, y1, x2, y2
    return None


def extract_ball_candidates_from_frame_detections(frame_detections: Any) -> list[dict[str, Any]]:
    """Safely extract per-frame raw ball candidates from timeline-like detections."""
    if not isinstance(frame_detections, (list, tuple)):
        return []
    candidates: list[dict[str, Any]] = []
    for frame_idx, frame_item in enumerate(frame_detections):
        if not isinstance(frame_item, dict):
            continue
        frame_index = _safe_int(frame_item.get("frame_index"), frame_idx)
        detections = frame_item.get("ball_detections") or frame_item.get("balls") or []
        if not isinstance(detections, (list, tuple)):
            continue
        for det in detections:
            if not isinstance(det, dict):
                continue
            center = _extract_center(det)
            if center is None:
                continue
            x, y = center
            confidence = _safe_float(det.get("confidence"), 0.0)
            candidate: dict[str, Any] = {
                "frame_index": frame_index,
                "x": float(x),
                "y": float(y),
                "confidence": min(max(confidence, 0.0), 1.0),
                "source": "raw_detection",
            }
            bbox = _extract_bbox(det)
            if bbox is not None:
                candidate["bbox"] = [round(v, 3) for v in bbox]
            candidates.append(candidate)
    return candidates


def select_best_cricket_path(
    candidates: Any,
    frame_size: Any = None,
    pitch_roi: Any = None,
    stump_context: Any = None,
    max_frame_gap: int = 12,
) -> dict[str, Any]:
    """Select a plausible smooth forward delivery path from raw candidates."""
    candidate_list = candidates if isinstance(candidates, list) else []
    if len(candidate_list) < 5:
        return {
            "observer_path": [],
            "rejected_candidates": _reject_all(candidate_list, "too_few_candidates"),
            "path_quality": "Unavailable",
            "path_score": 0.0,
            "reason_summary": {"too_few_candidates": len(candidate_list)},
            "notes": ["Need at least 5 candidates for observer path selection."],
        }

    width, height = _parse_frame_size(frame_size)
    roi_bbox = _coerce_pitch_roi_bbox(pitch_roi)
    direction_sign = _infer_direction_sign(stump_context)
    max_gap = max(1, _safe_int(max_frame_gap, 12))
    diag = float((width**2 + height**2) ** 0.5)
    max_jump = max(25.0, diag * 0.22)

    valid = [c for c in candidate_list if _valid_candidate(c)]
    valid.sort(key=lambda item: (item["frame_index"], item["x"], item["y"]))
    if len(valid) < 5:
        return {
            "observer_path": [],
            "rejected_candidates": _reject_all(candidate_list, "invalid_candidates"),
            "path_quality": "Unavailable",
            "path_score": 0.0,
            "reason_summary": {"invalid_candidates": len(candidate_list)},
            "notes": ["Candidates were malformed or missing coordinates."],
        }

    dp_score = np.full(len(valid), -1e9, dtype=float)
    prev_idx = np.full(len(valid), -1, dtype=int)
    node_penalties: list[list[str]] = [[] for _ in valid]

    for i, cur in enumerate(valid):
        base = 1.0 + min(max(_safe_float(cur.get("confidence"), 0.0), 0.0), 1.0) * 2.0
        penalties = []
        if roi_bbox is not None and not _inside_bbox(cur["x"], cur["y"], roi_bbox):
            base -= 1.8
            penalties.append("outside_roi")
        if not _inside_bbox(cur["x"], cur["y"], (0.0, 0.0, float(width), float(height))):
            base -= 2.0
            penalties.append("outside_frame")
        dp_score[i] = base
        node_penalties[i] = penalties

        for j in range(i):
            prv = valid[j]
            dt = cur["frame_index"] - prv["frame_index"]
            if dt <= 0 or dt > max_gap:
                continue
            dx = cur["x"] - prv["x"]
            dy = cur["y"] - prv["y"]
            dist = float((dx * dx + dy * dy) ** 0.5)

            trans = 2.2
            reason_penalties = []
            if dist > max_jump:
                trans -= 2.4
                reason_penalties.append("huge_jump")
            if dy * direction_sign < -1.0:
                trans -= 1.8
                reason_penalties.append("backward")
            if abs(dx) > abs(dy) * 1.7 and abs(dx) > 4.0:
                trans -= 1.3
                reason_penalties.append("sideways")
            if dt > 3:
                trans -= 0.2 * (dt - 3)
            score = dp_score[j] + trans + base
            if score > dp_score[i]:
                dp_score[i] = score
                prev_idx[i] = j
                node_penalties[i] = penalties + reason_penalties

    best_idx = int(np.argmax(dp_score))
    path_indices = []
    idx = best_idx
    while idx >= 0:
        path_indices.append(idx)
        idx = int(prev_idx[idx])
    path_indices.reverse()

    observer_path = [_format_path_point(valid[i]) for i in path_indices]
    if len(observer_path) < 5:
        return {
            "observer_path": [],
            "rejected_candidates": _reject_all(candidate_list, "isolated_or_short_path"),
            "path_quality": "Poor",
            "path_score": 0.0,
            "reason_summary": {"isolated_or_short_path": len(candidate_list)},
            "notes": ["Best path remained too short after plausibility filtering."],
        }

    selected_set = {(p["frame_index"], p["x"], p["y"]) for p in observer_path}
    rejected_candidates = []
    reason_summary: dict[str, int] = {}
    for cand in valid:
        key = (int(cand["frame_index"]), round(float(cand["x"]), 3), round(float(cand["y"]), 3))
        selected_key = (key[0], key[1], key[2])
        if selected_key in selected_set:
            continue
        reason = _candidate_reject_reason(cand, roi_bbox, width, height)
        rejected_candidates.append(
            {
                "frame_index": int(cand["frame_index"]),
                "x": round(float(cand["x"]), 3),
                "y": round(float(cand["y"]), 3),
                "confidence": round(float(cand.get("confidence", 0.0)), 3),
                "reason": reason,
            }
        )
        reason_summary[reason] = reason_summary.get(reason, 0) + 1

    path_score = _compute_path_score(observer_path, width=width, height=height, direction_sign=direction_sign)
    path_quality = _path_quality(path_score, len(observer_path))
    notes = [
        f"Selected {len(observer_path)} points from {len(valid)} valid candidates.",
        "Heuristic penalties applied for backward, sideways, large-jump, and out-of-ROI points.",
    ]
    return {
        "observer_path": observer_path,
        "rejected_candidates": rejected_candidates,
        "path_quality": path_quality,
        "path_score": float(round(path_score, 4)),
        "reason_summary": reason_summary,
        "notes": notes,
    }


def fit_observer_path(observer_path: Any, frame_size: Any = None) -> dict[str, Any]:
    """Apply safe smoothing over observer path with conservative fallback."""
    if not isinstance(observer_path, list) or len(observer_path) < 5:
        return {
            "fitted_path": [],
            "fit_quality": "Unavailable",
            "notes": ["Not enough observer points for fitting."],
        }

    width, height = _parse_frame_size(frame_size)
    frames = np.array([_safe_int(item.get("frame_index"), idx) for idx, item in enumerate(observer_path)], dtype=int)
    xs = np.array([_safe_float(item.get("x"), np.nan) for item in observer_path], dtype=float)
    ys = np.array([_safe_float(item.get("y"), np.nan) for item in observer_path], dtype=float)

    if np.isnan(xs).any() or np.isnan(ys).any():
        return {
            "fitted_path": [_format_path_point(item) for item in observer_path if _valid_candidate(item)],
            "fit_quality": "Poor",
            "notes": ["Observer path contains invalid coordinates; using raw observer path."],
        }

    kernel = np.array([0.25, 0.5, 0.25], dtype=float)
    smooth_x = np.convolve(xs, kernel, mode="same")
    smooth_y = np.convolve(ys, kernel, mode="same")
    smooth_x[0], smooth_x[-1] = xs[0], xs[-1]
    smooth_y[0], smooth_y[-1] = ys[0], ys[-1]

    jitter = np.sqrt((smooth_x - xs) ** 2 + (smooth_y - ys) ** 2)
    max_allowed = max(6.0, (width**2 + height**2) ** 0.5 * 0.08)
    if float(np.nanmax(jitter)) > max_allowed:
        return {
            "fitted_path": [_format_path_point(item) for item in observer_path if _valid_candidate(item)],
            "fit_quality": "Poor",
            "notes": ["Fit looked unstable; fell back to observer path."],
        }

    fitted = []
    for i in range(len(frames)):
        fitted.append(
            {
                "frame_index": int(frames[i]),
                "x": round(float(smooth_x[i]), 3),
                "y": round(float(smooth_y[i]), 3),
                "confidence": round(_safe_float(observer_path[i].get("confidence"), 0.0), 3),
                "source": "observer_fit",
            }
        )

    fit_quality = "Good" if len(fitted) >= 8 else "Partial"
    return {
        "fitted_path": fitted,
        "fit_quality": fit_quality,
        "notes": ["Applied conservative 3-point smoothing with endpoint anchoring."],
    }


def _parse_frame_size(frame_size: Any) -> tuple[int, int]:
    if isinstance(frame_size, dict):
        width = _safe_int(frame_size.get("width") or frame_size.get("frame_width"), 1280)
        height = _safe_int(frame_size.get("height") or frame_size.get("frame_height"), 720)
        return max(1, width), max(1, height)
    if isinstance(frame_size, (list, tuple)) and len(frame_size) >= 2:
        return max(1, _safe_int(frame_size[0], 1280)), max(1, _safe_int(frame_size[1], 720))
    return 1280, 720


def _valid_candidate(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    try:
        float(item.get("x"))
        float(item.get("y"))
    except (TypeError, ValueError):
        return False
    return True


def _inside_bbox(x: float, y: float, bbox: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _infer_direction_sign(stump_context: Any) -> float:
    if not isinstance(stump_context, dict):
        return 1.0
    view = str(stump_context.get("camera_view") or stump_context.get("view") or "").strip().lower()
    return -1.0 if "batter" in view else 1.0


def _format_path_point(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "frame_index": _safe_int(item.get("frame_index"), 0),
        "x": round(_safe_float(item.get("x"), 0.0), 3),
        "y": round(_safe_float(item.get("y"), 0.0), 3),
        "confidence": round(_safe_float(item.get("confidence"), 0.0), 3),
        "source": item.get("source") or "raw_detection",
    }


def _compute_path_score(
    observer_path: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    direction_sign: float,
) -> float:
    if len(observer_path) < 5:
        return 0.0
    conf = np.mean([_safe_float(item.get("confidence"), 0.0) for item in observer_path])
    penalties = 0.0
    diag = float((width**2 + height**2) ** 0.5)
    max_jump = max(25.0, diag * 0.25)
    for i in range(1, len(observer_path)):
        dx = observer_path[i]["x"] - observer_path[i - 1]["x"]
        dy = observer_path[i]["y"] - observer_path[i - 1]["y"]
        dist = float((dx * dx + dy * dy) ** 0.5)
        if dy * direction_sign < -1.0:
            penalties += 0.25
        if abs(dx) > abs(dy) * 1.6 and abs(dx) > 4.0:
            penalties += 0.2
        if dist > max_jump:
            penalties += 0.35
    raw = 0.55 * conf + 0.45 * (1.0 - min(1.0, penalties / max(1.0, len(observer_path) - 1)))
    return float(max(0.0, min(1.0, raw)))


def _path_quality(score: float, point_count: int) -> str:
    if point_count < 5:
        return "Unavailable"
    if score >= 0.72 and point_count >= 8:
        return "Good"
    if score >= 0.5:
        return "Partial"
    return "Poor"


def _candidate_reject_reason(
    candidate: dict[str, Any],
    roi_bbox: tuple[float, float, float, float] | None,
    width: int,
    height: int,
) -> str:
    if roi_bbox is not None and not _inside_bbox(candidate["x"], candidate["y"], roi_bbox):
        return "outside_roi"
    if not _inside_bbox(candidate["x"], candidate["y"], (0.0, 0.0, float(width), float(height))):
        return "outside_frame"
    if _safe_float(candidate.get("confidence"), 0.0) < 0.2:
        return "low_confidence"
    return "not_selected"


def _reject_all(candidates: list[dict[str, Any]], reason: str) -> list[dict[str, Any]]:
    output = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        output.append(
            {
                "frame_index": _safe_int(item.get("frame_index"), 0),
                "x": round(_safe_float(item.get("x"), 0.0), 3),
                "y": round(_safe_float(item.get("y"), 0.0), 3),
                "confidence": round(_safe_float(item.get("confidence"), 0.0), 3),
                "reason": reason,
            }
        )
    return output
