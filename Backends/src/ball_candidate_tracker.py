"""Ball Candidate Reliability Tracker v1 — app-side detection cleanup.

Pure Python helpers. No Streamlit, YOLO, OpenCV, or model loading.
Selects the most believable observed ball point per frame from raw
detections. Never invents points, never interpolates gaps as observed —
a missing frame is always better than a wrong point.
"""

from __future__ import annotations

from math import hypot, isfinite
from typing import Any

from Backends.src.pitch_calibration import (
    normalize_pitch_roi,
    score_point_against_pitch_roi,
)

QUALITY_UNAVAILABLE = "Unavailable"
QUALITY_POOR = "Poor"
QUALITY_PARTIAL = "Partial"
QUALITY_GOOD = "Good"

# ponytail: heuristic thresholds tuned for smartphone cricket clips; mirrors cricket_path_validity.
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720
MAX_JUMP_DIAG_RATIO = 0.22
MIN_MAX_JUMP_PX = 25.0
SIDEWAYS_RATIO = 1.7
SIDEWAYS_MIN_DX = 6.0
REVERSAL_MIN_DY = 12.0
MAX_GAP_FRAMES = 6
ACCEPT_SCORE = 0.35
MAX_BALL_SIDE_RATIO = 0.12  # ball bbox side vs shorter frame side
PITCH_ROI_MARGIN_PX = 40.0
MIN_TRACK_POINTS = 5
GOOD_MIN_POINTS = 10
GOOD_MAX_REJECT_RATIO = 0.35
POOR_REJECT_RATIO = 0.7


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or isinstance(value, bool):
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


def _parse_pair(pair: Any) -> tuple[float, float] | None:
    if isinstance(pair, dict):
        x = _safe_float(pair.get("x"))
        y = _safe_float(pair.get("y"))
    elif isinstance(pair, (list, tuple)) and len(pair) >= 2:
        x = _safe_float(pair[0])
        y = _safe_float(pair[1])
    else:
        return None
    if x is None or y is None:
        return None
    return x, y


def _parse_bbox(detection: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = detection.get("bbox") or detection.get("box") or detection.get("xyxy")
    if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
        return None
    values = [_safe_float(value) for value in bbox[:4]]
    if None in values:
        return None
    x1, y1, x2, y2 = values
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _parse_candidate(item: Any, frame_index: Any = None) -> dict[str, Any] | None:
    """Parse one raw detection/candidate into the common candidate dict."""
    if isinstance(item, dict):
        x = _safe_float(item.get("x"))
        y = _safe_float(item.get("y"))
        if x is None or y is None:
            for key in ("center", "centroid"):
                pair = _parse_pair(item.get(key))
                if pair is not None:
                    x, y = pair
                    break
        bbox = _parse_bbox(item)
        if (x is None or y is None) and bbox is not None:
            x = (bbox[0] + bbox[2]) / 2.0
            y = (bbox[1] + bbox[3]) / 2.0
        if x is None or y is None:
            return None
        candidate: dict[str, Any] = {
            "frame_index": _safe_int(item.get("frame_index"), _safe_int(frame_index)),
            "x": float(x),
            "y": float(y),
            "confidence": _safe_float(item.get("confidence"), _safe_float(item.get("conf"))),
            "source": str(item.get("source")) if item.get("source") else "raw_detection",
        }
        if bbox is not None:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            candidate["bbox"] = [round(v, 3) for v in bbox]
            candidate["width"] = round(width, 3)
            candidate["height"] = round(height, 3)
            candidate["area"] = round(width * height, 3)
        return candidate

    if isinstance(item, (list, tuple)):
        if len(item) >= 3:
            frame = _safe_int(item[0], _safe_int(frame_index))
            x = _safe_float(item[1])
            y = _safe_float(item[2])
            confidence = _safe_float(item[3]) if len(item) > 3 else None
        elif len(item) == 2:
            frame = _safe_int(frame_index)
            x = _safe_float(item[0])
            y = _safe_float(item[1])
            confidence = None
        else:
            return None
        if x is None or y is None:
            return None
        return {
            "frame_index": frame,
            "x": float(x),
            "y": float(y),
            "confidence": confidence,
            "source": "raw_detection",
        }

    return None


def normalize_detection_candidates(raw_detections: Any) -> list[dict[str, Any]]:
    """Convert mixed raw detection/candidate formats into safe candidate dicts."""
    if not isinstance(raw_detections, (list, tuple)):
        return []
    candidates: list[dict[str, Any]] = []
    for item in raw_detections:
        candidate = _parse_candidate(item)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _frame_bounds(frame_size: Any) -> tuple[float, float, float]:
    if isinstance(frame_size, dict):
        width = _safe_float(frame_size.get("width") or frame_size.get("frame_width"))
        height = _safe_float(frame_size.get("height") or frame_size.get("frame_height"))
    elif isinstance(frame_size, (list, tuple)) and len(frame_size) >= 2:
        width = _safe_float(frame_size[0])
        height = _safe_float(frame_size[1])
    else:
        width = height = None
    if not width or not height or width <= 0 or height <= 0:
        width, height = float(DEFAULT_FRAME_WIDTH), float(DEFAULT_FRAME_HEIGHT)
    return float(width), float(height), hypot(width, height)


def _motion_context(previous_points: Any) -> list[dict[str, Any]]:
    points = []
    for item in previous_points or []:
        candidate = _parse_candidate(item)
        if candidate is not None:
            points.append(candidate)
    return points


def score_ball_candidate(
    candidate: Any,
    previous_points: Any = None,
    frame_size: Any = None,
    pitch_roi: Any = None,
) -> dict[str, Any]:
    """Score how believable one candidate is as the cricket ball.

    High confidence never rescues physically impossible movement.
    """
    reasons: list[str] = []
    parsed = _parse_candidate(candidate)
    if parsed is None:
        return {"score": 0.0, "accepted": False, "reasons": ["invalid_candidate"]}

    width, height, diag = _frame_bounds(frame_size)
    prev = _motion_context(previous_points)
    score = 0.2
    hard_reject = False

    x, y = parsed["x"], parsed["y"]
    if x < 0 or y < 0 or x > width or y > height:
        hard_reject = True
        reasons.append("outside_frame")

    confidence = parsed.get("confidence")
    if confidence is not None:
        score += 0.3 * min(max(confidence, 0.0), 1.0)
        reasons.append(f"confidence={round(confidence, 3)}")
    else:
        reasons.append("confidence_unknown")

    side = max(parsed.get("width") or 0.0, parsed.get("height") or 0.0)
    if side > 0:
        if side > min(width, height) * MAX_BALL_SIDE_RATIO:
            hard_reject = True
            reasons.append("implausible_size")
        else:
            score += 0.1
            reasons.append("plausible_size")

    roi = normalize_pitch_roi(pitch_roi, frame_size=frame_size)
    if roi["available"]:
        pitch_score = score_point_against_pitch_roi(parsed, roi, margin=PITCH_ROI_MARGIN_PX)
        score += pitch_score["score_bonus"]
        if pitch_score["inside"]:
            reasons.append("inside_pitch_roi")
        elif pitch_score["near"]:
            reasons.append("near_pitch_roi")
        else:
            hard_reject = True
            reasons.append("off_pitch")

    if prev:
        last = prev[-1]
        gap = 1
        if parsed.get("frame_index") is not None and last.get("frame_index") is not None:
            gap = max(1, parsed["frame_index"] - last["frame_index"])
        gap = min(gap, MAX_GAP_FRAMES)
        jump_limit = max(MIN_MAX_JUMP_PX, diag * MAX_JUMP_DIAG_RATIO) * gap
        dx = x - last["x"]
        dy = y - last["y"]
        dist = hypot(dx, dy)

        if dist > jump_limit:
            hard_reject = True
            reasons.append("huge_jump")
        elif abs(dx) > abs(dy) * SIDEWAYS_RATIO and abs(dx) > SIDEWAYS_MIN_DX * gap:
            hard_reject = True
            reasons.append("impossible_sideways")
        else:
            score += 0.2 * (1.0 - dist / jump_limit)
            reasons.append("smooth_movement")

        if not hard_reject and len(prev) >= 2:
            overall_dy = prev[-1]["y"] - prev[0]["y"]
            if abs(overall_dy) > 2.0:
                sign = 1.0 if overall_dy > 0 else -1.0
                if dy * sign < 0 and abs(dy) > max(REVERSAL_MIN_DY * gap, abs(dx)):
                    hard_reject = True
                    reasons.append("impossible_reversal")

        if not hard_reject and len(prev) >= 2:
            second_last = prev[-2]
            step = 1
            if last.get("frame_index") is not None and second_last.get("frame_index") is not None:
                step = max(1, last["frame_index"] - second_last["frame_index"])
            vx = (last["x"] - second_last["x"]) / step
            vy = (last["y"] - second_last["y"]) / step
            pred_x = last["x"] + vx * gap
            pred_y = last["y"] + vy * gap
            pred_dist = hypot(x - pred_x, y - pred_y)
            if pred_dist <= jump_limit:
                score += 0.2 * (1.0 - pred_dist / jump_limit)
                reasons.append("close_to_predicted")
            else:
                score -= 0.2
                reasons.append("far_from_predicted")

    score = max(0.0, min(1.0, score))
    if hard_reject:
        return {"score": round(min(score, ACCEPT_SCORE), 3), "accepted": False, "reasons": reasons}
    if score < ACCEPT_SCORE:
        reasons.append("low_reliability_score")
        return {"score": round(score, 3), "accepted": False, "reasons": reasons}
    return {"score": round(score, 3), "accepted": True, "reasons": reasons}


def select_best_candidate_for_frame(
    candidates: Any,
    previous_points: Any = None,
    frame_size: Any = None,
    pitch_roi: Any = None,
) -> dict[str, Any]:
    """Pick the most believable ball candidate for one frame, or nothing."""
    notes: list[str] = []
    parsed = normalize_detection_candidates(candidates)
    if not parsed:
        return {
            "selected": None,
            "rejected": [],
            "selection_quality": QUALITY_UNAVAILABLE,
            "notes": ["No usable candidates in this frame."],
        }

    scored = []
    for candidate in parsed:
        verdict = score_ball_candidate(
            candidate,
            previous_points=previous_points,
            frame_size=frame_size,
            pitch_roi=pitch_roi,
        )
        scored.append({**candidate, "score": verdict["score"], "reasons": verdict["reasons"], "accepted": verdict["accepted"]})

    accepted = [item for item in scored if item["accepted"]]
    rejected = [
        {**item, "reasons": item["reasons"]}
        for item in scored
        if not item["accepted"]
    ]

    if not accepted:
        notes.append("All candidates rejected; a missing frame beats a wrong point.")
        return {
            "selected": None,
            "rejected": rejected,
            "selection_quality": QUALITY_POOR,
            "notes": notes,
        }

    accepted.sort(key=lambda item: item["score"], reverse=True)
    selected = accepted[0]
    for runner_up in accepted[1:]:
        rejected.append({**runner_up, "reasons": runner_up["reasons"] + ["not_best_in_frame"]})

    if selected["score"] >= 0.6:
        quality = QUALITY_GOOD
    else:
        quality = QUALITY_PARTIAL
        notes.append("Best candidate accepted with only moderate reliability.")
    notes.append(f"Selected 1 of {len(parsed)} candidates (score={selected['score']}).")
    return {
        "selected": selected,
        "rejected": rejected,
        "selection_quality": quality,
        "notes": notes,
    }


def _group_candidates_by_frame(frame_candidates: Any) -> dict[int, list[dict[str, Any]]]:
    """Group timeline-style or flat candidate inputs into frame_index -> candidates."""
    grouped: dict[int, list[dict[str, Any]]] = {}

    def _add(candidate: dict[str, Any] | None, fallback_frame: int) -> None:
        if candidate is None:
            return
        frame = candidate.get("frame_index")
        if frame is None:
            frame = fallback_frame
            candidate["frame_index"] = frame
        grouped.setdefault(int(frame), []).append(candidate)

    if isinstance(frame_candidates, dict):
        items = list(frame_candidates.items())
    elif isinstance(frame_candidates, (list, tuple)):
        items = list(enumerate(frame_candidates))
    else:
        return grouped

    for fallback_index, frame_item in items:
        fallback = _safe_int(fallback_index, 0) or 0
        if isinstance(frame_item, dict) and (
            "ball_detections" in frame_item or "balls" in frame_item
        ):
            # App timeline format: {"frame_index": n, "ball_detections": [...]}.
            frame = _safe_int(frame_item.get("frame_index"), fallback)
            detections = frame_item.get("ball_detections") or frame_item.get("balls") or []
            if isinstance(detections, (list, tuple)):
                for detection in detections:
                    _add(_parse_candidate(detection, frame_index=frame), frame)
        elif isinstance(frame_item, (list, tuple)) and frame_item and all(
            isinstance(det, dict) for det in frame_item
        ):
            # Dict-of-lists / list-of-lists: one candidate list per frame.
            for detection in frame_item:
                _add(_parse_candidate(detection, frame_index=fallback), fallback)
        else:
            _add(_parse_candidate(frame_item, frame_index=fallback), fallback)
    return grouped


def build_reliable_ball_track(
    frame_candidates: Any,
    frame_size: Any = None,
    pitch_roi: Any = None,
) -> dict[str, Any]:
    """Build a clean selected ball track from frame-wise raw candidates."""
    try:
        return _build_reliable_ball_track_impl(
            frame_candidates, frame_size=frame_size, pitch_roi=pitch_roi
        )
    except Exception as exc:  # never crash UI consumers
        return {
            "track_points": [],
            "selected_candidates": [],
            "rejected_candidates": [],
            "missing_frames": [],
            "track_quality": QUALITY_UNAVAILABLE,
            "total_frames": 0,
            "frames_with_candidates": 0,
            "notes": [f"Reliable track build failed safely: {exc}"],
        }


def _build_reliable_ball_track_impl(
    frame_candidates: Any,
    *,
    frame_size: Any = None,
    pitch_roi: Any = None,
) -> dict[str, Any]:
    grouped = _group_candidates_by_frame(frame_candidates)
    notes: list[str] = []
    roi = normalize_pitch_roi(pitch_roi, frame_size=frame_size)
    if roi["available"]:
        notes.append("Pitch ROI active: obvious off-pitch candidates are rejected.")

    if not grouped:
        return {
            "track_points": [],
            "selected_candidates": [],
            "rejected_candidates": [],
            "missing_frames": [],
            "track_quality": QUALITY_UNAVAILABLE,
            "total_frames": 0,
            "frames_with_candidates": 0,
            "notes": notes + ["No usable ball candidates found."],
        }

    selected_candidates: list[dict[str, Any]] = []
    rejected_candidates: list[dict[str, Any]] = []
    previous_points: list[dict[str, Any]] = []

    for frame in sorted(grouped):
        if previous_points:
            last_frame = previous_points[-1].get("frame_index")
            if last_frame is not None and frame - last_frame > MAX_GAP_FRAMES:
                # ponytail: after a long gap, motion history is stale; restart the
                # track instead of force-connecting across the gap. No points invented.
                previous_points = []
                notes.append(
                    f"Gap of {frame - last_frame} frames before frame {frame}; motion history restarted."
                )
        selection = select_best_candidate_for_frame(
            grouped[frame],
            previous_points=previous_points,
            frame_size=frame_size,
            pitch_roi=roi if roi["available"] else None,
        )
        rejected_candidates.extend(selection["rejected"])
        if selection["selected"] is not None:
            selected_candidates.append(selection["selected"])
            previous_points.append(selection["selected"])

    frames_seen = sorted(grouped)
    total_frames = frames_seen[-1] - frames_seen[0] + 1 if frames_seen else 0
    selected_frames = {item["frame_index"] for item in selected_candidates}
    missing_frames = (
        [f for f in range(frames_seen[0], frames_seen[-1] + 1) if f not in selected_frames]
        if selected_frames
        else list(frames_seen)
    )

    track_points = [
        {
            "frame_index": item["frame_index"],
            "x": round(item["x"], 3),
            "y": round(item["y"], 3),
            "confidence": item.get("confidence"),
            "source": "reliable_track",
        }
        for item in selected_candidates
    ]

    reject_ratio = len(rejected_candidates) / max(1, len(selected_candidates) + len(rejected_candidates))
    if not track_points:
        quality = QUALITY_UNAVAILABLE
        notes.append("No candidate survived reliability checks.")
    elif len(track_points) < MIN_TRACK_POINTS or reject_ratio > POOR_REJECT_RATIO:
        quality = QUALITY_POOR
        notes.append("Too few reliable points or too many rejections; track not trustworthy.")
    elif len(track_points) >= GOOD_MIN_POINTS and reject_ratio <= GOOD_MAX_REJECT_RATIO:
        quality = QUALITY_GOOD
        notes.append("Coherent reliable ball track selected from raw candidates.")
    else:
        quality = QUALITY_PARTIAL
        notes.append("Reliable track is usable but short or noisy.")

    notes.append(
        f"Selected {len(track_points)} points, rejected {len(rejected_candidates)} candidates, "
        f"{len(missing_frames)} frames without a trusted point (gaps kept, never interpolated)."
    )
    return {
        "track_points": track_points,
        "selected_candidates": selected_candidates,
        "rejected_candidates": rejected_candidates,
        "missing_frames": missing_frames,
        "track_quality": quality,
        "total_frames": total_frames,
        "frames_with_candidates": len(frames_seen),
        "notes": notes,
    }


def build_ball_candidate_debug_report(track_result: Any) -> dict[str, Any]:
    """UI-friendly summary of a build_reliable_ball_track result."""
    track = track_result if isinstance(track_result, dict) else {}
    return {
        "track_quality": track.get("track_quality") or QUALITY_UNAVAILABLE,
        "total_frames": _safe_int(track.get("total_frames"), 0) or 0,
        "frames_with_candidates": _safe_int(track.get("frames_with_candidates"), 0) or 0,
        "selected_points": len(track.get("track_points") or []),
        "rejected_candidates": len(track.get("rejected_candidates") or []),
        "missing_frames": list(track.get("missing_frames") or []),
        "notes": list(track.get("notes") or []),
    }
