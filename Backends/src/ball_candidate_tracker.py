"""Ball Candidate Reliability Tracker v1 — app-side detection cleanup.

Pure Python helpers. No Streamlit, YOLO, OpenCV, or model loading.
Selects the most believable *moving delivery* ball point per frame from
raw detections. Never invents points, never interpolates gaps as observed —
a missing frame is always better than a wrong/static ball.
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

# Motion / static separation — prefer missing over locking onto a ground ball.
# ponytail: pixel thresholds mirror tracking/trajectory_scorer static cells without importing it.
STATIC_CELL_PX = 10.0
STATIC_MATCH_PX = 12.0
STATIC_MIN_HITS = 3
NEAR_ZERO_MOVE_PX = 8.0
MIN_DELIVERY_MOVE_PX = 4.0
STATIC_STREAK_FRAMES = 3
BOOTSTRAP_MIN_FRAMES = 3
BOOTSTRAP_MIN_MOVEMENT_PX = 15.0
BOOTSTRAP_MAX_STEP_PX = 80.0  # per-frame link limit while bootstrapping
DELIVERY_DY_BIAS = 0.08  # small bonus when motion is mostly down-pitch (image +y)


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


def _static_grid_key(x: float, y: float) -> tuple[int, int]:
    return (int(round(x / STATIC_CELL_PX)), int(round(y / STATIC_CELL_PX)))


def _normalize_static_locations(static_locations: Any) -> list[dict[str, Any]]:
    """Accept [{x,y}, ...] or grid keys / hit maps and return point dicts."""
    locations: list[dict[str, Any]] = []
    if not static_locations:
        return locations
    if isinstance(static_locations, dict):
        # hit-map style: {(gx,gy): hits} or {"x":..,"y":..} single point
        if "x" in static_locations and "y" in static_locations:
            parsed = _parse_candidate(static_locations)
            if parsed is not None:
                locations.append({"x": parsed["x"], "y": parsed["y"]})
            return locations
        for key, hits in static_locations.items():
            hit_count = _safe_int(hits, 0) or 0
            if hit_count < 1:
                continue
            if isinstance(key, (list, tuple)) and len(key) >= 2:
                gx, gy = _safe_float(key[0]), _safe_float(key[1])
                if gx is None or gy is None:
                    continue
                locations.append(
                    {
                        "x": gx * STATIC_CELL_PX,
                        "y": gy * STATIC_CELL_PX,
                        "hits": hit_count,
                    }
                )
        return locations
    for item in static_locations:
        if isinstance(item, (list, tuple)) and len(item) >= 2 and not isinstance(item[0], dict):
            # bare (x,y) or grid key — treat as pixel if values look large, else grid
            x = _safe_float(item[0])
            y = _safe_float(item[1])
            if x is None or y is None:
                continue
            if abs(x) <= 200 and abs(y) <= 200 and (abs(x) < 50 or abs(y) < 50):
                # likely grid cell indices
                locations.append({"x": x * STATIC_CELL_PX, "y": y * STATIC_CELL_PX})
            else:
                locations.append({"x": float(x), "y": float(y)})
            continue
        parsed = _parse_candidate(item)
        if parsed is not None:
            locations.append({"x": parsed["x"], "y": parsed["y"]})
    return locations


def _matches_static_location(
    x: float, y: float, static_locations: Any, radius: float = STATIC_MATCH_PX
) -> bool:
    for loc in _normalize_static_locations(static_locations):
        if hypot(x - loc["x"], y - loc["y"]) <= radius:
            return True
    return False


def _recent_velocity(prev: list[dict[str, Any]]) -> tuple[float, float] | None:
    if len(prev) < 2:
        return None
    last = prev[-1]
    second_last = prev[-2]
    step = 1
    if last.get("frame_index") is not None and second_last.get("frame_index") is not None:
        step = max(1, last["frame_index"] - second_last["frame_index"])
    return (last["x"] - second_last["x"]) / step, (last["y"] - second_last["y"]) / step


def _recent_path_is_static(prev: list[dict[str, Any]]) -> bool:
    """True when the last few accepted points barely moved (background ball)."""
    if len(prev) < STATIC_STREAK_FRAMES:
        return False
    window = prev[-STATIC_STREAK_FRAMES:]
    total = 0.0
    for idx in range(1, len(window)):
        total += hypot(window[idx]["x"] - window[idx - 1]["x"], window[idx]["y"] - window[idx - 1]["y"])
    return total < NEAR_ZERO_MOVE_PX * (STATIC_STREAK_FRAMES - 1)


def score_ball_candidate(
    candidate: Any,
    previous_points: Any = None,
    frame_size: Any = None,
    pitch_roi: Any = None,
    static_locations: Any = None,
) -> dict[str, Any]:
    """Score how believable one candidate is as the *moving delivery* ball.

    High confidence never rescues a static ground ball or impossible movement.
    """
    reasons: list[str] = []
    parsed = _parse_candidate(candidate)
    if parsed is None:
        return {"score": 0.0, "accepted": False, "reasons": ["invalid_candidate"]}

    width, height, diag = _frame_bounds(frame_size)
    prev = _motion_context(previous_points)
    score = 0.2
    hard_reject = False
    static_reject = False

    x, y = parsed["x"], parsed["y"]
    if x < 0 or y < 0 or x > width or y > height:
        hard_reject = True
        reasons.append("outside_frame")

    confidence = parsed.get("confidence")
    if confidence is not None:
        # Confidence is a weak signal only — motion decides the delivery ball.
        score += 0.15 * min(max(confidence, 0.0), 1.0)
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

    near_static = _matches_static_location(x, y, static_locations)
    if near_static:
        reasons.append("matches_known_static_location")

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
        min_move = MIN_DELIVERY_MOVE_PX * min(gap, 2)

        # Near-zero movement from the active moving track → static / background ball.
        if dist < NEAR_ZERO_MOVE_PX * min(gap, 2):
            static_reject = True
            hard_reject = True
            if confidence is not None and confidence >= 0.5:
                reasons.append("high_confidence_no_delivery_motion")
            else:
                reasons.append("likely_static_ball")
            reasons.append("near_zero_movement")
        elif dist < min_move:
            score -= 0.15
            reasons.append("weak_delivery_motion")
        else:
            # Prefer consistent frame-to-frame movement without huge jumps.
            if dist > jump_limit:
                hard_reject = True
                reasons.append("huge_jump")
            elif abs(dx) > abs(dy) * SIDEWAYS_RATIO and abs(dx) > SIDEWAYS_MIN_DX * gap:
                hard_reject = True
                reasons.append("impossible_sideways")
            else:
                # Mid-range movement scores best; huge-but-legal jumps score lower.
                motion_quality = 1.0 - abs(dist - jump_limit * 0.35) / jump_limit
                score += 0.25 * max(0.0, min(1.0, motion_quality))
                reasons.append("consistent_delivery_motion")
                if dy > abs(dx) * 0.4:
                    score += DELIVERY_DY_BIAS
                    reasons.append("delivery_direction")

        if not hard_reject and len(prev) >= 2:
            overall_dy = prev[-1]["y"] - prev[0]["y"]
            if abs(overall_dy) > 2.0:
                sign = 1.0 if overall_dy > 0 else -1.0
                if dy * sign < 0 and abs(dy) > max(REVERSAL_MIN_DY * gap, abs(dx)):
                    hard_reject = True
                    reasons.append("impossible_reversal")

        velocity = _recent_velocity(prev)
        if not hard_reject and velocity is not None:
            vx, vy = velocity
            pred_x = last["x"] + vx * gap
            pred_y = last["y"] + vy * gap
            pred_dist = hypot(x - pred_x, y - pred_y)
            speed = hypot(vx, vy)
            if pred_dist <= jump_limit:
                score += 0.25 * (1.0 - pred_dist / jump_limit)
                reasons.append("close_to_predicted")
            else:
                # Far from the moving path — reject even if confidence is high.
                if pred_dist > jump_limit * 0.85 and speed >= MIN_DELIVERY_MOVE_PX:
                    hard_reject = True
                    reasons.append("far_from_moving_path")
                else:
                    score -= 0.2
                    reasons.append("far_from_predicted")

            # Velocity direction consistency (dot product of step vs recent velocity).
            if speed >= MIN_DELIVERY_MOVE_PX and dist >= MIN_DELIVERY_MOVE_PX:
                step_speed = dist / gap
                if step_speed > 1e-6:
                    cos_sim = (dx / gap * vx + dy / gap * vy) / (step_speed * speed)
                    if cos_sim > 0.3:
                        score += 0.12 * min(1.0, cos_sim)
                        reasons.append("velocity_direction_consistent")
                    elif cos_sim < -0.2:
                        score -= 0.15
                        reasons.append("velocity_direction_inconsistent")

        # Known static hotspot that is not on the predicted moving path.
        if near_static and not hard_reject:
            if velocity is not None:
                vx, vy = velocity
                pred_x = last["x"] + vx * gap
                pred_y = last["y"] + vy * gap
                if hypot(x - pred_x, y - pred_y) > STATIC_MATCH_PX * 2:
                    static_reject = True
                    hard_reject = True
                    reasons.append("static_location_off_path")
            else:
                score -= 0.25
                reasons.append("static_location_penalty")
    else:
        # No moving-ball history yet: refuse known static hotspots and never
        # accept on confidence alone (bootstrap confirms motion separately).
        if near_static:
            static_reject = True
            hard_reject = True
            if confidence is not None and confidence >= 0.5:
                reasons.append("high_confidence_no_delivery_motion")
            else:
                reasons.append("likely_static_ball")
        else:
            reasons.append("awaiting_motion_confirmation")
            # Soft score for ranking only — never accept without motion history.
            score = min(score, ACCEPT_SCORE - 0.01)

    score = max(0.0, min(1.0, score))
    if hard_reject:
        if static_reject and "likely_static_ball" not in reasons and "high_confidence_no_delivery_motion" not in reasons:
            reasons.append("likely_static_ball")
        return {
            "score": round(min(score, ACCEPT_SCORE), 3),
            "accepted": False,
            "reasons": reasons,
            "static_reject": static_reject,
        }
    if score < ACCEPT_SCORE:
        reasons.append("low_reliability_score")
        return {
            "score": round(score, 3),
            "accepted": False,
            "reasons": reasons,
            "static_reject": static_reject,
        }
    return {
        "score": round(score, 3),
        "accepted": True,
        "reasons": reasons,
        "static_reject": False,
    }


def _human_reject_note(reasons: list[str], confidence: float | None = None) -> str:
    if "high_confidence_no_delivery_motion" in reasons or (
        confidence is not None and confidence >= 0.5 and (
            "likely_static_ball" in reasons
            or "near_zero_movement" in reasons
            or "matches_known_static_location" in reasons
        )
    ):
        return "Rejected candidate: high confidence but no delivery motion"
    if (
        "likely_static_ball" in reasons
        or "near_zero_movement" in reasons
        or "matches_known_static_location" in reasons
        or "static_location_off_path" in reasons
    ):
        return "Rejected candidate: likely static ball"
    return ""


def select_best_candidate_for_frame(
    candidates: Any,
    previous_points: Any = None,
    frame_size: Any = None,
    pitch_roi: Any = None,
    static_locations: Any = None,
) -> dict[str, Any]:
    """Pick the most believable *moving* ball candidate for one frame, or nothing."""
    notes: list[str] = []
    parsed = normalize_detection_candidates(candidates)
    if not parsed:
        return {
            "selected": None,
            "rejected": [],
            "selection_quality": QUALITY_UNAVAILABLE,
            "notes": ["No usable candidates in this frame."],
            "rejected_static_count": 0,
        }

    scored = []
    for candidate in parsed:
        verdict = score_ball_candidate(
            candidate,
            previous_points=previous_points,
            frame_size=frame_size,
            pitch_roi=pitch_roi,
            static_locations=static_locations,
        )
        scored.append(
            {
                **candidate,
                "score": verdict["score"],
                "reasons": verdict["reasons"],
                "accepted": verdict["accepted"],
                "static_reject": bool(verdict.get("static_reject")),
            }
        )

    accepted = [item for item in scored if item["accepted"]]
    rejected = [
        {**item, "reasons": item["reasons"]}
        for item in scored
        if not item["accepted"]
    ]
    rejected_static_count = sum(1 for item in rejected if item.get("static_reject"))

    for item in rejected:
        note = _human_reject_note(item.get("reasons") or [], item.get("confidence"))
        if note and note not in notes:
            notes.append(note)

    if not accepted:
        notes.append("No moving delivery ball found in this frame")
        notes.append("All candidates rejected; a missing frame beats a wrong point.")
        return {
            "selected": None,
            "rejected": rejected,
            "selection_quality": QUALITY_POOR,
            "notes": notes,
            "rejected_static_count": rejected_static_count,
        }

    # Prefer motion-backed scores; never fall back to a static high-confidence ball.
    accepted.sort(key=lambda item: item["score"], reverse=True)
    selected = accepted[0]
    for runner_up in accepted[1:]:
        rejected.append({**runner_up, "reasons": runner_up["reasons"] + ["not_best_in_frame"]})

    if selected["score"] >= 0.6:
        quality = QUALITY_GOOD
    else:
        quality = QUALITY_PARTIAL
        notes.append("Best candidate accepted with only moderate reliability.")
    notes.append("Selected candidate: consistent moving delivery ball")
    notes.append(f"Selected 1 of {len(parsed)} candidates (score={selected['score']}).")
    return {
        "selected": selected,
        "rejected": rejected,
        "selection_quality": quality,
        "notes": notes,
        "rejected_static_count": rejected_static_count,
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


def _record_static_hit(
    static_hits: dict[tuple[int, int], int],
    static_locations: list[dict[str, Any]],
    x: float,
    y: float,
) -> None:
    key = _static_grid_key(x, y)
    static_hits[key] = static_hits.get(key, 0) + 1
    if static_hits[key] >= STATIC_MIN_HITS:
        cx = key[0] * STATIC_CELL_PX
        cy = key[1] * STATIC_CELL_PX
        if not _matches_static_location(cx, cy, static_locations, radius=STATIC_CELL_PX):
            static_locations.append({"x": cx, "y": cy, "hits": static_hits[key]})


def _path_travel(points: list[dict[str, Any]]) -> float:
    total = 0.0
    for idx in range(1, len(points)):
        total += hypot(points[idx]["x"] - points[idx - 1]["x"], points[idx]["y"] - points[idx - 1]["y"])
    return total


def _is_static_reject_item(item: dict[str, Any]) -> bool:
    if item.get("static_reject"):
        return True
    reasons = item.get("reasons") or []
    return any(
        tag in reasons
        for tag in (
            "likely_static_ball",
            "near_zero_movement",
            "high_confidence_no_delivery_motion",
            "matches_known_static_location",
            "static_location_off_path",
        )
    )


def _append_unique_note(notes: list[str], seen: set[str], note: str) -> None:
    if note and note not in seen:
        notes.append(note)
        seen.add(note)


def _pick_bootstrap_link(
    frame_dets: list[dict[str, Any]],
    provisional: list[dict[str, Any]],
    *,
    frame_size: Any,
    pitch_roi: Any,
    static_locations: list[dict[str, Any]],
    seed_frame_dets: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]] | None:
    """Continue a provisional chain, or start one from a moving pair across frames.

    Returns one linked candidate, a (prev, curr) seed pair, or None.
    Never seeds from confidence alone — motion between frames is required.
    """
    if provisional:
        last = provisional[-1]
        best: dict[str, Any] | None = None
        best_score = -1.0
        for det in frame_dets:
            parsed = _parse_candidate(det, frame_index=det.get("frame_index"))
            if parsed is None:
                continue
            if _matches_static_location(parsed["x"], parsed["y"], static_locations):
                continue
            dist = hypot(parsed["x"] - last["x"], parsed["y"] - last["y"])
            if dist < MIN_DELIVERY_MOVE_PX or dist > BOOTSTRAP_MAX_STEP_PX:
                continue
            verdict = score_ball_candidate(
                parsed,
                previous_points=provisional,
                frame_size=frame_size,
                pitch_roi=pitch_roi,
                static_locations=static_locations,
            )
            if verdict["accepted"] or (
                not verdict.get("static_reject")
                and "huge_jump" not in verdict["reasons"]
                and "impossible_sideways" not in verdict["reasons"]
                and verdict["score"] >= ACCEPT_SCORE * 0.85
            ):
                if verdict["score"] > best_score:
                    best_score = verdict["score"]
                    best = {
                        **parsed,
                        "score": verdict["score"],
                        "reasons": verdict["reasons"] + ["bootstrap_motion_link"],
                        "accepted": True,
                        "static_reject": False,
                    }
        return best

    # No provisional yet: require a moving pair from the previous frame's detections.
    if not seed_frame_dets:
        return None
    best_pair: tuple[dict[str, Any], dict[str, Any]] | None = None
    best_score = -1.0
    for prev_raw in seed_frame_dets:
        prev = _parse_candidate(prev_raw, frame_index=prev_raw.get("frame_index"))
        if prev is None:
            continue
        if _matches_static_location(prev["x"], prev["y"], static_locations):
            continue
        for curr_raw in frame_dets:
            curr = _parse_candidate(curr_raw, frame_index=curr_raw.get("frame_index"))
            if curr is None:
                continue
            if _matches_static_location(curr["x"], curr["y"], static_locations):
                continue
            dist = hypot(curr["x"] - prev["x"], curr["y"] - prev["y"])
            if dist < MIN_DELIVERY_MOVE_PX or dist > BOOTSTRAP_MAX_STEP_PX:
                continue
            verdict = score_ball_candidate(
                curr,
                previous_points=[prev],
                frame_size=frame_size,
                pitch_roi=pitch_roi,
                static_locations=static_locations,
            )
            if verdict.get("static_reject"):
                continue
            if "huge_jump" in verdict["reasons"] or "impossible_sideways" in verdict["reasons"]:
                continue
            pair_score = verdict["score"] + min(0.2, dist / 100.0)
            if pair_score > best_score:
                best_score = pair_score
                best_pair = (
                    {
                        **prev,
                        "score": round(pair_score, 3),
                        "reasons": ["bootstrap_motion_seed"],
                        "accepted": True,
                        "static_reject": False,
                    },
                    {
                        **curr,
                        "score": verdict["score"],
                        "reasons": verdict["reasons"] + ["bootstrap_motion_seed"],
                        "accepted": True,
                        "static_reject": False,
                    },
                )
    return best_pair


def build_reliable_ball_track(
    frame_candidates: Any,
    frame_size: Any = None,
    pitch_roi: Any = None,
) -> dict[str, Any]:
    """Build a clean selected *moving delivery* ball track from frame-wise raw candidates."""
    try:
        return _build_reliable_ball_track_impl(
            frame_candidates, frame_size=frame_size, pitch_roi=pitch_roi
        )
    except Exception as exc:  # never crash UI consumers
        return {
            "track_points": [],
            "selected_candidates": [],
            "rejected_candidates": [],
            "rejected_static_candidates": [],
            "static_ball_locations": [],
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
    active_roi = roi if roi["available"] else None
    if roi["available"]:
        notes.append("Pitch ROI active: obvious off-pitch candidates are rejected.")

    empty = {
        "track_points": [],
        "selected_candidates": [],
        "rejected_candidates": [],
        "rejected_static_candidates": [],
        "static_ball_locations": [],
        "missing_frames": [],
        "track_quality": QUALITY_UNAVAILABLE,
        "total_frames": 0,
        "frames_with_candidates": 0,
        "notes": notes + ["No usable ball candidates found."],
    }
    if not grouped:
        return empty

    selected_candidates: list[dict[str, Any]] = []
    rejected_candidates: list[dict[str, Any]] = []
    rejected_static_candidates: list[dict[str, Any]] = []
    previous_points: list[dict[str, Any]] = []  # confirmed moving delivery points
    provisional: list[dict[str, Any]] = []  # bootstrap chain awaiting motion proof
    bootstrap_prev_frame: list[dict[str, Any]] = []
    track_confirmed = False
    static_hits: dict[tuple[int, int], int] = {}
    static_locations: list[dict[str, Any]] = []
    frame_notes_seen: set[str] = set()

    def _mark_static_from_item(item: dict[str, Any]) -> None:
        rejected_static_candidates.append(item)
        _record_static_hit(static_hits, static_locations, item["x"], item["y"])

    def _reject_provisional_as_static(reason: str) -> None:
        nonlocal provisional
        for pt in provisional:
            tagged = {
                **pt,
                "accepted": False,
                "reasons": (pt.get("reasons") or []) + ["likely_static_ball", reason],
                "static_reject": True,
            }
            rejected_candidates.append(tagged)
            _mark_static_from_item(tagged)
        provisional = []
        _append_unique_note(notes, frame_notes_seen, "Rejected candidate: likely static ball")

    for frame in sorted(grouped):
        frame_dets = grouped[frame]

        if track_confirmed and previous_points:
            last_frame = previous_points[-1].get("frame_index")
            if last_frame is not None and frame - last_frame > MAX_GAP_FRAMES:
                # Keep static_locations; do not jump onto a ground ball after a gap.
                previous_points = []
                track_confirmed = False
                provisional = []
                bootstrap_prev_frame = []
                notes.append(
                    f"Gap of {frame - last_frame} frames before frame {frame}; motion history restarted."
                )
            elif _recent_path_is_static(previous_points):
                for pt in previous_points[-STATIC_STREAK_FRAMES:]:
                    _record_static_hit(static_hits, static_locations, pt["x"], pt["y"])
                notes.append(
                    "Recent selected path barely moved; treating as static and clearing moving track."
                )
                previous_points = []
                track_confirmed = False
                provisional = []
                bootstrap_prev_frame = []

        if not track_confirmed:
            # --- Bootstrap: require multi-frame motion before committing ---
            link = _pick_bootstrap_link(
                frame_dets,
                provisional,
                frame_size=frame_size,
                pitch_roi=active_roi,
                static_locations=static_locations,
                seed_frame_dets=bootstrap_prev_frame,
            )
            # Score all for rejection bookkeeping.
            selection = select_best_candidate_for_frame(
                frame_dets,
                previous_points=provisional or None,
                frame_size=frame_size,
                pitch_roi=active_roi,
                static_locations=static_locations,
            )
            for item in selection["rejected"]:
                rejected_candidates.append(item)
                if _is_static_reject_item(item):
                    _mark_static_from_item(item)
            for note in selection.get("notes") or []:
                if note.startswith("Rejected candidate:"):
                    _append_unique_note(notes, frame_notes_seen, note)

            if isinstance(link, tuple):
                # Fresh moving pair — start provisional from motion evidence only.
                provisional = [link[0], link[1]]
                bootstrap_prev_frame = frame_dets
                if _path_travel(provisional) >= BOOTSTRAP_MIN_MOVEMENT_PX and len(provisional) >= BOOTSTRAP_MIN_FRAMES:
                    selected_candidates.extend(provisional)
                    previous_points = list(provisional)
                    provisional = []
                    track_confirmed = True
                    _append_unique_note(
                        notes,
                        frame_notes_seen,
                        "Selected candidate: consistent moving delivery ball",
                    )
                continue

            if link is None:
                if provisional:
                    last_f = provisional[-1].get("frame_index")
                    if last_f is not None and frame - last_f > 2:
                        _reject_provisional_as_static("bootstrap_gap")
                else:
                    # Accumulate static hits while waiting for a moving pair.
                    for det in frame_dets:
                        parsed = _parse_candidate(det, frame_index=frame)
                        if parsed is not None:
                            _record_static_hit(
                                static_hits, static_locations, parsed["x"], parsed["y"]
                            )
                bootstrap_prev_frame = frame_dets
                _append_unique_note(
                    notes, frame_notes_seen, "No moving delivery ball found in this frame"
                )
                continue

            # Continue provisional chain with a moving link.
            dist = hypot(link["x"] - provisional[-1]["x"], link["y"] - provisional[-1]["y"])
            if dist < MIN_DELIVERY_MOVE_PX:
                _mark_static_from_item(
                    {
                        **link,
                        "accepted": False,
                        "static_reject": True,
                        "reasons": ["likely_static_ball", "near_zero_movement"],
                    }
                )
                rejected_candidates.append(
                    {
                        **link,
                        "accepted": False,
                        "static_reject": True,
                        "reasons": ["likely_static_ball", "near_zero_movement"],
                    }
                )
                _append_unique_note(
                    notes, frame_notes_seen, "Rejected candidate: likely static ball"
                )
                bootstrap_prev_frame = frame_dets
                continue

            provisional.append(link)
            bootstrap_prev_frame = frame_dets
            if len(provisional) >= BOOTSTRAP_MIN_FRAMES:
                travel = _path_travel(provisional)
                if travel >= BOOTSTRAP_MIN_MOVEMENT_PX:
                    selected_candidates.extend(provisional)
                    previous_points = list(provisional)
                    provisional = []
                    track_confirmed = True
                    _append_unique_note(
                        notes,
                        frame_notes_seen,
                        "Selected candidate: consistent moving delivery ball",
                    )
                else:
                    _reject_provisional_as_static("bootstrap_insufficient_motion")
            continue

        # --- Confirmed moving track ---
        selection = select_best_candidate_for_frame(
            frame_dets,
            previous_points=previous_points,
            frame_size=frame_size,
            pitch_roi=active_roi,
            static_locations=static_locations,
        )
        rejected_candidates.extend(selection["rejected"])
        for item in selection["rejected"]:
            if _is_static_reject_item(item):
                _mark_static_from_item(item)
        for note in selection.get("notes") or []:
            if (
                note.startswith("Rejected candidate:")
                or note.startswith("Selected candidate:")
                or note == "No moving delivery ball found in this frame"
            ):
                _append_unique_note(notes, frame_notes_seen, note)

        if selection["selected"] is not None:
            selected = selection["selected"]
            last = previous_points[-1]
            dist = hypot(selected["x"] - last["x"], selected["y"] - last["y"])
            if dist < NEAR_ZERO_MOVE_PX:
                tagged = {
                    **selected,
                    "accepted": False,
                    "static_reject": True,
                    "reasons": (selected.get("reasons") or [])
                    + ["likely_static_ball", "near_zero_movement"],
                }
                rejected_candidates.append(tagged)
                _mark_static_from_item(tagged)
                _append_unique_note(
                    notes, frame_notes_seen, "No moving delivery ball found in this frame"
                )
                continue
            selected_candidates.append(selected)
            previous_points.append(selected)
            _append_unique_note(
                notes, frame_notes_seen, "Selected candidate: consistent moving delivery ball"
            )
        else:
            # Short gap allowed; do not switch to a static high-confidence ball.
            for det in frame_dets:
                parsed = _parse_candidate(det, frame_index=frame)
                if parsed is not None:
                    _record_static_hit(static_hits, static_locations, parsed["x"], parsed["y"])

    # Leftover provisional chain never confirmed — do not emit as observed track.
    if provisional:
        travel = _path_travel(provisional)
        if len(provisional) >= BOOTSTRAP_MIN_FRAMES and travel >= BOOTSTRAP_MIN_MOVEMENT_PX:
            selected_candidates.extend(provisional)
            _append_unique_note(
                notes, frame_notes_seen, "Selected candidate: consistent moving delivery ball"
            )
        else:
            _reject_provisional_as_static("bootstrap_unconfirmed")

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

    reject_ratio = len(rejected_candidates) / max(
        1, len(selected_candidates) + len(rejected_candidates)
    )
    if not track_points:
        quality = QUALITY_UNAVAILABLE
        notes.append("No reliable moving ball detected")
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
        f"Selected {len(track_points)} points, rejected {len(rejected_candidates)} candidates "
        f"({len(rejected_static_candidates)} static), "
        f"{len(missing_frames)} frames without a trusted point (gaps kept, never interpolated)."
    )
    return {
        "track_points": track_points,
        "selected_candidates": selected_candidates,
        "rejected_candidates": rejected_candidates,
        "rejected_static_candidates": rejected_static_candidates,
        "static_ball_locations": [
            {"x": round(loc["x"], 3), "y": round(loc["y"], 3), "hits": loc.get("hits")}
            for loc in static_locations
        ],
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
        "rejected_static_candidates": len(track.get("rejected_static_candidates") or []),
        "static_ball_locations": len(track.get("static_ball_locations") or []),
        "missing_frames": list(track.get("missing_frames") or []),
        "notes": list(track.get("notes") or []),
    }
