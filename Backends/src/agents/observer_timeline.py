"""Observer timeline summary for uploaded video detection quality."""

from __future__ import annotations

DEBUG_OBSERVER_TIMELINE = False

MAX_REASONABLE_BALL_JUMP_PX = 180
LOW_CONFIDENCE_THRESHOLD = 0.35


def build_observer_timeline(frame_detections, total_frames=None, fps=None):
    """Build a detection timeline summary for uploaded video analysis."""
    frames = _normalize_frame_detections(frame_detections)
    timeline_total = int(total_frames) if total_frames else len(frames)
    processed_frames = len(frames)

    ball_frames = sum(1 for item in frames if item["ball_detections"])
    bat_frames = sum(1 for item in frames if item["bat_detections"])
    stump_frames = sum(1 for item in frames if item["stump_detections"])

    ball_coverage = _coverage(ball_frames, processed_frames)
    bat_coverage = _coverage(bat_frames, processed_frames)
    stump_coverage = _coverage(stump_frames, processed_frames)

    missing_ball_frames, false_jumps, low_confidence_ball_frames = _timeline_gaps_and_quality(frames)
    detection_quality = _detection_quality(
        ball_coverage,
        missing_ball_frames,
        false_jumps,
        low_confidence_ball_frames,
    )
    observer_notes = _observer_notes(
        detection_quality,
        ball_coverage,
        missing_ball_frames,
        low_confidence_ball_frames,
        false_jumps,
        processed_frames,
        timeline_total,
    )

    result = {
        "total_frames": timeline_total,
        "processed_frames": processed_frames,
        "fps": float(fps) if fps else None,
        "ball_detected_frames": ball_frames,
        "bat_detected_frames": bat_frames,
        "stump_detected_frames": stump_frames,
        "ball_tracking_coverage": ball_coverage,
        "bat_detection_coverage": bat_coverage,
        "stump_detection_coverage": stump_coverage,
        "missing_ball_frames": missing_ball_frames,
        "low_confidence_ball_frames": low_confidence_ball_frames,
        "possible_false_ball_detections": false_jumps,
        "detection_quality": detection_quality,
        "observer_notes": observer_notes,
        "timeline_ready_for_live_agent": detection_quality in {"High", "Medium"},
    }
    if DEBUG_OBSERVER_TIMELINE:
        result["debug"] = {"normalized_frames": len(frames)}
    return result


def _normalize_frame_detections(frame_detections):
    if not frame_detections:
        return []

    items = frame_detections.items() if isinstance(frame_detections, dict) else enumerate(frame_detections)
    normalized = []
    for fallback_index, raw_frame in items:
        raw_frame = raw_frame or {}
        if not isinstance(raw_frame, dict):
            continue
        frame_index = raw_frame.get("frame_index", fallback_index)
        try:
            frame_index = int(frame_index)
        except (TypeError, ValueError):
            frame_index = len(normalized)
        normalized.append(
            {
                "frame_index": frame_index,
                "ball_detections": list(raw_frame.get("ball_detections") or raw_frame.get("balls") or []),
                "bat_detections": list(raw_frame.get("bat_detections") or raw_frame.get("bats") or []),
                "stump_detections": list(
                    raw_frame.get("stump_detections")
                    or raw_frame.get("stumps")
                    or []
                ),
            }
        )
    return sorted(normalized, key=lambda item: item["frame_index"])


def _coverage(detected_frames, total_frames):
    if total_frames <= 0:
        return None
    return round((detected_frames / total_frames) * 100, 1)


def _ball_center(detection):
    if not isinstance(detection, dict):
        return None
    center = detection.get("center")
    if center is not None and len(center) >= 2:
        return float(center[0]), float(center[1])
    bbox = detection.get("bbox") or detection.get("box")
    try:
        if bbox is None or len(bbox) < 4:
            return None
        return (float(bbox[0]) + float(bbox[2])) / 2, (float(bbox[1]) + float(bbox[3])) / 2
    except (TypeError, ValueError):
        return None


def _detection_confidence(detection):
    try:
        return float(detection.get("confidence", 0))
    except (TypeError, ValueError, AttributeError):
        return 0.0


def _timeline_gaps_and_quality(frames):
    missing_ball_frames = 0
    false_jumps = 0
    low_confidence_ball_frames = 0
    previous_center = None
    previous_had_ball = False

    for frame_item in frames:
        ball_detections = frame_item["ball_detections"]
        has_ball = bool(ball_detections)
        if previous_had_ball and not has_ball:
            missing_ball_frames += 1

        if has_ball:
            best = max(ball_detections, key=_detection_confidence)
            if _detection_confidence(best) < LOW_CONFIDENCE_THRESHOLD:
                low_confidence_ball_frames += 1
            center = _ball_center(best)
            if center is not None and previous_center is not None:
                jump = ((center[0] - previous_center[0]) ** 2 + (center[1] - previous_center[1]) ** 2) ** 0.5
                if jump > MAX_REASONABLE_BALL_JUMP_PX:
                    false_jumps += 1
            if center is not None:
                previous_center = center
        previous_had_ball = has_ball

    return missing_ball_frames, false_jumps, low_confidence_ball_frames


def _detection_quality(ball_coverage, missing_ball_frames, false_jumps, low_confidence_frames):
    if ball_coverage is None:
        return "Unknown"
    if ball_coverage >= 65 and missing_ball_frames <= 2 and false_jumps <= 1:
        return "High"
    if ball_coverage >= 40 and missing_ball_frames <= 8:
        return "Medium"
    if ball_coverage > 0:
        return "Low"
    return "Unknown"


def _observer_notes(
    detection_quality,
    ball_coverage,
    missing_ball_frames,
    low_confidence_frames,
    false_jumps,
    processed_frames,
    total_frames,
):
    coverage_text = f"{ball_coverage:.1f}%" if ball_coverage is not None else "unknown"
    notes = (
        f"Observer reviewed {processed_frames} sampled frames out of {total_frames} total frames. "
        f"Ball tracking coverage is {coverage_text} with {missing_ball_frames} missing-ball gaps."
    )
    if low_confidence_frames:
        notes += f" {low_confidence_frames} low-confidence ball frames were detected."
    if false_jumps:
        notes += f" {false_jumps} possible false ball detections were flagged from unrealistic jumps."
    if detection_quality == "High":
        notes += " Detection quality is strong enough for live-agent style review."
    elif detection_quality == "Medium":
        notes += " Ball tracking is usable, but several frames were missed near impact."
    elif detection_quality == "Low":
        notes += " Detection quality is weak; review confidence should stay conservative."
    else:
        notes += " Detection quality could not be estimated reliably."
    return notes
