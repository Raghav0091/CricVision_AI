"""Software vision agent that reviews detection quality and analysis consistency."""

DEBUG_VISION_AGENT = False

MAX_REASONABLE_BALL_JUMP_PX = 180


def run_vision_agent(
    frame_detections,
    delivery_report=None,
    impact_result=None,
    shot_result=None,
    direction_result=None,
    outcome_result=None,
    fps=None,
):
    """Review the complete video analysis and return a quality/control report."""
    frames = _normalize_frame_detections(frame_detections)
    total_frames = len(frames)
    impact_result = impact_result or {}
    shot_result = shot_result or {}
    direction_result = direction_result or {}
    outcome_result = outcome_result or {}
    delivery_report = delivery_report or {}

    ball_frames = sum(1 for item in frames if item["ball_detections"])
    bat_frames = sum(1 for item in frames if item["bat_detections"])
    stump_frames = sum(1 for item in frames if item["stump_detections"])

    ball_coverage = _coverage(ball_frames, total_frames)
    bat_coverage = _coverage(bat_frames, total_frames)
    stump_coverage = _coverage(stump_frames, total_frames)

    missing_ball_frames, false_ball_jumps = _ball_gap_and_jump_stats(frames)
    review_flags = _consistency_flags(
        impact_result,
        shot_result,
        direction_result,
        outcome_result,
        ball_coverage,
        missing_ball_frames,
        false_ball_jumps,
    )
    analysis_consistency = _analysis_consistency(review_flags, ball_coverage)
    agent_quality = _agent_quality(ball_coverage, review_flags, total_frames)
    agent_confidence = _agent_confidence(agent_quality, ball_coverage, impact_result)
    agent_notes = _agent_notes(
        ball_coverage,
        bat_coverage,
        impact_result,
        direction_result,
        outcome_result,
        review_flags,
        missing_ball_frames,
        false_ball_jumps,
    )
    review_frames_recommended = (
        ball_coverage is not None
        and ball_coverage < 45
    ) or bool(review_flags)

    result = {
        "agent_quality": agent_quality,
        "agent_confidence": agent_confidence,
        "ball_tracking_coverage": ball_coverage,
        "bat_detection_coverage": bat_coverage,
        "stump_detection_coverage": stump_coverage,
        "missing_ball_frames": missing_ball_frames,
        "possible_false_ball_detections": false_ball_jumps,
        "analysis_consistency": analysis_consistency,
        "review_flags": review_flags,
        "agent_notes": agent_notes,
        "review_frames_recommended": review_frames_recommended,
        "review_reason": (
            "Low ball tracking coverage or consistency flags suggest manual frame review."
            if review_frames_recommended
            else "Tracking was stable enough for automated review."
        ),
    }
    if DEBUG_VISION_AGENT:
        result["debug"] = {
            "total_frames": total_frames,
            "ball_frames": ball_frames,
            "bat_frames": bat_frames,
            "stump_frames": stump_frames,
            "delivery_report_keys": list(delivery_report.keys()) if isinstance(delivery_report, dict) else [],
        }
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


def _ball_gap_and_jump_stats(frames):
    missing_ball_frames = 0
    false_ball_jumps = 0
    previous_center = None
    previous_had_ball = False

    for frame_item in frames:
        has_ball = bool(frame_item["ball_detections"])
        if previous_had_ball and not has_ball:
            missing_ball_frames += 1

        if has_ball:
            center = _ball_center(max(frame_item["ball_detections"], key=_detection_confidence))
            if center is not None and previous_center is not None:
                jump = ((center[0] - previous_center[0]) ** 2 + (center[1] - previous_center[1]) ** 2) ** 0.5
                if jump > MAX_REASONABLE_BALL_JUMP_PX:
                    false_ball_jumps += 1
            if center is not None:
                previous_center = center
        previous_had_ball = has_ball

    return missing_ball_frames, false_ball_jumps


def _detection_confidence(detection):
    try:
        return float(detection.get("confidence", 0))
    except (TypeError, ValueError, AttributeError):
        return 0


def _consistency_flags(
    impact_result,
    shot_result,
    direction_result,
    outcome_result,
    ball_coverage,
    missing_ball_frames,
    false_ball_jumps,
):
    flags = []
    impact_detected = bool(impact_result.get("impact_detected")) or impact_result.get("impact_frame") is not None
    shot_type = shot_result.get("shot_type", "Unknown")
    shot_confidence = shot_result.get("shot_confidence", "Unknown")
    predicted_outcome = outcome_result.get("predicted_outcome", "Unknown")
    outcome_confidence = outcome_result.get("outcome_confidence", "Unknown")
    field_zone = direction_result.get("field_zone", "Unknown")
    shot_height = shot_result.get("shot_height", "Unknown")

    if not impact_detected and shot_type not in {"Unknown", "Defence"} and shot_confidence in {"High", "Medium"}:
        flags.append("Confident shot type without detected impact")
    if not impact_detected and predicted_outcome in {"Four", "Six"}:
        flags.append("Boundary outcome predicted without detected impact")
    if shot_height == "Aerial" and predicted_outcome == "Dot Ball":
        flags.append("Aerial shot classified but outcome predicted as dot ball")
    if ball_coverage is not None and ball_coverage < 35 and outcome_confidence == "High":
        flags.append("High outcome confidence despite low ball tracking coverage")
    if field_zone == "Unknown" and predicted_outcome in {"Four", "Six"}:
        flags.append("Boundary outcome predicted without a clear field zone")
    if missing_ball_frames >= 3:
        flags.append("Multiple missing-ball frames in sequence")
    if false_ball_jumps >= 2:
        flags.append("Possible false ball detections from unrealistic jumps")

    return flags


def _analysis_consistency(review_flags, ball_coverage):
    if not review_flags and ball_coverage is not None and ball_coverage >= 55:
        return "Good"
    if review_flags and ball_coverage is not None and ball_coverage >= 35:
        return "Needs Review"
    if review_flags:
        return "Poor"
    if ball_coverage is None:
        return "Unknown"
    return "Needs Review"


def _agent_quality(ball_coverage, review_flags, total_frames):
    if total_frames <= 0 or ball_coverage is None:
        return "Unknown"
    if ball_coverage >= 60 and len(review_flags) <= 1:
        return "High"
    if ball_coverage >= 35 and len(review_flags) <= 3:
        return "Medium"
    return "Low"


def _agent_confidence(agent_quality, ball_coverage, impact_result):
    impact_confidence = impact_result.get("impact_confidence", "Not Detected")
    if agent_quality == "High" and impact_confidence in {"High", "Medium"}:
        return "High"
    if agent_quality == "Medium":
        return "Medium"
    if agent_quality == "Low":
        return "Low"
    if ball_coverage is None:
        return "Unknown"
    return "Medium"


def _agent_notes(
    ball_coverage,
    bat_coverage,
    impact_result,
    direction_result,
    outcome_result,
    review_flags,
    missing_ball_frames,
    false_ball_jumps,
):
    coverage_text = "unknown"
    if ball_coverage is not None:
        coverage_text = f"{ball_coverage:.1f}%"

    impact_confidence = impact_result.get("impact_confidence", "Not Detected")
    field_zone = direction_result.get("field_zone", "Unknown")
    outcome = outcome_result.get("predicted_outcome", "Unknown")
    outcome_confidence = outcome_result.get("outcome_confidence", "Unknown")

    notes = (
        f"Ball tracking coverage was {coverage_text}"
        f"{f' with bat coverage {bat_coverage:.1f}%' if bat_coverage is not None else ''}. "
        f"Impact confidence is {impact_confidence}. "
    )
    if field_zone != "Unknown":
        notes += f"Shot direction appears toward {field_zone}. "
    else:
        notes += "Shot direction zone remained uncertain. "
    notes += (
        f"Outcome prediction is {outcome} ({outcome_confidence} confidence). "
        f"Missing ball frames: {missing_ball_frames}; possible false detections: {false_ball_jumps}."
    )
    if review_flags:
        notes += f" Review flags: {', '.join(review_flags)}."
    return notes
