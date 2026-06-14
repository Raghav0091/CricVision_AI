def _get_number(result, key, default=0):
    value = result.get(key, default)

    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_text(result, key, default="Unknown"):
    value = result.get(key, default)

    if value is None or value == "":
        return default

    return str(value)


def _has_bounce_point(result):
    return result.get("estimated_bounce_point") is not None


def _quality_label(score):
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 45:
        return "Medium"
    return "Poor"


def calculate_detection_quality(result):
    ball_detection_rate = _get_number(result, "ball_detection_rate")
    stump_detection_rate = _get_number(result, "stump_detection_rate")
    average_ball_confidence = _get_number(result, "average_ball_confidence")
    ball_tracking_rate = _get_number(
        result,
        "ball_tracking_rate",
        ball_detection_rate,
    )
    interpolated_ball_frames = _get_number(result, "interpolated_ball_frames")
    estimated_line = _get_text(result, "estimated_line")
    estimated_length = _get_text(result, "estimated_length")

    confidence_score = min(max(average_ball_confidence, 0), 1) * 100
    quality_score = (
        ball_detection_rate * 0.25
        + ball_tracking_rate * 0.25
        + stump_detection_rate * 0.15
        + confidence_score * 0.15
    )

    if _has_bounce_point(result):
        quality_score += 8

    if estimated_line != "Unknown":
        quality_score += 6

    if estimated_length != "Unknown":
        quality_score += 6

    if interpolated_ball_frames > 12:
        quality_score -= min(18, (interpolated_ball_frames - 12) * 1.5)

    quality_score = int(round(min(max(quality_score, 0), 100)))

    return {
        "quality_score": quality_score,
        "quality_label": _quality_label(quality_score),
    }


def _describe_length(estimated_length):
    length_text = estimated_length.lower()

    if estimated_length == "Unknown":
        return "a delivery with unclear length"

    return f"a {length_text} ball"


def _describe_line(estimated_line):
    if estimated_line == "Off side":
        return "outside off stump"
    if estimated_line == "Leg side":
        return "toward the leg side"
    if estimated_line == "Middle":
        return "on or around middle stump"
    return "with an unclear line"


def generate_delivery_report(result):
    quality = calculate_detection_quality(result)
    estimated_line = _get_text(result, "estimated_line")
    estimated_length = _get_text(result, "estimated_length")
    ball_tracking_rate = _get_number(
        result,
        "ball_tracking_rate",
        _get_number(result, "ball_detection_rate"),
    )
    stump_detection_rate = _get_number(result, "stump_detection_rate")

    length_description = _describe_length(estimated_length)
    line_description = _describe_line(estimated_line)

    if ball_tracking_rate >= 70:
        tracking_sentence = "Ball tracking quality was good, with the ball followed through most of the delivery."
    elif ball_tracking_rate >= 45:
        tracking_sentence = "Ball tracking was usable, though a few parts of the delivery needed interpolation."
    else:
        tracking_sentence = "Ball tracking quality was limited, so the delivery estimate should be reviewed carefully."

    if stump_detection_rate >= 45 and estimated_line != "Unknown":
        stump_sentence = "Stumps were detected, so line estimation is reasonably reliable."
    elif estimated_line != "Unknown":
        stump_sentence = "Line was estimated, but stump visibility was limited."
    else:
        stump_sentence = "Line estimation is uncertain because stump detection was weak or unavailable."

    return (
        f"This delivery appears to be {length_description} {line_description}. "
        f"{tracking_sentence} {stump_sentence} "
        f"Overall analysis quality is {quality['quality_label'].lower()}."
    )


def generate_coaching_feedback(result):
    feedback = []
    estimated_line = _get_text(result, "estimated_line")
    estimated_length = _get_text(result, "estimated_length")
    ball_tracking_rate = _get_number(
        result,
        "ball_tracking_rate",
        _get_number(result, "ball_detection_rate"),
    )
    interpolated_ball_frames = _get_number(result, "interpolated_ball_frames")

    if estimated_length == "Good Length":
        feedback.append("Good length delivery: useful for building pressure.")
    elif estimated_length == "Full":
        feedback.append("Full length delivery: useful for swing or attacking the stumps.")
    elif estimated_length == "Yorker":
        feedback.append("Yorker length detected: strong attacking option if controlled.")
    elif estimated_length == "Short":
        feedback.append("Short length delivery: use it as a variation with clear intent.")
    else:
        feedback.append("Length is unclear, so record a longer clip with the bounce visible.")

    if estimated_line == "Off side":
        feedback.append("Line appears outside off stump.")
    elif estimated_line == "Leg side":
        feedback.append("Line appears toward the leg side.")
    elif estimated_line == "Middle":
        feedback.append("Line appears close to middle stump.")
    else:
        feedback.append("Line is unclear; keep stumps visible for a better line estimate.")

    if ball_tracking_rate < 45:
        feedback.append("Tracking quality was low, so record with better lighting or a closer camera.")
    elif ball_tracking_rate < 70:
        feedback.append("Tracking was usable, but a steadier camera angle will improve accuracy.")
    else:
        feedback.append("Ball tracking was strong enough for a useful first-pass review.")

    if interpolated_ball_frames > 8:
        feedback.append("Several ball positions were interpolated, so verify the processed video visually.")

    feedback.append("Use 60 FPS or 120 FPS for fast bowling.")

    return feedback[:5]


def detect_analysis_warnings(result):
    warnings = []
    ball_detection_rate = _get_number(result, "ball_detection_rate")
    stump_detection_rate = _get_number(result, "stump_detection_rate")
    average_ball_confidence = _get_number(result, "average_ball_confidence")
    interpolated_ball_frames = _get_number(result, "interpolated_ball_frames")
    estimated_line = _get_text(result, "estimated_line")
    estimated_length = _get_text(result, "estimated_length")

    if ball_detection_rate < 35:
        warnings.append("Low ball detection rate.")

    if stump_detection_rate < 35:
        warnings.append("Poor stump detection.")

    if not _has_bounce_point(result):
        warnings.append("Bounce point not found.")

    if estimated_line == "Unknown" or estimated_length == "Unknown":
        warnings.append("Line or length unknown.")

    if interpolated_ball_frames > 12:
        warnings.append("Too many interpolated frames.")

    if average_ball_confidence < 0.35:
        warnings.append("Low confidence.")

    return warnings
