"""Shared helpers for analysis pages (session persistence and report defaults)."""


def persist_result_to_session(result, source_type, video_name=None):
    """Save analysis result to local session storage without crashing the UI."""
    if not result or not result.get("success"):
        return result
    try:
        from Backends.src.storage.session_store import persist_analysis_to_session

        saved = persist_analysis_to_session(result, source_type, video_name=video_name)
        result["session_saved"] = True
        result["session_result_id"] = saved.get("id")
        result["session_save_error"] = None
    except Exception as error:
        result["session_saved"] = False
        result["session_save_error"] = str(error)
    return result


def ensure_delivery_report_fields(result):
    """Fill delivery report defaults for older or partial analysis results."""
    result.setdefault("ball_tracking_rate", result.get("ball_detection_rate", 0))
    result.setdefault("interpolated_ball_frames", 0)
    result.setdefault("estimated_line", "Unknown")
    result.setdefault("estimated_length", "Unknown")
    result.setdefault("estimated_bounce_point", None)
    result.setdefault("average_ball_confidence", 0)
    result.setdefault("kalman_predicted_frames", 0)
    result.setdefault("tracker_recoveries", 0)
    result.setdefault("overall_tracking_quality", "Poor")
    result.setdefault("pitch_normalized_bounce_point", None)
    result.setdefault("calibration_status", "Not calibrated")
    result.setdefault("calibration_source", "None")
    result.setdefault("calibration_warning", "Confidence warning: pitch calibration is missing.")
    result.setdefault("wagon_wheel", {})
