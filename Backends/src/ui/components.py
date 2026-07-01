"""Page-level reusable UI blocks for CricVision AI."""

from datetime import datetime
from pathlib import Path

import streamlit as st

from Backends.src.calibration.calibration_context import (
    normalize_calibration_context,
)
from Backends.src.ui.theme import (
    render_empty_state,
    render_metric_card,
    render_section_title,
    render_status_pill,
)

DEBUG_IMPACT = False
DEBUG_SHOT_CLASSIFICATION = False
DEBUG_OUTCOME_PREDICTION = False
DEBUG_SHOT_DIRECTION = False
DEBUG_VISION_AGENT = False
DEBUG_PERFORMANCE = False


# LEGACY / NOT ACTIVE: Compatibility renderer for dev-only or future pages.
def hero_section(title, subtitle, description):
    st.markdown(
        f"""
        <div class="cv-hero">
            <h1>{title}</h1>
            <p style="color:#34d399;font-weight:600;margin-bottom:0.65rem;">{subtitle}</p>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# LEGACY / NOT ACTIVE: Compatibility renderer for dev-only or future pages.
def primary_action_card(title, description, button_label=None):
    st.markdown(
        f"""
        <div class="cv-glass" style="border-color:rgba(16,185,129,0.28);">
            <div style="color:#f8fafc;font-size:1.05rem;font-weight:700;margin-bottom:0.35rem;">{title}</div>
            <div style="color:#94a3b8;line-height:1.5;margin-bottom:0.75rem;">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if button_label:
        st.caption(button_label)


def metric_grid(items, columns=4):
    cols = st.columns(columns)
    for index, item in enumerate(items):
        label, value = item[0], item[1]
        hint = item[2] if len(item) > 2 else None
        status = item[3] if len(item) > 3 else None
        with cols[index % columns]:
            render_metric_card(label, value, hint, status)


def clean_upload_box(label="Upload cricket video"):
    st.markdown(
        f"""
        <div class="cv-glass" style="padding:0.85rem 1rem;margin-bottom:0.75rem;">
            <div style="color:#cbd5e1;font-weight:600;">{label}</div>
            <div style="color:#64748b;font-size:0.86rem;margin-top:0.25rem;">MP4, MOV, AVI, MKV supported</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def video_preview_card(title="Processed Delivery"):
    render_section_title(title)


def render_delivery_report(report):
    """Render a clean report-only delivery summary with no map visuals."""
    report = report or {}
    tracking_quality = _format_tracking_quality(report)

    st.subheader("Delivery Report")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Line", _display_value(report.get("estimated_line")))
    metric_cols[1].metric("Length", _display_value(report.get("estimated_length")))
    metric_cols[2].metric("Bounce Point", _format_bounce(report))
    metric_cols[3].metric("Ball Tracking Quality", tracking_quality)

    detail_cols = st.columns([1, 1])
    with detail_cols[0]:
        st.markdown("**Detected Objects**")
        st.info(_format_detected_objects(report))
    with detail_cols[1]:
        st.markdown("**Result Summary**")
        st.info(_format_result_summary(report))

    st.markdown("**AI Coach Feedback**")
    for feedback_item in _format_coach_feedback(report):
        st.markdown(f"- {feedback_item}")


def build_analysis_summary_data(result):
    """Build a compact summary payload for delivery analysis results."""
    result = result or {}
    calibration = normalize_calibration_context(result.get("calibration_context"))
    impact = _normalize_impact(result)
    shot = _normalize_shot(result)
    direction = _normalize_direction(result)
    outcome = _normalize_outcome(result)
    observer = _normalize_observer_timeline(result)
    repair = result.get("visual_observer_repair") or {}

    stump = (calibration.get("stumps") or {}).get("batter_end") or {}
    if not calibration.get("enabled"):
        calibration_display = "Disabled"
    elif str(stump.get("source") or "").lower() == "estimated":
        calibration_display = f"Estimated / {calibration.get('calibration_quality', 'Low')}"
    else:
        calibration_display = _display_value(calibration.get("calibration_quality"))

    bat_coverage = observer.get("bat_detection_coverage")
    bat_coverage_zero = False
    try:
        bat_coverage_zero = bat_coverage is not None and float(bat_coverage) == 0.0
    except (TypeError, ValueError):
        bat_coverage_zero = False

    impact_detected = bool(impact.get("impact_detected"))
    shot_available = impact_detected and not bat_coverage_zero

    coach_note = _build_analysis_coach_note(
        result,
        impact_detected=impact_detected,
        bat_coverage_zero=bat_coverage_zero,
    )

    return {
        "line": _display_value(result.get("estimated_line")),
        "length": _display_value(result.get("estimated_length")),
        "ball_tracking": _format_tracking_quality(result),
        "calibration_quality": calibration_display,
        "visual_observer_repair_confidence": _display_value(
            repair.get("repair_confidence", "Unknown")
        ),
        "impact_status": "Detected" if impact_detected else "Not Detected",
        "shot_type": _display_value(shot.get("shot_type"))
        if shot_available
        else "Unavailable",
        "field_zone": _display_value(direction.get("field_zone"))
        if shot_available
        else "Unavailable",
        "predicted_outcome": _display_value(outcome.get("predicted_outcome"))
        if shot_available
        else "Unavailable",
        "coach_note": coach_note,
        "impact_detected": impact_detected,
        "bat_coverage_zero": bat_coverage_zero,
        "shot_available": shot_available,
    }


def render_analysis_summary_card(result):
    """Render a compact professional delivery summary."""
    summary = build_analysis_summary_data(result)

    st.subheader("Quick Result Summary")
    metric_grid(
        [
            ("Line", summary["line"]),
            ("Length", summary["length"]),
            ("Ball Tracking Quality", summary["ball_tracking"]),
            ("Calibration Quality", summary["calibration_quality"]),
            ("Visual Observer Repair Confidence", summary["visual_observer_repair_confidence"]),
            ("Impact Status", summary["impact_status"]),
            ("Shot Type", summary["shot_type"]),
            ("Field Zone", summary["field_zone"]),
            ("Predicted Outcome", summary["predicted_outcome"]),
        ],
        columns=3,
    )

    if summary["bat_coverage_zero"]:
        st.warning(
            "Shot analysis unavailable because bat was not detected "
            "(0% bat detection coverage)."
        )
    elif not summary["impact_detected"]:
        st.info(
            "Shot and outcome analysis are unavailable because bat-ball impact "
            "was not detected."
        )

    if summary["visual_observer_repair_confidence"] == "Medium":
        st.caption(
            "Visual observer repair confidence is medium — treat tracking repairs "
            "with moderate caution."
        )

    st.markdown("**Coach Note**")
    st.info(summary["coach_note"])


def render_processed_video_preview(result, download_key="download_processed_video"):
    """Validate and preview the processed video with a safe fallback."""
    from Backends.src.video_pipeline.annotation_writer import (
        validate_processed_video_path,
    )

    result = result or {}
    if result.get("processed_video_skipped") or not result.get(
        "processed_video_generated", True
    ):
        st.info("Processed video generation skipped to speed up analysis.")
        return

    output_path = result.get("output_path")
    raw_output_path = result.get("raw_output_path")
    validation = validate_processed_video_path(output_path)
    fallback_validation = (
        validate_processed_video_path(raw_output_path)
        if raw_output_path and str(raw_output_path) != str(output_path)
        else {"valid": False}
    )

    if validation["can_preview"]:
        st.video(str(output_path))
        with open(output_path, "rb") as video_file:
            st.download_button(
                "Download Processed Video",
                data=video_file,
                file_name=Path(output_path).name,
                mime="video/mp4",
                use_container_width=True,
                key=download_key,
            )
        return

    if result.get("processed_video_conversion") == "failed":
        st.warning(
            "Browser MP4 conversion failed. Analysis results are still available."
        )
        if result.get("processed_video_conversion_error"):
            st.caption(str(result["processed_video_conversion_error"]))

    if fallback_validation.get("can_preview"):
        st.caption("Showing the raw processed video because browser conversion failed.")
        st.video(str(raw_output_path))
        with open(raw_output_path, "rb") as video_file:
            st.download_button(
                "Download Raw Processed Video",
                data=video_file,
                file_name=Path(raw_output_path).name,
                mime="video/mp4",
                use_container_width=True,
                key=f"{download_key}_raw",
            )
        return

    st.warning(
        "Processed video could not be previewed, but analysis results are available."
    )
    if validation.get("error"):
        st.caption(validation["error"])


def render_impact_and_shot_section(result):
    """Render impact and shot reports, hiding noisy unknown cards by default."""
    impact = _normalize_impact(result)
    summary = build_analysis_summary_data(result)

    render_impact_report(result)
    render_impact_frame_preview(result)

    if not summary["shot_available"]:
        with st.expander("Unavailable shot/outcome details", expanded=False):
            render_shot_report(result)
            render_shot_direction_report(result)
            render_outcome_prediction(result)
        return

    render_shot_report(result)
    render_shot_direction_report(result)
    render_outcome_prediction(result)


def render_video_analysis_results_layout(
    result,
    *,
    context_label="Video Analysis",
    show_status_banner=False,
):
    """Organize analysis results into a clean default view with tabbed details."""
    from Backends.src.ui.theme import render_status_pill

    result = result or {}
    if show_status_banner:
        st.markdown(
            f'<div style="margin:0.75rem 0 1rem 0;">{render_status_pill("Analysis Complete", "success")} '
            f'{render_status_pill(result.get("analysis_mode", "Full Delivery Analysis"), "gold")}</div>',
            unsafe_allow_html=True,
        )

    video_preview_card("Processed Video Preview")
    render_processed_video_preview(
        result,
        download_key=f"download_{context_label.lower().replace(' ', '_')}",
    )

    render_analysis_summary_card(result)
    render_save_status(result, context_label)

    (
        tab_summary,
        tab_tracking,
        tab_impact,
        tab_calibration,
        tab_technical,
    ) = st.tabs(
        [
            "Summary",
            "Tracking Quality",
            "Impact & Shot",
            "Calibration",
            "Technical Details",
        ]
    )

    with tab_summary:
        render_delivery_report(result)

    with tab_tracking:
        render_visual_observer_repair_card(result)
        render_observer_timeline_report(result)
        render_vision_agent_report(result)

    with tab_impact:
        render_impact_and_shot_section(result)

    with tab_calibration:
        render_calibration_context_card(result)

    with tab_technical:
        render_performance_details(result)
        if result.get("report_path"):
            st.caption(f"Report JSON: {result['report_path']}")
        if result.get("output_path"):
            st.caption(f"Processed video: {result['output_path']}")
        elif result.get("raw_output_path"):
            st.caption(f"Raw processed video: {result['raw_output_path']}")


def render_impact_report(impact):
    """Render a safe bat-ball impact report."""
    impact = _normalize_impact(impact)

    st.subheader("Impact Report")
    impact_cols = st.columns(5)
    impact_cols[0].metric("Bat-Ball Impact", "Detected" if impact["impact_detected"] else "Not Detected")
    impact_cols[1].metric("Impact Frame", _display_value(impact.get("impact_frame")))
    impact_cols[2].metric("Impact Time", _format_seconds(impact.get("impact_time_sec")))
    impact_cols[3].metric("Ball-to-Bat Distance", _format_pixel_distance(impact.get("min_ball_bat_distance_px")))
    impact_cols[4].metric("Impact Confidence", _display_value(impact.get("impact_confidence")))

    reason = impact.get("reason") or impact.get("impact_reason")
    if reason:
        st.info(str(reason))
    else:
        st.info("Impact detection data is not available for this result.")

    if DEBUG_IMPACT:
        with st.expander("Impact Debug", expanded=False):
            st.json(impact.get("debug", {}))


def render_impact_frame_preview(result_or_impact):
    """Render the likely impact frame preview if one was saved."""
    impact = _normalize_impact(result_or_impact)
    image_path = impact.get("impact_frame_image_path")

    st.subheader("Impact Frame Preview")
    if not image_path:
        if impact.get("impact_detected"):
            st.info("Impact frame preview is not available for this result.")
        else:
            st.info("No impact frame preview because impact was not detected.")
        return

    path = Path(str(image_path))
    if not path.exists():
        st.warning("Impact frame preview file is missing.")
        return

    st.image(str(path), caption="Likely impact frame", use_container_width=True)


def render_shot_report(shot):
    """Render a safe rule-based shot type report."""
    shot = _normalize_shot(shot)

    st.subheader("Shot Report")
    shot_cols = st.columns(4)
    shot_cols[0].metric("Shot Type", _display_value(shot.get("shot_type")))
    shot_cols[1].metric("Shot Confidence", _display_value(shot.get("shot_confidence")))
    shot_cols[2].metric("Shot Direction", _display_value(shot.get("shot_direction")))
    shot_cols[3].metric("Shot Height", _display_value(shot.get("shot_height")))

    reason = shot.get("reason") or shot.get("shot_reason")
    if reason:
        st.info(str(reason))
    else:
        st.info("Shot type detection requires impact frame and post-impact ball tracking.")

    if DEBUG_SHOT_CLASSIFICATION:
        with st.expander("Shot Classification Debug", expanded=False):
            st.json(shot.get("debug", {}))


def render_outcome_prediction(outcome):
    """Render a safe predicted shot outcome report."""
    outcome = _normalize_outcome(outcome)

    st.subheader("Outcome Prediction")
    outcome_cols = st.columns(5)
    outcome_cols[0].metric("Predicted Outcome", _display_value(outcome.get("predicted_outcome")))
    outcome_cols[1].metric("Outcome Confidence", _display_value(outcome.get("outcome_confidence")))
    outcome_cols[2].metric("Run Estimate", _format_run_estimate(outcome.get("run_estimate")))
    outcome_cols[3].metric("Dismissal Risk", _display_value(outcome.get("dismissal_risk")))
    outcome_cols[4].metric("Boundary Chance", _display_value(outcome.get("boundary_chance")))

    reason = outcome.get("reason") or outcome.get("outcome_reason")
    if reason:
        st.info(str(reason))
    else:
        st.info("Outcome prediction requires impact frame and post-impact ball tracking.")

    if DEBUG_OUTCOME_PREDICTION:
        with st.expander("Outcome Prediction Debug", expanded=False):
            st.json(outcome.get("debug", {}))


def render_shot_direction_report(direction):
    """Render text-based shot direction and field zone report safely."""
    direction = _normalize_direction(direction)

    st.subheader("Shot Direction / Field Zone")
    direction_cols = st.columns(3)
    direction_cols[0].metric("Shot Direction", _display_value(direction.get("shot_direction")))
    direction_cols[1].metric("Field Zone", _display_value(direction.get("field_zone")))
    direction_cols[2].metric("Zone Confidence", _display_value(direction.get("zone_confidence")))

    angle = direction.get("direction_angle_degrees")
    if angle is not None:
        st.metric("Direction Angle", f"{float(angle):.1f}°")
    else:
        st.metric("Direction Angle", "N/A")

    reason = direction.get("reason") or direction.get("direction_reason")
    if reason:
        st.info(str(reason))
    elif direction.get("field_zone") in {None, "", "Unknown"}:
        st.info("Field zone estimation requires impact frame and post-impact ball tracking.")

    if DEBUG_SHOT_DIRECTION:
        with st.expander("Shot Direction Debug", expanded=False):
            st.json(direction.get("debug", {}))


def render_vision_agent_report(agent):
    """Render CricVision Agent Review safely."""
    agent = _normalize_agent(agent)

    st.subheader("CricVision Agent Review")
    quality = agent.get("agent_quality", "Unknown")
    if quality == "High":
        st.success(f"Agent quality: {quality}")
    elif quality == "Medium":
        st.info(f"Agent quality: {quality}")
    elif quality == "Low":
        st.warning(f"Agent quality: {quality}")
    else:
        st.info(f"Agent quality: {_display_value(quality)}")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Agent Confidence", _display_value(agent.get("agent_confidence")))
    metric_cols[1].metric(
        "Ball Tracking Coverage",
        _format_percentage(agent.get("ball_tracking_coverage")),
    )
    metric_cols[2].metric(
        "Bat Detection Coverage",
        _format_percentage(agent.get("bat_detection_coverage")),
    )
    metric_cols[3].metric("Analysis Consistency", _display_value(agent.get("analysis_consistency")))

    detail_cols = st.columns(2)
    detail_cols[0].metric("Missing Ball Frames", _display_value(agent.get("missing_ball_frames")))
    detail_cols[1].metric(
        "Possible False Ball Detections",
        _display_value(agent.get("possible_false_ball_detections")),
    )

    stump_coverage = agent.get("stump_detection_coverage")
    if stump_coverage is not None:
        st.metric("Stump Detection Coverage", _format_percentage(stump_coverage))

    review_flags = agent.get("review_flags") or []
    if review_flags:
        st.warning("Review flags:\n" + "\n".join(f"- {flag}" for flag in review_flags))

    notes = agent.get("agent_notes")
    if notes:
        st.info(str(notes))

    if agent.get("review_frames_recommended"):
        st.caption(_display_value(agent.get("review_reason"), "Manual frame review recommended."))

    if DEBUG_VISION_AGENT:
        with st.expander("Vision Agent Debug", expanded=False):
            st.json(agent.get("debug", {}))


def render_observer_timeline_report(timeline):
    """Render observer timeline summary safely."""
    timeline = _normalize_observer_timeline(timeline)

    st.subheader("Observer Timeline / Detection Quality")
    quality = timeline.get("detection_quality", "Unknown")
    if quality == "High":
        st.success(f"Detection quality: {quality}")
    elif quality == "Medium":
        st.info(f"Detection quality: {quality}")
    elif quality == "Low":
        st.warning(f"Detection quality: {quality}")
    else:
        st.info(f"Detection quality: {_display_value(quality)}")

    cols = st.columns(4)
    cols[0].metric("Total Frames", _display_value(timeline.get("total_frames")))
    cols[1].metric("Frames Processed", _display_value(timeline.get("processed_frames")))
    cols[2].metric("Ball Tracking Coverage", _format_percentage(timeline.get("ball_tracking_coverage")))
    cols[3].metric("Bat Detection Coverage", _format_percentage(timeline.get("bat_detection_coverage")))

    cols2 = st.columns(4)
    cols2[0].metric("Stump Detection Coverage", _format_percentage(timeline.get("stump_detection_coverage")))
    cols2[1].metric("Missing Ball Frames", _display_value(timeline.get("missing_ball_frames")))
    cols2[2].metric("Low Confidence Frames", _display_value(timeline.get("low_confidence_ball_frames")))
    cols2[3].metric(
        "Possible False Ball Detections",
        _display_value(timeline.get("possible_false_ball_detections")),
    )

    notes = timeline.get("observer_notes")
    if notes:
        st.info(str(notes))
    else:
        st.info("Observer timeline data is not available for this result.")


def render_calibration_context_card(calibration_context, compact=False):
    """Render text-only practice-environment calibration context."""
    data = calibration_context or {}
    if isinstance(data, dict) and "calibration_context" in data:
        data = data.get("calibration_context")
    data = normalize_calibration_context(data)

    st.subheader("Calibration Context")
    enabled = bool(data.get("enabled"))
    stump = (data.get("stumps") or {}).get("batter_end") or {}
    corridor = data.get("pitch_corridor") or {}

    if compact:
        cols = st.columns(4)
        cols[0].metric("Enabled", "Yes" if enabled else "No")
        cols[1].metric(
            "Camera View",
            _display_value(data.get("camera_view")).replace("_", " ").title(),
        )
        cols[2].metric(
            "Handedness",
            _display_value(data.get("batter_handedness")).replace("_", " ").title(),
        )
        cols[3].metric(
            "Quality",
            _display_value(data.get("calibration_quality")),
        )
        st.caption(
            "Stumps: "
            f"{_display_value(stump.get('status')).title()} "
            f"({_display_value(stump.get('source'))}); "
            "pitch corridor: "
            f"{_display_value(corridor.get('status')).title()}."
        )
    else:
        cols = st.columns(4)
        cols[0].metric("Calibration Enabled", "Yes" if enabled else "No")
        cols[1].metric(
            "Camera View",
            _display_value(data.get("camera_view")).replace("_", " ").title(),
        )
        cols[2].metric(
            "Batter Handedness",
            _display_value(data.get("batter_handedness")).replace("_", " ").title(),
        )
        cols[3].metric(
            "Calibration Quality",
            _display_value(data.get("calibration_quality")),
        )
        detail_cols = st.columns(2)
        detail_cols[0].metric(
            "Stumps",
            f"{_display_value(stump.get('status')).title()} "
            f"({_display_value(stump.get('source'))})",
        )
        detail_cols[1].metric(
            "Pitch Corridor",
            _display_value(corridor.get("status")).title(),
        )

    notes = data.get("notes") or []
    if not enabled and not notes:
        notes = ["Practice environment calibration was disabled for this analysis."]
    for note in notes:
        st.caption(f"• {note}")


def render_visual_observer_repair_card(repair_report, compact=False):
    """Render Visual Observer tracking-repair quality without map output."""
    data = repair_report or {}
    if isinstance(data, dict) and "visual_observer_repair" in data:
        data = data.get("visual_observer_repair") or {}
    if not isinstance(data, dict) or not data:
        return

    st.subheader("Visual Observer Repair")
    confidence = data.get("repair_confidence", "Unknown")
    if confidence == "High":
        st.success(f"Repair confidence: {confidence}")
    elif confidence == "Medium":
        st.info(f"Repair confidence: {confidence}")
    elif confidence == "Low":
        st.warning(f"Repair confidence: {confidence}")
    else:
        st.info(f"Repair confidence: {_display_value(confidence)}")

    if compact:
        cols = st.columns(3)
        cols[0].metric(
            "Original Coverage",
            _format_percentage(data.get("original_coverage")),
        )
        cols[1].metric(
            "Repaired Coverage",
            _format_percentage(data.get("repaired_coverage")),
        )
        cols[2].metric(
            "Repaired Frames",
            _display_value(data.get("repaired_frames")),
        )
    else:
        cols = st.columns(3)
        cols[0].metric(
            "Original Ball Tracking Coverage",
            _format_percentage(data.get("original_coverage")),
        )
        cols[1].metric(
            "Repaired Ball Tracking Coverage",
            _format_percentage(data.get("repaired_coverage")),
        )
        cols[2].metric(
            "Missing Ball Frames",
            _display_value(data.get("missing_frames")),
        )

        cols2 = st.columns(2)
        cols2[0].metric(
            "Repaired Frames",
            _display_value(data.get("repaired_frames")),
        )
        cols2[1].metric(
            "Suspicious / False Detections Downgraded",
            _display_value(
                data.get(
                    "suspicious_detections",
                    data.get("removed_or_downgraded_frames"),
                )
            ),
        )

    decision = data.get("agent_decision")
    if decision:
        st.info(f"Agent decision: {decision}")
    notes = data.get("notes") or []
    if isinstance(notes, str):
        notes = [notes]
    for note in notes:
        st.caption(f"• {note}")


def render_performance_details(result):
    """Render optional analysis performance metrics."""
    result = result or {}
    if not (result.get("show_performance_details") or DEBUG_PERFORMANCE):
        return

    profile = result.get("performance_profile") or {}
    if not profile:
        return

    with st.expander("Performance Details", expanded=False):
        cols = st.columns(3)
        cols[0].metric("Total Analysis Time", _format_seconds(profile.get("total_analysis_time_sec")))
        cols[1].metric("Video Read Time", _format_seconds(profile.get("video_read_time_sec")))
        cols[2].metric("Video Write Time", _format_seconds(profile.get("annotation_write_time_sec")))
        cols2 = st.columns(3)
        cols2[0].metric("Ball Detection Time", _format_seconds(profile.get("ball_detection_time_sec")))
        cols2[1].metric("Bat Detection Time", _format_seconds(profile.get("bat_detection_time_sec")))
        cols2[2].metric("Stump Detection Time", _format_seconds(profile.get("stump_detection_time_sec")))
        cols3 = st.columns(3)
        cols3[0].metric("Observer Timeline Time", _format_seconds(profile.get("observer_timeline_time_sec")))
        cols3[1].metric("Report Generation Time", _format_seconds(profile.get("report_generation_time_sec")))
        cols3[2].metric("Model Inference Time", _format_seconds(profile.get("model_inference_time_sec")))
        cols4 = st.columns(3)
        cols4[0].metric("Frames Processed", _display_value(profile.get("frames_processed")))
        cols4[1].metric("Frames Read", _display_value(profile.get("frames_read")))
        cols4[2].metric(
            "Avg Time / Frame",
            _format_seconds(profile.get("avg_time_per_frame_sec")),
        )
        cols5 = st.columns(2)
        cols5[0].metric(
            "Avg ms / Processed Frame",
            _display_value(profile.get("average_ms_per_processed_frame")),
        )
        cols5[1].metric(
            "Processed Video",
            "Yes" if profile.get("processed_video_generated", True) else "Skipped",
        )
        st.caption(f"Analysis mode: {profile.get('speed_mode', result.get('speed_mode', 'Unknown'))}")
        if profile.get("smart_pipeline_used"):
            st.caption("Smart accurate video pipeline enabled.")
        invalid_count = profile.get("invalid_detection_count")
        if invalid_count:
            st.caption(f"Skipped invalid detections: {invalid_count}")


def render_save_status(result, context_label="Analysis"):
    """Render save/session status without exposing raw debug output."""
    result = result or {}
    st.subheader("Save Status")

    if result.get("session_saved"):
        st.success("Saved to Session Results")
    elif result.get("session_save_error"):
        st.warning(f"Could not save to Session Results: {result['session_save_error']}")
    elif result.get("success"):
        st.info(f"{context_label} completed. Session save status is unavailable.")

    status_lines = []
    if result.get("session_result_id"):
        status_lines.append(f"Session result id: `{result['session_result_id']}`")
    if result.get("report_path"):
        status_lines.append(f"Report saved: `{result['report_path']}`")
    if result.get("output_path"):
        status_lines.append(f"Processed video saved: `{result['output_path']}`")
    if result.get("processed_file_name"):
        status_lines.append(f"Processed clip ready: `{result['processed_file_name']}`")

    for line in status_lines:
        st.markdown(f"- {line}")

    if not result.get("session_saved") and not result.get("success"):
        st.warning(f"{context_label} result is not available yet.")


def render_session_summary(summary: dict):
    """Render summary metric cards."""
    summary = summary or {}
    st.subheader("Overall Summary")
    cols = st.columns(4)
    cols[0].metric("Total Deliveries Analyzed", summary.get("total_deliveries", 0))
    cols[1].metric("Total Predicted Runs", summary.get("total_predicted_runs", 0))
    cols[2].metric("Most Common Shot Type", _display_value(summary.get("most_common_shot_type")))
    cols[3].metric("Most Common Field Zone", _display_value(summary.get("most_common_field_zone")))

    cols2 = st.columns(4)
    cols2[0].metric("Most Common Outcome", _display_value(summary.get("most_common_outcome")))
    cols2[1].metric("Average Agent Quality", _display_value(summary.get("average_agent_quality")))
    coverage = summary.get("average_ball_tracking_coverage")
    cols2[2].metric(
        "Average Ball Tracking Coverage",
        f"{coverage:.1f}%" if coverage is not None else "N/A",
    )
    cols2[3].metric("Most Common Length", _display_value(summary.get("most_common_length")))

    insights = summary.get("insights") or []
    if insights:
        st.markdown("**Session Insights**")
        for insight in insights:
            st.markdown(f"- {insight}")


def render_session_analytics_sections(summary: dict):
    """Render optional tendency and reliability sections."""
    summary = summary or {}

    st.subheader("Batting Tendencies")
    tendency_cols = st.columns(2)
    with tendency_cols[0]:
        st.markdown("**Shot Type Distribution**")
        shot_dist = summary.get("shot_type_distribution") or {}
        if shot_dist:
            st.bar_chart(shot_dist)
        else:
            st.info("No shot type data yet.")
    with tendency_cols[1]:
        st.markdown("**Field Zone Distribution**")
        zone_dist = summary.get("field_zone_distribution") or {}
        if zone_dist:
            st.bar_chart(zone_dist)
        else:
            st.info("No field zone data yet.")

    st.subheader("Bowling / Delivery Tendencies")
    delivery_cols = st.columns(2)
    with delivery_cols[0]:
        st.markdown("**Line Distribution**")
        line_dist = summary.get("line_distribution") or {}
        if line_dist:
            st.bar_chart(line_dist)
        else:
            st.info("No line data yet.")
    with delivery_cols[1]:
        st.markdown("**Length Distribution**")
        length_dist = summary.get("length_distribution") or {}
        if length_dist:
            st.bar_chart(length_dist)
        else:
            st.info("No length data yet.")

    st.subheader("Agent Quality / Reliability")
    agent_cols = st.columns(2)
    with agent_cols[0]:
        st.markdown("**Agent Quality Distribution**")
        agent_dist = summary.get("agent_quality_distribution") or {}
        if agent_dist:
            st.bar_chart(agent_dist)
        else:
            st.info("No agent quality data yet.")
    with agent_cols[1]:
        st.markdown("**Outcome Distribution**")
        outcome_dist = summary.get("outcome_distribution") or {}
        if outcome_dist:
            st.bar_chart(outcome_dist)
        else:
            st.info("No outcome data yet.")


def render_results_filters(results: list):
    """Render filters and return filtered results."""
    results = results or []
    with st.expander("Filters", expanded=False):
        filter_cols = st.columns(3)
        source_options = ["All"] + sorted(
            {str(item.get("source_type", "Unknown")) for item in results if item.get("source_type")}
        )
        shot_options = ["All"] + sorted(
            {str(item.get("shot_type", "Unknown")) for item in results if item.get("shot_type")}
        )
        zone_options = ["All"] + sorted(
            {str(item.get("field_zone", "Unknown")) for item in results if item.get("field_zone")}
        )
        outcome_options = ["All"] + sorted(
            {
                str(item.get("predicted_outcome", "Unknown"))
                for item in results
                if item.get("predicted_outcome")
            }
        )
        agent_options = ["All"] + sorted(
            {str(item.get("agent_quality", "Unknown")) for item in results if item.get("agent_quality")}
        )

        source_filter = filter_cols[0].selectbox("Source Type", source_options, key="session_filter_source")
        shot_filter = filter_cols[1].selectbox("Shot Type", shot_options, key="session_filter_shot")
        zone_filter = filter_cols[2].selectbox("Field Zone", zone_options, key="session_filter_zone")

        filter_cols2 = st.columns(3)
        outcome_filter = filter_cols2[0].selectbox(
            "Predicted Outcome",
            outcome_options,
            key="session_filter_outcome",
        )
        agent_filter = filter_cols2[1].selectbox("Agent Quality", agent_options, key="session_filter_agent")
        search_text = filter_cols2[2].text_input("Search video name", value="", key="session_filter_search")

    filtered = []
    search_lower = search_text.strip().lower()
    for item in results:
        if source_filter != "All" and str(item.get("source_type", "Unknown")) != source_filter:
            continue
        if shot_filter != "All" and str(item.get("shot_type", "Unknown")) != shot_filter:
            continue
        if zone_filter != "All" and str(item.get("field_zone", "Unknown")) != zone_filter:
            continue
        if outcome_filter != "All" and str(item.get("predicted_outcome", "Unknown")) != outcome_filter:
            continue
        if agent_filter != "All" and str(item.get("agent_quality", "Unknown")) != agent_filter:
            continue
        if search_lower:
            video_name = str(item.get("video_name", "")).lower()
            if search_lower not in video_name:
                continue
        filtered.append(item)

    return list(reversed(filtered))


def render_session_result_card(result: dict, expanded: bool = False):
    """Render one saved result safely with summary and expandable details."""
    from Backends.src.storage.session_store import session_record_to_report_view

    result = result or {}
    report_view = session_record_to_report_view(result)
    title = result.get("video_name", "Delivery")
    created_at = _format_created_at(result.get("created_at"))
    source_type = _display_value(result.get("source_type"))

    with st.expander(f"{title} — {created_at} ({source_type})", expanded=expanded):
        render_analysis_summary_card(report_view)

        video_path = result.get("processed_video_path") or report_view.get("output_path")
        if video_path:
            preview_result = {
                **report_view,
                "output_path": video_path,
                "processed_video_generated": True,
            }
            render_processed_video_preview(
                preview_result,
                download_key=f"session_video_{result.get('id', title)}",
            )

        (
            tab_summary,
            tab_tracking,
            tab_impact,
            tab_calibration,
        ) = st.tabs(["Summary", "Tracking", "Impact & Shot", "Calibration"])

        with tab_summary:
            render_delivery_report(report_view)

        with tab_tracking:
            render_observer_timeline_report(report_view)
            if (report_view.get("calibration_context") or {}).get("enabled"):
                render_calibration_context_card(report_view, compact=True)
            render_visual_observer_repair_card(report_view, compact=True)
            render_vision_agent_report(report_view)

        with tab_impact:
            impact_path = result.get("impact_frame_image_path")
            render_impact_and_shot_section(report_view)
            if impact_path:
                path = Path(str(impact_path))
                if path.exists():
                    st.image(
                        str(path),
                        caption="Impact frame preview",
                        use_container_width=True,
                    )

        with tab_calibration:
            render_calibration_context_card(report_view)


def analysis_report_card(result):
    render_delivery_report(result)


def delivery_summary_card(result):
    if result is None or not result.get("success"):
        render_empty_state(
            "No delivery analyzed yet",
            "Record or upload a delivery to see a summary here.",
        )
        return

    analysis_report_card(result)


# LEGACY / NOT ACTIVE: Compatibility renderer for dev-only or future pages.
def report_history_card(report):
    st.markdown(
        f"""
        <div class="cv-glass" style="margin-bottom:0.75rem;">
            <div style="display:flex;justify-content:space-between;gap:0.75rem;flex-wrap:wrap;">
                <div>
                    <div style="color:#f8fafc;font-weight:700;">{report.get("title", "Delivery Report")}</div>
                    <div style="color:#64748b;font-size:0.84rem;margin-top:0.2rem;">{report.get("timestamp", "—")}</div>
                </div>
                {render_status_pill(report.get("tracking_quality", "Report"), "success" if report.get("tracking_quality") in {"Excellent", "Good"} else "default")}
            </div>
            <div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0.75rem;margin-top:0.85rem;">
                <div><div style="color:#64748b;font-size:0.75rem;">Line</div><div style="color:#e2e8f0;font-weight:700;">{report.get("line", "—")}</div></div>
                <div><div style="color:#64748b;font-size:0.75rem;">Length</div><div style="color:#e2e8f0;font-weight:700;">{report.get("length", "—")}</div></div>
                <div><div style="color:#64748b;font-size:0.75rem;">Type</div><div style="color:#e2e8f0;font-weight:700;">{report.get("analysis_type", "—")}</div></div>
                <div><div style="color:#64748b;font-size:0.75rem;">Impact</div><div style="color:#e2e8f0;font-weight:700;">{report.get("impact_confidence", "—")}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def model_status_card():
    from Backends.src.models.model_registry import validate_model_paths

    render_section_title("Model Status")
    cols = st.columns(2)
    statuses = list(validate_model_paths().values())
    for index, status in enumerate(statuses):
        with cols[index % 2]:
            if status["local_ready"]:
                hint = "Ready for analysis"
                if status["lazy_only"]:
                    hint += " (lazy only)"
                pill_status = "Ready"
            elif status["remote_available"]:
                hint = "Downloads from Hugging Face on first use"
                if status["lazy_only"]:
                    hint += " (lazy only)"
                pill_status = "Remote available"
            else:
                hint = status["path"]
                pill_status = "Missing"
            render_metric_card(
                status["name"],
                status["status"],
                hint=hint,
                status=pill_status,
            )


# LEGACY / NOT ACTIVE: The active field setup card lives in interactive_field_map.py.
def field_setup_card(title, body_html):
    st.markdown(
        f"""
        <div class="cv-glass">
            <div style="color:#34d399;font-weight:700;margin-bottom:0.45rem;">{title}</div>
            <div style="color:#cbd5e1;line-height:1.5;">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# LEGACY / NOT ACTIVE: Kept for optional developer-facing report details.
def developer_details_expander(result):
    with st.expander("Developer Details", expanded=False):
        st.json(result)


# LEGACY / NOT ACTIVE: Kept for optional coaching detail layouts.
def coaching_details_expander(result):
    from Backends.src.analysis.cricket_agent import (
        detect_analysis_warnings,
        generate_coaching_feedback,
        generate_delivery_report,
        calculate_detection_quality,
    )

    quality = calculate_detection_quality(result)
    report = generate_delivery_report(result)
    feedback_items = generate_coaching_feedback(result)
    warnings = detect_analysis_warnings(result)

    st.markdown(f"**Analysis Quality:** {quality['quality_score']}/100 — {quality['quality_label']}")
    st.info(report)
    st.markdown("**Coaching Feedback**")
    for item in feedback_items:
        st.markdown(f"- {item}")
    if warnings:
        st.warning("\n".join(f"- {warning}" for warning in warnings))


def _display_value(value, default="N/A"):
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    return str(value)


def _normalize_impact(result_or_impact):
    data = result_or_impact or {}
    if isinstance(data, dict) and "impact_info" in data:
        data = data.get("impact_info") or {}
    if not isinstance(data, dict):
        data = {}

    distance = data.get("min_ball_bat_distance_px", data.get("min_distance"))
    confidence = data.get("impact_confidence", "Not Detected")
    impact_frame = data.get("impact_frame")
    impact_detected = data.get("impact_detected")
    if impact_detected is None:
        impact_detected = impact_frame is not None and confidence != "Not Detected"

    return {
        **data,
        "impact_detected": bool(impact_detected),
        "impact_frame": impact_frame,
        "impact_time_sec": data.get("impact_time_sec"),
        "min_ball_bat_distance_px": distance,
        "impact_confidence": confidence or "Not Detected",
        "reason": data.get("reason", data.get("impact_reason", "")),
        "impact_frame_image_path": data.get("impact_frame_image_path"),
    }


def _normalize_shot(result_or_shot):
    data = result_or_shot or {}
    if isinstance(data, dict) and "shot_info" in data:
        data = data.get("shot_info") or data
    if not isinstance(data, dict):
        data = {}

    reason = data.get("reason", data.get("shot_reason", ""))
    return {
        **data,
        "shot_type": data.get("shot_type", "Unknown"),
        "shot_confidence": data.get("shot_confidence", "Unknown"),
        "shot_direction": data.get("shot_direction", "Unknown"),
        "shot_height": data.get("shot_height", "Unknown"),
        "reason": reason,
        "shot_reason": reason,
    }


def _normalize_outcome(result_or_outcome):
    data = result_or_outcome or {}
    if isinstance(data, dict) and "outcome_info" in data:
        data = data.get("outcome_info") or data
    if not isinstance(data, dict):
        data = {}

    reason = data.get("reason", data.get("outcome_reason", ""))
    return {
        **data,
        "predicted_outcome": data.get("predicted_outcome", "Unknown"),
        "outcome_confidence": data.get("outcome_confidence", "Unknown"),
        "run_estimate": data.get("run_estimate"),
        "dismissal_risk": data.get("dismissal_risk", "Unknown"),
        "boundary_chance": data.get("boundary_chance", "Unknown"),
        "reason": reason,
        "outcome_reason": reason,
    }


def _normalize_direction(result_or_direction):
    data = result_or_direction or {}
    if isinstance(data, dict) and "direction_info" in data:
        data = data.get("direction_info") or data
    if not isinstance(data, dict):
        data = {}

    reason = data.get("reason", data.get("direction_reason", ""))
    return {
        **data,
        "shot_direction": data.get(
            "shot_direction",
            data.get("direction_shot_category", "Unknown"),
        ),
        "field_zone": data.get("field_zone", "Unknown"),
        "zone_confidence": data.get("zone_confidence", "Unknown"),
        "direction_angle_degrees": data.get("direction_angle_degrees"),
        "reason": reason,
        "direction_reason": reason,
    }


def _normalize_observer_timeline(result_or_timeline):
    data = result_or_timeline or {}
    if isinstance(data, dict) and "observer_timeline" in data:
        data = data.get("observer_timeline") or data
    if not isinstance(data, dict):
        data = {}
    return {
        **data,
        "total_frames": data.get("total_frames"),
        "processed_frames": data.get("processed_frames"),
        "ball_tracking_coverage": data.get("ball_tracking_coverage"),
        "bat_detection_coverage": data.get("bat_detection_coverage"),
        "stump_detection_coverage": data.get("stump_detection_coverage"),
        "missing_ball_frames": data.get("missing_ball_frames", 0),
        "low_confidence_ball_frames": data.get("low_confidence_ball_frames", 0),
        "possible_false_ball_detections": data.get("possible_false_ball_detections", 0),
        "detection_quality": data.get("detection_quality", "Unknown"),
        "observer_notes": data.get("observer_notes", ""),
    }


def _normalize_agent(result_or_agent):
    data = result_or_agent or {}
    if isinstance(data, dict) and "agent_info" in data:
        data = data.get("agent_info") or data
    if not isinstance(data, dict):
        data = {}

    return {
        **data,
        "agent_quality": data.get("agent_quality", "Unknown"),
        "agent_confidence": data.get("agent_confidence", "Unknown"),
        "ball_tracking_coverage": data.get("ball_tracking_coverage"),
        "bat_detection_coverage": data.get("bat_detection_coverage"),
        "stump_detection_coverage": data.get("stump_detection_coverage"),
        "missing_ball_frames": data.get("missing_ball_frames", 0),
        "possible_false_ball_detections": data.get("possible_false_ball_detections", 0),
        "analysis_consistency": data.get("analysis_consistency", "Unknown"),
        "review_flags": list(data.get("review_flags") or []),
        "agent_notes": data.get("agent_notes", ""),
        "review_frames_recommended": bool(data.get("review_frames_recommended", False)),
        "review_reason": data.get("review_reason", ""),
    }


def _format_percentage(value):
    if value is None or value == "":
        return "N/A"
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return _display_value(value)


def _format_created_at(value):
    if not value:
        return "Unknown time"
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(value)


def _format_seconds(value):
    try:
        return f"{float(value):.2f} sec"
    except (TypeError, ValueError):
        return "N/A"


def _format_pixel_distance(value):
    try:
        return f"{float(value):.0f} px"
    except (TypeError, ValueError):
        return "N/A"


def _format_run_estimate(value):
    if value is None or value == "":
        return "N/A"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _format_count_label(label, count):
    try:
        count_value = int(count)
    except (TypeError, ValueError):
        count_value = None

    if count_value is None or count_value <= 0:
        return None
    return f"{label} ({count_value} frames)"


def _format_detected_objects(result):
    explicit = result.get("detected_objects")
    if isinstance(explicit, str) and explicit.strip() and explicit != "N/A":
        return explicit

    object_items = [
        _format_count_label("Ball", result.get("ball_detected_frames")),
        _format_count_label("Bat", result.get("bat_detected_frames")),
        _format_count_label("Stumps", result.get("stump_detected_frames")),
    ]
    object_items = [item for item in object_items if item]

    if object_items:
        return ", ".join(object_items)

    known_detection_keys = {
        "ball_detected_frames",
        "bat_detected_frames",
        "stump_detected_frames",
    }
    if known_detection_keys.intersection(result):
        return "No tracked objects detected"
    return "N/A"


def _format_tracking_quality(result):
    quality = result.get("overall_tracking_quality")
    if quality:
        return str(quality)

    rate = result.get("ball_tracking_rate", result.get("ball_detection_rate"))
    try:
        return f"{float(rate):.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _format_bounce(result):
    for key in ("bounce_distance", "bounce_distance_m", "bounce_point_distance"):
        if result.get(key) is not None and str(result.get(key)) not in {"", "N/A"}:
            return _display_value(result.get(key))

    point = result.get("estimated_bounce_point")
    frame = result.get("estimated_bounce_frame")
    if point is None and frame is None:
        return "N/A"
    if point is None:
        return f"Frame {_display_value(frame)}"

    try:
        bx, by = point
    except (TypeError, ValueError):
        return _display_value(point)

    if frame is None:
        return f"({bx}, {by})"
    return f"Frame {frame} ({bx}, {by})"


def _format_result_summary(result):
    explicit_summary = result.get("delivery_result_summary") or result.get("result_summary")
    if explicit_summary and str(explicit_summary) not in {"", "N/A"}:
        return str(explicit_summary)

    try:
        from Backends.src.analysis.cricket_agent import generate_delivery_report

        return generate_delivery_report(result)
    except Exception:
        return _build_recommendation(result)


def _build_analysis_coach_note(result, *, impact_detected=False, bat_coverage_zero=False):
    length = _display_value((result or {}).get("estimated_length"))
    line = _display_value((result or {}).get("estimated_line"))
    notes = []

    if length not in {"", "Unknown", "N/A"}:
        notes.append(f"{length} length detected.")
    if line not in {"", "Unknown", "N/A"} and line != length:
        notes.append(f"Line read as {line}.")

    if bat_coverage_zero:
        notes.append(
            "Shot analysis unavailable because bat was not detected."
        )
    elif not impact_detected:
        notes.append(
            "Bat impact was not detected, so shot and outcome are unavailable."
        )

    if notes:
        return " ".join(notes)

    feedback = _format_coach_feedback(result)
    if feedback:
        return feedback[0]
    return "Review the detailed reports for coaching feedback on this delivery."


def _format_coach_feedback(result):
    explicit_feedback = (
        result.get("delivery_coach_feedback")
        or result.get("ai_coach_feedback")
        or result.get("coach_feedback")
    )
    if isinstance(explicit_feedback, list) and explicit_feedback:
        return [str(item) for item in explicit_feedback if str(item).strip()]
    if isinstance(explicit_feedback, str) and explicit_feedback.strip():
        return [explicit_feedback.strip()]

    try:
        from Backends.src.analysis.cricket_agent import generate_coaching_feedback

        feedback_items = generate_coaching_feedback(result)
        if feedback_items:
            return feedback_items
    except Exception:
        pass

    return [_build_recommendation(result)]


def _build_recommendation(result):
    line = result.get("estimated_line", "Unknown")
    length = result.get("estimated_length", "Unknown")
    tracking = result.get("overall_tracking_quality", "Unknown")

    if tracking in {"Poor", "Low"}:
        return "Tracking was limited. Use stronger lighting, a stable landscape camera, and keep the full pitch in frame."
    if length == "Good Length":
        return f"Strong {length.lower()} on {line.lower()} line. Repeat this area if it matches your plan."
    if length != "Unknown" and line != "Unknown":
        return f"Delivery landed {length.lower()} on the {line.lower()} line. Review the processed clip to confirm intent."
    return "Review the processed clip and confirm line, length, and shot direction before the next delivery."
