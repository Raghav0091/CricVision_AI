"""Page-level reusable UI blocks for CricVision AI."""

import streamlit as st

from Backends.src.ui.theme import (
    render_empty_state,
    render_metric_card,
    render_section_title,
    render_status_pill,
)


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


def render_save_status(result, context_label="Analysis"):
    """Render save/session status without exposing raw debug output."""
    result = result or {}
    st.subheader("Save Status")

    status_lines = []
    if result.get("report_path"):
        status_lines.append(f"Report saved: `{result['report_path']}`")
    if result.get("output_path"):
        status_lines.append(f"Processed video saved: `{result['output_path']}`")
    if result.get("processed_file_name"):
        status_lines.append(f"Processed clip ready: `{result['processed_file_name']}`")

    if status_lines:
        st.success(f"{context_label} result is ready.")
        for line in status_lines:
            st.markdown(f"- {line}")
        return

    if result.get("processed_video_bytes"):
        st.info("Session result is ready. Download the processed clip to save a local copy.")
    elif result.get("success"):
        st.info(f"{context_label} result is ready for this session.")
    else:
        st.warning(f"{context_label} result is not available yet.")


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


def developer_details_expander(result):
    with st.expander("Developer Details", expanded=False):
        st.json(result)


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


def _format_count_label(label, count):
    try:
        count_value = int(count)
    except (TypeError, ValueError):
        count_value = None

    if count_value is None or count_value <= 0:
        return None
    return f"{label} ({count_value} frames)"


def _format_detected_objects(result):
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
        if result.get(key) is not None:
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
    explicit_summary = result.get("result_summary")
    if explicit_summary:
        return str(explicit_summary)

    try:
        from Backends.src.analysis.cricket_agent import generate_delivery_report

        return generate_delivery_report(result)
    except Exception:
        return _build_recommendation(result)


def _format_coach_feedback(result):
    explicit_feedback = result.get("ai_coach_feedback") or result.get("coach_feedback")
    if isinstance(explicit_feedback, str) and explicit_feedback.strip():
        return [explicit_feedback.strip()]
    if isinstance(explicit_feedback, list) and explicit_feedback:
        return [str(item) for item in explicit_feedback if str(item).strip()]

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
