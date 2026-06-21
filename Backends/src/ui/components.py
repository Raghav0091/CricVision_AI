"""Page-level reusable UI blocks for CricVision AI."""

import streamlit as st

from Backends.src.ui.theme import (
    render_empty_state,
    render_feature_card,
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


def analysis_report_card(result):
    impact = result.get("impact_info", {}) or {}
    impact_frame = impact.get("impact_frame")
    recommendation = _build_recommendation(result)

    rows = [
        ("Line", result.get("estimated_line", "Unknown")),
        ("Length", result.get("estimated_length", "Unknown")),
        ("Bounce", _format_bounce(result)),
        ("Tracking Quality", result.get("overall_tracking_quality", "Unknown")),
        ("Ball Detected", "Yes" if result.get("ball_detected_frames") else "No"),
        ("Bat Detected", "Yes" if result.get("bat_detected_frames") else "No"),
        (
            "Possible Impact Frame",
            str(impact_frame) if impact_frame is not None else "Not found",
        ),
        ("Impact Confidence", impact.get("impact_confidence", "Unknown")),
    ]

    row_html = "".join(
        f"""
        <div class="cv-report-row">
            <div class="cv-report-label">{label}</div>
            <div class="cv-report-value">{value}</div>
        </div>
        """
        for label, value in rows
    )

    st.markdown(
        f"""
        <div class="cv-glass">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
                <div style="color:#f8fafc;font-size:1.05rem;font-weight:800;">Delivery Report</div>
                {render_status_pill(result.get("overall_tracking_quality", "Report"), "success" if result.get("overall_tracking_quality") in {"Excellent", "Good"} else "warning")}
            </div>
            {row_html}
            <div style="margin-top:1rem;padding-top:0.85rem;border-top:1px solid rgba(51,65,85,0.45);">
                <div style="color:#94a3b8;font-size:0.78rem;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;">Recommendation</div>
                <div style="color:#e2e8f0;margin-top:0.35rem;line-height:1.55;">{recommendation}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
            render_metric_card(
                status["name"],
                status["status"],
                hint=status["path"] if not status["found"] else "Ready for analysis",
                status="Ready" if status["found"] else "Missing",
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


def _format_bounce(result):
    if result.get("estimated_bounce_point") is None:
        return "Not found"
    bx, by = result["estimated_bounce_point"]
    return f"Frame {result.get('estimated_bounce_frame', '—')} · ({bx}, {by})"


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
