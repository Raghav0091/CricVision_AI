"""Backward-compatible UI exports. Prefer theme.py and components.py for new code."""

from Backends.src.ui.components import (
    analysis_report_card,
    clean_upload_box,
    delivery_summary_card,
    developer_details_expander,
    hero_section,
    metric_grid,
    primary_action_card,
    video_preview_card,
)
from Backends.src.ui.theme import (
    apply_global_theme,
    render_empty_state,
    render_feature_card,
    render_metric_card,
    render_page_header,
    render_section_title,
    render_sidebar,
    render_status_pill,
)

apply_global_styles = apply_global_theme
page_header = render_page_header
section_header = render_section_title
metric_card = render_metric_card
feature_card = render_feature_card
empty_state = render_empty_state
status_badge = lambda label, tone="default": render_status_pill(
    label,
    {
        "cyan": "success",
        "green": "success",
        "blue": "default",
        "amber": "warning",
        "muted": "default",
    }.get(tone, "default"),
)


def badge_row(badges):
    import streamlit as st

    st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin:0.5rem 0 1rem 0;">{"".join(badges)}</div>', unsafe_allow_html=True)


def card(title="", content_html="", accent=False):
    import streamlit as st

    border = "rgba(16,185,129,0.28)" if accent else "rgba(148,163,184,0.14)"
    title_html = f'<div style="color:#34d399;font-weight:700;margin-bottom:0.5rem;">{title}</div>' if title else ""
    st.markdown(
        f'<div class="cv-glass" style="border-color:{border};">{title_html}{content_html}</div>',
        unsafe_allow_html=True,
    )


def info_panel(content, tone="info"):
    import streamlit as st

    color = "#fcd34d" if tone != "info" else "#cbd5e1"
    st.markdown(
        f'<div class="cv-glass" style="border-left:4px solid #10b981;color:{color};">{content}</div>',
        unsafe_allow_html=True,
    )


def workflow_step(step_number, text):
    import streamlit as st

    st.markdown(
        f"""
        <div class="cv-glass" style="padding:0.8rem 0.95rem;margin-bottom:0.55rem;">
            <div style="color:#34d399;font-size:0.75rem;font-weight:800;letter-spacing:0.06em;text-transform:uppercase;">Step {step_number}</div>
            <div style="color:#e2e8f0;margin-top:0.2rem;">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


NAV_PAGES = [
    "Dashboard",
    "Video Analysis",
    "Live Session",
    "Field Map",
    "Results",
    "Training",
    "Datasets",
]


def render_sidebar_nav():
    return render_sidebar()
