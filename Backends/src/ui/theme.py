"""Global theme and layout primitives for CricVision AI."""

import streamlit as st

NAV_ITEMS = [
    ("Dashboard", "Dashboard"),
    ("Live Session", "Live Session"),
    ("Video Analysis", "Video Analysis"),
    ("Session Results", "Results"),
]

NAV_ICONS = {
    "Dashboard": "🏠",
    "Live Session": "📹",
    "Video Analysis": "🎯",
    "Session Results": "📋",
}

PAGE_ROUTE = {label: route for label, route in NAV_ITEMS}


def apply_global_theme():
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
        }

        .stApp {
            background: radial-gradient(circle at top, #0f1a14 0%, #0a0f14 45%, #070b10 100%);
            color: #e8edf2;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1180px;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1210 0%, #0a1018 100%);
            border-right: 1px solid rgba(16, 185, 129, 0.22);
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            background: rgba(12, 18, 24, 0.85);
            border: 1px solid rgba(51, 65, 85, 0.55);
            border-radius: 12px;
            padding: 0.55rem 0.8rem;
            margin-bottom: 0.35rem;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.18), rgba(245, 158, 11, 0.08));
            border-color: rgba(16, 185, 129, 0.55);
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label p {
            color: #e8edf2 !important;
            font-weight: 600 !important;
            font-size: 0.92rem !important;
        }

        .cv-brand-title {
            color: #f8fafc;
            font-size: 1.28rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.02em;
        }

        .cv-brand-sub {
            color: #94a3b8;
            font-size: 0.82rem;
            margin: 0.15rem 0 0 0;
        }

        .cv-page-header {
            margin-bottom: 1.5rem;
        }

        .cv-page-title {
            color: #f8fafc;
            font-size: 2rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.03em;
        }

        .cv-page-subtitle {
            color: #94a3b8;
            font-size: 1rem;
            margin: 0.35rem 0 0 0;
            line-height: 1.55;
            max-width: 760px;
        }

        .cv-glass {
            background: rgba(14, 20, 28, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 18px;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.22);
            backdrop-filter: blur(10px);
        }

        .cv-section-title {
            color: #f8fafc;
            font-size: 1.05rem;
            font-weight: 700;
            margin: 1.35rem 0 0.75rem 0;
        }

        .cv-section-sub {
            color: #64748b;
            font-size: 0.88rem;
            margin: -0.45rem 0 0.85rem 0;
        }

        .cv-metric {
            background: rgba(10, 16, 22, 0.88);
            border: 1px solid rgba(16, 185, 129, 0.18);
            border-radius: 16px;
            padding: 1rem 1.05rem;
            min-height: 108px;
        }

        .cv-metric-label {
            color: #94a3b8;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .cv-metric-value {
            color: #34d399;
            font-size: 1.65rem;
            font-weight: 800;
            margin: 0.25rem 0;
        }

        .cv-metric-hint {
            color: #64748b;
            font-size: 0.84rem;
            line-height: 1.35;
        }

        .cv-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.28rem 0.72rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            border: 1px solid transparent;
        }

        .cv-pill-default {
            background: rgba(51, 65, 85, 0.45);
            color: #cbd5e1;
            border-color: rgba(100, 116, 139, 0.35);
        }

        .cv-pill-success {
            background: rgba(16, 185, 129, 0.14);
            color: #6ee7b7;
            border-color: rgba(16, 185, 129, 0.35);
        }

        .cv-pill-warning {
            background: rgba(245, 158, 11, 0.14);
            color: #fcd34d;
            border-color: rgba(245, 158, 11, 0.35);
        }

        .cv-pill-error {
            background: rgba(239, 68, 68, 0.12);
            color: #fca5a5;
            border-color: rgba(239, 68, 68, 0.28);
        }

        .cv-pill-gold {
            background: rgba(245, 158, 11, 0.12);
            color: #fbbf24;
            border-color: rgba(245, 158, 11, 0.28);
        }

        .cv-empty {
            text-align: center;
            padding: 2.4rem 1.5rem;
            border: 1px dashed rgba(100, 116, 139, 0.45);
            border-radius: 18px;
            color: #94a3b8;
            background: rgba(10, 16, 22, 0.55);
        }

        .cv-empty h3 {
            color: #e2e8f0;
            margin-bottom: 0.45rem;
        }

        .cv-report-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.55rem 0;
            border-bottom: 1px solid rgba(51, 65, 85, 0.45);
        }

        .cv-report-label {
            color: #94a3b8;
            font-size: 0.88rem;
        }

        .cv-report-value {
            color: #f8fafc;
            font-size: 0.92rem;
            font-weight: 700;
            text-align: right;
        }

        .cv-hero {
            background: linear-gradient(135deg, rgba(6, 78, 59, 0.28), rgba(10, 16, 24, 0.92));
            border: 1px solid rgba(16, 185, 129, 0.24);
            border-radius: 22px;
            padding: 2rem 2.1rem;
            margin-bottom: 1.25rem;
        }

        .cv-hero h1 {
            color: #f8fafc;
            font-size: 2.35rem;
            margin: 0 0 0.35rem 0;
        }

        .cv-hero p {
            color: #cbd5e1;
            margin: 0;
            line-height: 1.6;
            max-width: 720px;
        }

        .stButton > button {
            background: linear-gradient(135deg, #059669, #047857) !important;
            color: #fff !important;
            border: 1px solid rgba(16, 185, 129, 0.45) !important;
            border-radius: 14px !important;
            font-weight: 700 !important;
            min-height: 2.75rem;
        }

        .stButton > button:hover {
            border-color: rgba(251, 191, 36, 0.55) !important;
            box-shadow: 0 10px 24px rgba(16, 185, 129, 0.18) !important;
        }

        div[data-testid="stFileUploader"] section {
            background: rgba(10, 16, 22, 0.75);
            border: 1px dashed rgba(16, 185, 129, 0.35);
            border-radius: 16px;
        }

        div[data-baseweb="tab-list"] button[aria-selected="true"] {
            color: #34d399 !important;
            border-bottom: 2px solid #10b981 !important;
        }

        .stExpander {
            border: 1px solid rgba(51, 65, 85, 0.55) !important;
            border-radius: 14px !important;
            background: rgba(10, 16, 22, 0.55) !important;
        }

        div[data-testid="stMetricValue"] { color: #34d399 !important; }
        div[data-testid="stMetricLabel"] { color: #94a3b8 !important; }

        .stInfo, .stSuccess, .stWarning, .stError {
            border-radius: 12px !important;
        }

        @media (max-width: 768px) {
            .cv-page-title { font-size: 1.55rem; }
            .cv-hero h1 { font-size: 1.75rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    st.sidebar.markdown(
        """
        <div style="padding:0.2rem 0 1rem 0;border-bottom:1px solid rgba(51,65,85,0.55);margin-bottom:1rem;">
            <p class="cv-brand-title">🏏 CricVision AI</p>
            <p class="cv-brand-sub">AI-powered cricket performance analysis</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    labels = [f"{NAV_ICONS[label]}  {label}" for label, _ in NAV_ITEMS]
    routes = [route for _, route in NAV_ITEMS]

    if "app_page" not in st.session_state:
        st.session_state.app_page = routes[0]

    nav_target = st.session_state.pop("nav_target", None)
    if nav_target in routes:
        st.session_state.app_page = nav_target

    default_index = routes.index(st.session_state.app_page) if st.session_state.app_page in routes else 0
    selected_label = st.sidebar.radio("Navigation", labels, index=default_index, label_visibility="collapsed")
    selected_nav = selected_label.split("  ", 1)[-1]
    route = PAGE_ROUTE[selected_nav]
    st.session_state.app_page = route

    return route


def render_model_status_sidebar():
    from Backends.src.models.model_registry import validate_model_paths

    for status in validate_model_paths().values():
        pill_class = "cv-pill-success" if status["found"] else "cv-pill-warning"
        st.markdown(
            f'<div style="margin-bottom:0.45rem;">'
            f'<span class="cv-pill {pill_class}">{status["name"]}: {status["status"]}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )


def render_page_header(title, subtitle=None, badge=None):
    badge_html = ""
    if badge:
        badge_html = f'<div style="margin-top:0.65rem;">{render_status_pill(badge, "success")}</div>'
    subtitle_html = f'<p class="cv-page-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="cv-page-header">
            <h1 class="cv-page-title">{title}</h1>
            {subtitle_html}
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(title, subtitle=None):
    subtitle_html = f'<div class="cv-section-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="cv-section-title">{title}</div>{subtitle_html}',
        unsafe_allow_html=True,
    )


def render_metric_card(label, value, hint=None, status=None):
    status_html = ""
    if status:
        tone = "success" if status.lower() in {"ready", "good", "found", "yes"} else "warning"
        status_html = f'<div style="margin-top:0.45rem;">{render_status_pill(status, tone)}</div>'
    hint_html = f'<div class="cv-metric-hint">{hint}</div>' if hint else ""
    st.markdown(
        f"""
        <div class="cv-metric">
            <div class="cv-metric-label">{label}</div>
            <div class="cv-metric-value">{value}</div>
            {hint_html}
            {status_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_card(title, description, icon=None):
    heading = f"{icon} {title}" if icon else title
    st.markdown(
        f"""
        <div class="cv-glass" style="min-height:130px;">
            <div style="color:#34d399;font-weight:700;margin-bottom:0.45rem;">{heading}</div>
            <div style="color:#cbd5e1;line-height:1.5;font-size:0.94rem;">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(title, message, action_label=None):
    action_html = f'<div style="margin-top:0.85rem;color:#34d399;font-weight:700;">{action_label}</div>' if action_label else ""
    st.markdown(
        f"""
        <div class="cv-empty">
            <h3>{title}</h3>
            <p style="margin:0;">{message}</p>
            {action_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_pill(text, status="default"):
    css = {
        "default": "cv-pill-default",
        "success": "cv-pill-success",
        "warning": "cv-pill-warning",
        "error": "cv-pill-error",
        "gold": "cv-pill-gold",
    }.get(status, "cv-pill-default")
    return f'<span class="cv-pill {css}">{text}</span>'
