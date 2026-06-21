"""Reusable UI helpers for CricVision AI Streamlit pages."""

import streamlit as st


NAV_PAGES = [
    "Dashboard",
    "Live Session",
    "Video Analysis",
    "Field Map",
    "Results",
]

NAV_ICONS = {
    "Dashboard": "📊",
    "Live Session": "📹",
    "Video Analysis": "🎥",
    "Field Map": "🗺️",
    "Datasets": "📁",
    "Training": "🧠",
    "Results": "📈",
}


def apply_global_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(160deg, #070B12 0%, #0B1220 45%, #0F172A 100%);
            color: #E2E8F0;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0A1628 0%, #111827 100%);
            border-right: 1px solid rgba(34, 211, 238, 0.35);
        }

        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li {
            color: #CBD5E1;
        }

        section[data-testid="stSidebar"] .stRadio > label {
            color: #94A3B8 !important;
            font-size: 0.75rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid rgba(51, 65, 85, 0.8);
            border-radius: 12px;
            padding: 0.55rem 0.75rem;
            margin-bottom: 0.35rem;
            transition: all 0.2s ease;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            border-color: rgba(34, 211, 238, 0.55);
            background: rgba(14, 165, 233, 0.12);
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.22) 0%, rgba(16, 185, 129, 0.14) 100%);
            border-color: #22D3EE;
            box-shadow: 0 0 0 1px rgba(34, 211, 238, 0.25);
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label p {
            color: #E2E8F0 !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }

        h1, h2, h3, h4 {
            color: #F8FAFC;
            font-weight: 700;
        }

        .cv-page-header {
            margin-bottom: 1.25rem;
        }

        .cv-page-title {
            color: #22D3EE;
            font-size: 2rem;
            font-weight: 800;
            margin: 0 0 0.35rem 0;
            line-height: 1.2;
        }

        .cv-page-subtitle {
            color: #94A3B8;
            font-size: 1.05rem;
            margin: 0;
            line-height: 1.5;
        }

        .cv-card {
            background: linear-gradient(145deg, rgba(15, 23, 42, 0.95) 0%, rgba(17, 24, 39, 0.92) 100%);
            border: 1px solid rgba(51, 65, 85, 0.85);
            border-radius: 18px;
            padding: 1.35rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 30px rgba(2, 6, 23, 0.35);
        }

        .cv-card-accent {
            border-color: rgba(34, 211, 238, 0.45);
            box-shadow: 0 12px 32px rgba(14, 165, 233, 0.12);
        }

        .cv-card-title {
            color: #67E8F9;
            font-size: 1.05rem;
            font-weight: 700;
            margin: 0 0 0.75rem 0;
        }

        .cv-metric-card {
            background: linear-gradient(145deg, rgba(15, 23, 42, 0.98) 0%, rgba(30, 41, 59, 0.92) 100%);
            border: 1px solid rgba(34, 211, 238, 0.35);
            border-radius: 16px;
            padding: 1.1rem 1.2rem;
            min-height: 118px;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
        }

        .cv-metric-title {
            color: #94A3B8;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .cv-metric-value {
            color: #22D3EE;
            font-size: 1.85rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.25rem;
        }

        .cv-metric-note {
            color: #64748B;
            font-size: 0.88rem;
            line-height: 1.35;
        }

        .cv-feature-card {
            background: linear-gradient(145deg, rgba(15, 23, 42, 0.96) 0%, rgba(30, 58, 95, 0.55) 100%);
            border: 1px solid rgba(34, 211, 238, 0.28);
            border-radius: 16px;
            padding: 1.15rem 1.2rem;
            min-height: 150px;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        .cv-feature-card:hover {
            transform: translateY(-2px);
            border-color: rgba(52, 211, 153, 0.55);
        }

        .cv-feature-card h3 {
            color: #22D3EE;
            font-size: 1rem;
            margin: 0 0 0.5rem 0;
        }

        .cv-feature-card p {
            color: #CBD5E1;
            font-size: 0.95rem;
            margin: 0;
            line-height: 1.45;
        }

        .cv-hero {
            background: linear-gradient(135deg, rgba(8, 47, 73, 0.95) 0%, rgba(15, 23, 42, 0.98) 55%, rgba(6, 78, 59, 0.35) 100%);
            border: 1px solid rgba(34, 211, 238, 0.45);
            border-radius: 22px;
            padding: 2rem 2.2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 18px 45px rgba(2, 6, 23, 0.45);
        }

        .cv-hero h1 {
            color: #67E8F9;
            font-size: 2.4rem !important;
            margin: 0 0 0.5rem 0;
        }

        .cv-hero h3 {
            color: #A5F3FC;
            font-size: 1.15rem;
            margin: 0 0 0.75rem 0;
        }

        .cv-hero p {
            color: #CBD5E1 !important;
            font-size: 1.05rem;
            margin: 0;
            line-height: 1.6;
            max-width: 920px;
        }

        .cv-section-header {
            color: #E2E8F0;
            font-size: 1.25rem;
            font-weight: 800;
            margin: 1.5rem 0 0.85rem 0;
            padding-bottom: 0.45rem;
            border-bottom: 1px solid rgba(51, 65, 85, 0.9);
        }

        .cv-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0.5rem 0 1rem 0;
        }

        .cv-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            border: 1px solid transparent;
        }

        .cv-badge-cyan {
            background: rgba(14, 165, 233, 0.16);
            color: #7DD3FC;
            border-color: rgba(34, 211, 238, 0.35);
        }

        .cv-badge-green {
            background: rgba(16, 185, 129, 0.16);
            color: #6EE7B7;
            border-color: rgba(52, 211, 153, 0.35);
        }

        .cv-badge-blue {
            background: rgba(59, 130, 246, 0.16);
            color: #93C5FD;
            border-color: rgba(96, 165, 250, 0.35);
        }

        .cv-badge-amber {
            background: rgba(245, 158, 11, 0.16);
            color: #FCD34D;
            border-color: rgba(251, 191, 36, 0.35);
        }

        .cv-badge-muted {
            background: rgba(51, 65, 85, 0.45);
            color: #CBD5E1;
            border-color: rgba(100, 116, 139, 0.45);
        }

        .cv-info-panel {
            background: rgba(14, 116, 144, 0.14);
            border: 1px solid rgba(34, 211, 238, 0.28);
            border-left: 4px solid #22D3EE;
            border-radius: 14px;
            padding: 0.95rem 1.1rem;
            color: #CBD5E1;
            line-height: 1.55;
            margin: 0.75rem 0 1rem 0;
        }

        .cv-info-panel strong {
            color: #E2E8F0;
        }

        .cv-warning-panel {
            background: rgba(120, 53, 15, 0.22);
            border: 1px solid rgba(251, 191, 36, 0.35);
            border-left: 4px solid #F59E0B;
            border-radius: 14px;
            padding: 0.95rem 1.1rem;
            color: #FDE68A;
            line-height: 1.55;
            margin: 0.75rem 0 1rem 0;
        }

        .cv-workflow-step {
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(51, 65, 85, 0.85);
            border-radius: 14px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.65rem;
        }

        .cv-workflow-step-num {
            color: #22D3EE;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .cv-workflow-step-text {
            color: #E2E8F0;
            font-size: 0.98rem;
            margin-top: 0.2rem;
        }

        .cv-empty-state {
            text-align: center;
            background: rgba(15, 23, 42, 0.65);
            border: 1px dashed rgba(100, 116, 139, 0.65);
            border-radius: 18px;
            padding: 2.2rem 1.5rem;
            color: #94A3B8;
        }

        .cv-empty-state h3 {
            color: #CBD5E1;
            margin-bottom: 0.5rem;
        }

        .cv-sidebar-brand {
            padding: 0.35rem 0 1rem 0;
            border-bottom: 1px solid rgba(51, 65, 85, 0.75);
            margin-bottom: 1rem;
        }

        .cv-sidebar-brand-title {
            color: #67E8F9;
            font-size: 1.35rem;
            font-weight: 800;
            margin: 0;
        }

        .cv-sidebar-brand-subtitle {
            color: #94A3B8;
            font-size: 0.88rem;
            margin: 0.15rem 0 0 0;
        }

        .stButton > button {
            background: linear-gradient(135deg, #0284C7 0%, #0891B2 100%) !important;
            color: #FFFFFF !important;
            border: 1px solid rgba(34, 211, 238, 0.45) !important;
            font-weight: 700 !important;
            border-radius: 12px !important;
            box-shadow: 0 6px 18px rgba(14, 165, 233, 0.25) !important;
        }

        .stButton > button:hover {
            border-color: #67E8F9 !important;
            box-shadow: 0 8px 22px rgba(34, 211, 238, 0.28) !important;
        }

        div[data-baseweb="tab-list"] button {
            background: rgba(15, 23, 42, 0.75) !important;
            color: #CBD5E1 !important;
            border-radius: 10px 10px 0 0 !important;
        }

        div[data-baseweb="tab-list"] button[aria-selected="true"] {
            background: rgba(14, 165, 233, 0.18) !important;
            color: #67E8F9 !important;
            border-bottom: 2px solid #22D3EE !important;
        }

        .stInfo {
            background: rgba(14, 116, 144, 0.18) !important;
            color: #CBD5E1 !important;
            border-left: 4px solid #22D3EE !important;
        }

        .stWarning {
            background: rgba(120, 53, 15, 0.22) !important;
            color: #FDE68A !important;
            border-left: 4px solid #F59E0B !important;
        }

        .stSuccess {
            background: rgba(6, 78, 59, 0.28) !important;
            color: #A7F3D0 !important;
            border-left: 4px solid #10B981 !important;
        }

        .stError {
            background: rgba(127, 29, 29, 0.28) !important;
            color: #FECACA !important;
            border-left: 4px solid #EF4444 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #67E8F9 !important;
        }

        div[data-testid="stMetricLabel"] {
            color: #94A3B8 !important;
        }

        div[data-testid="stMetricDelta"] {
            color: #6EE7B7 !important;
        }

        input, textarea, select,
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            background-color: #111827 !important;
            color: #F8FAFC !important;
            border-color: #334155 !important;
        }

        div[role="radiogroup"] label p {
            color: #E2E8F0 !important;
        }

        @media (max-width: 768px) {
            .cv-hero h1 {
                font-size: 1.75rem !important;
            }

            .cv-page-title {
                font-size: 1.55rem;
            }

            .cv-metric-value {
                font-size: 1.45rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_branding():
    st.sidebar.markdown(
        """
        <div class="cv-sidebar-brand">
            <p class="cv-sidebar-brand-title">🏏 CricVision AI</p>
            <p class="cv-sidebar-brand-subtitle">Cricket Vision Analytics</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_nav(current_page=None):
    sidebar_branding()
    labels = [f"{NAV_ICONS.get(name, name)}  {name}" for name in NAV_PAGES]
    label_to_page = dict(zip(labels, NAV_PAGES))

    selected_label = st.sidebar.radio(
        "Navigation",
        labels,
        label_visibility="collapsed",
    )

    selected_page = label_to_page[selected_label]

    if current_page is not None:
        st.sidebar.markdown(
            f"""
            <div style="margin-top:1rem;padding:0.65rem 0.75rem;border-radius:12px;
                        background:rgba(14,165,233,0.12);border:1px solid rgba(34,211,238,0.25);">
                <div style="color:#64748B;font-size:0.72rem;font-weight:700;
                            letter-spacing:0.08em;text-transform:uppercase;">Active Page</div>
                <div style="color:#67E8F9;font-weight:700;margin-top:0.15rem;">
                    {NAV_ICONS.get(selected_page, '')} {selected_page}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.sidebar.markdown(
        """
        <div style="margin-top:1.5rem;padding-top:1rem;border-top:1px solid rgba(51,65,85,0.75);">
            <p style="color:#64748B;font-size:0.78rem;margin:0;line-height:1.5;">
                AI-powered ball tracking, line & length analytics, and delivery reports.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return selected_page


def page_header(title, subtitle=""):
    subtitle_html = f'<p class="cv-page-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="cv-page-header">
            <h1 class="cv-page-title">{title}</h1>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title):
    st.markdown(f'<div class="cv-section-header">{title}</div>', unsafe_allow_html=True)


def card(title="", content_html="", accent=False):
    accent_class = " cv-card-accent" if accent else ""
    title_html = f'<div class="cv-card-title">{title}</div>' if title else ""
    st.markdown(
        f"""
        <div class="cv-card{accent_class}">
            {title_html}
            {content_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(title, value, note=""):
    st.markdown(
        f"""
        <div class="cv-metric-card">
            <div class="cv-metric-title">{title}</div>
            <div class="cv-metric-value">{value}</div>
            <div class="cv-metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_card(title, description, icon=""):
    display_title = f"{icon} {title}" if icon else title
    st.markdown(
        f"""
        <div class="cv-feature-card">
            <h3>{display_title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero_section(title, subtitle, description):
    st.markdown(
        f"""
        <div class="cv-hero">
            <h1>{title}</h1>
            <h3>{subtitle}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(label, tone="cyan"):
    tone_class = {
        "cyan": "cv-badge-cyan",
        "green": "cv-badge-green",
        "blue": "cv-badge-blue",
        "amber": "cv-badge-amber",
        "muted": "cv-badge-muted",
    }.get(tone, "cv-badge-cyan")
    return f'<span class="cv-badge {tone_class}">{label}</span>'


def badge_row(badges):
    badge_html = "".join(badges)
    st.markdown(
        f'<div class="cv-badge-row">{badge_html}</div>',
        unsafe_allow_html=True,
    )


def info_panel(content, tone="info"):
    css_class = "cv-info-panel" if tone == "info" else "cv-warning-panel"
    st.markdown(f'<div class="{css_class}">{content}</div>', unsafe_allow_html=True)


def workflow_step(step_number, text):
    st.markdown(
        f"""
        <div class="cv-workflow-step">
            <div class="cv-workflow-step-num">Step {step_number}</div>
            <div class="cv-workflow-step-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(title, message, icon="📭"):
    st.markdown(
        f"""
        <div class="cv-empty-state">
            <div style="font-size:2rem;margin-bottom:0.5rem;">{icon}</div>
            <h3>{title}</h3>
            <p style="margin:0;">{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_row(metrics, columns=4):
    cols = st.columns(columns)
    for index, (title, value, note) in enumerate(metrics):
        with cols[index % columns]:
            metric_card(title, value, note)
