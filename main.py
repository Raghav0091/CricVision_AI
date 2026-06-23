import streamlit as st

from Backends.src.ui.theme import apply_global_theme, render_sidebar

st.set_page_config(
    page_title="CricVision AI",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_global_theme()
page = render_sidebar()

if page == "Dashboard":
    from Backends.src.ui.dashboard import show_dashboard

    show_dashboard()

elif page == "Live Session":
    from Backends.src.ui.live_session import show_live_session_page

    show_live_session_page()

elif page == "Video Analysis":
    from Backends.src.ui.video_analysis import show_video_analysis_page

    show_video_analysis_page()

elif page == "Results":
    from Backends.src.ui.results_page import show_results_page

    show_results_page()
