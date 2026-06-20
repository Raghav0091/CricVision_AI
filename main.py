import streamlit as st

from Backends.src.ui.ui_components import apply_global_styles, render_sidebar_nav

st.set_page_config(
    page_title="CricVision AI",
    page_icon="🏏",
    layout="wide",
)

apply_global_styles()

page = render_sidebar_nav()

if page == "Dashboard":
    from Backends.src.ui.dashboard import show_dashboard

    show_dashboard()

elif page == "Live Session":
    from Backends.src.ui.live_session import show_live_session_page

    show_live_session_page()

elif page == "Video Analysis":
    from Backends.src.ui.video_analysis import show_video_analysis_page

    show_video_analysis_page()

elif page == "Datasets":
    from Backends.src.ui.datasets_page import show_datasets_page

    show_datasets_page()

elif page == "Training":
    from Backends.src.ui.training_page import show_training_page

    show_training_page()

elif page == "Results":
    from Backends.src.ui.results_page import show_results_page

    show_results_page()
