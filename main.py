import streamlit as st

from Backends.src.ui.theme import apply_global_theme, render_sidebar

# Set True to expose Field Setup Lab, Datasets, and Training Lab in the sidebar.
SHOW_DEV_PAGES = False

st.set_page_config(
    page_title="CricVision AI",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_global_theme()
page = render_sidebar(show_dev_pages=SHOW_DEV_PAGES)

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

elif SHOW_DEV_PAGES and page == "Field Map":
    from Backends.src.ui.field_map import show_field_map_page

    show_field_map_page()

elif SHOW_DEV_PAGES and page == "Datasets":
    from Backends.src.ui.datasets_page import show_datasets_page

    show_datasets_page()

elif SHOW_DEV_PAGES and page == "Training":
    from Backends.src.ui.training_page import show_training_page

    show_training_page()
