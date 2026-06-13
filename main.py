import streamlit as st



st.set_page_config(
    page_title="CricVision AI",
    page_icon="🏏",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0B0F19 0%, #0F1419 100%);
    color: #F1F5F9;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D2847 0%, #1a1f2e 100%);
    border-right: 2px solid #00D9FF;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #00D9FF;
    font-weight: 700;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #E0F2FE;
}

section[data-testid="stSidebar"] .stRadio > label {
    color: #E0F2FE !important;
    font-weight: 600;
}

section[data-testid="stSidebar"] .stRadio > div > label > div {
    color: #E0F2FE !important;
}

h1, h2, h3 {
    color: #FFFFFF;
    font-weight: 700;
}

h1 { color: #00D9FF; }
h2 { color: #00F5FF; }

.metric-card {
    background: linear-gradient(135deg, #1E3A5F 0%, #2D5A7B 100%);
    padding: 22px;
    border-radius: 18px;
    border: 2px solid #00D9FF;
    box-shadow: 0 8px 30px rgba(0, 217, 255, 0.15);
    transition: all 0.3s ease;
}

.metric-card:hover {
    border-color: #00F5FF;
    box-shadow: 0 12px 40px rgba(0, 245, 255, 0.25);
    transform: translateY(-2px);
}

.metric-title {
    color: #7DD3FC;
    font-size: 15px;
    font-weight: 600;
}

.metric-value {
    color: #00F5FF;
    font-size: 34px;
    font-weight: 900;
}

.hero-card {
    background: linear-gradient(135deg, #0D47A1 0%, #1565C0 50%, #0D3A66 100%);
    padding: 32px;
    border-radius: 24px;
    border: 2px solid #00D9FF;
    box-shadow: 0 12px 40px rgba(0, 217, 255, 0.2);
}

.hero-card h1 {
    color: #00F5FF;
    font-size: 48px !important;
}

.hero-card h3 {
    color: #7DD3FC;
}

.hero-card p {
    color: #E0F2FE !important;
}

.feature-card {
    background: linear-gradient(135deg, #1E3A5F 0%, #16497D 100%);
    padding: 20px;
    border-radius: 18px;
    border: 2px solid #00D9FF;
    box-shadow: 0 8px 25px rgba(0, 217, 255, 0.1);
    transition: all 0.3s ease;
}   

.feature-card:hover {
    border-color: #00F5FF;
    box-shadow: 0 12px 35px rgba(0, 245, 255, 0.2);
    transform: translateY(-3px);
}

.feature-card h3 {
    color: #00D9FF;
}

.feature-card p {
    color: #E0F2FE;
}

button,
.stButton > button {
    background: linear-gradient(135deg, #0EA5E9 0%, #06B6D4 100%) !important;
    color: #FFFFFF !important;
    border: 2px solid #0EA5E9 !important;
    font-weight: 700 !important;
    padding: 10px 24px !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3) !important;
}

button:hover,
.stButton > button:hover {
    background: linear-gradient(135deg, #06B6D4 0%, #0EA5E9 100%) !important;
    border-color: #00F5FF !important;
    box-shadow: 0 6px 20px rgba(0, 245, 255, 0.4) !important;
    transform: translateY(-2px) !important;
}

.stInfo {
    background: linear-gradient(135deg, #1E5A7A 0%, #0D47A1 100%) !important;
    border-left: 4px solid #00D9FF !important;
    color: #E0F2FE !important;
}

.stWarning {
    background: linear-gradient(135deg, #7A4A1E 0%, #A16207 100%) !important;
    border-left: 4px solid #FFA500 !important;
    color: #FFF7ED !important;
}

.stSuccess {
    background: linear-gradient(135deg, #1E5A3A 0%, #047857 100%) !important;
    border-left: 4px solid #10B981 !important;
    color: #D1FAE5 !important;
}

.stError {
    background: linear-gradient(135deg, #5A1E1E 0%, #991B1B 100%) !important;
    border-left: 4px solid #EF4444 !important;
    color: #FEE2E2 !important;
}

.stSubheader {
    color: #00D9FF !important;
    font-weight: 700 !important;
}

input, textarea, select {
    background: #1F2937 !important;
    color: #F1F5F9 !important;
    border: 2px solid #334155 !important;
    border-radius: 12px !important;
}

input:focus, textarea:focus, select:focus {
    border-color: #00D9FF !important;
    box-shadow: 0 0 0 3px rgba(0, 217, 255, 0.1) !important;
}

div[role="radiogroup"] label {
    color: #E0F2FE !important;
    font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🏏 CricVision AI")
st.sidebar.caption("🎯 Live cricket analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Live Session",
        "Video Analysis",
        "Datasets",
        "Training",
        "Results",
    ],
)

if page == "Dashboard":
    from Backends.src.ui.dashboard import show_dashboard
    show_dashboard()

elif page == "Live Session":
    st.warning("Live camera session is disabled on Streamlit Cloud for now.")
    st.info("Use Video Analysis to upload cricket clips from your phone.")


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