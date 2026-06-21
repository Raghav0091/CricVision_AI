import streamlit as st

from Backends.src.ui.components import metric_grid, model_status_card
from Backends.src.ui.theme import render_page_header, render_section_title, render_feature_card


def show_training_page():
    render_page_header(
        "Training Lab",
        "Train, evaluate, and deploy cricket vision models without leaving your workflow.",
    )

    render_section_title("Lab Overview")
    overview = st.columns(3)
    with overview[0]:
        render_feature_card(
            "Model Training",
            "Run YOLO and sequence-model training in Google Colab with GPU support.",
            "🧪",
        )
    with overview[1]:
        render_feature_card(
            "Model Evaluation",
            "Validate on real clips before replacing production weights.",
            "📊",
        )
    with overview[2]:
        render_feature_card(
            "Deployment",
            "Drop validated weights into Models/ without changing app logic.",
            "🚀",
        )

    metric_grid(
        [
            ("Training History", "—", "Future notebook runs"),
            ("Best Checkpoint", "—", "Latest validated model"),
            ("Dataset Status", "Review exports available", "Use Datasets page"),
        ],
        columns=3,
    )

    with st.expander("Recommended Pipeline", expanded=True):
        st.markdown(
            """
            1. Collect review frames from Analyze and Live Session  
            2. Label ball, stump, and bat frames consistently  
            3. Train in Google Colab with augmentation and early stopping  
            4. Validate on unseen clips and compare tracking metrics  
            5. Replace model weights and re-run Analyze / Live Session
            """
        )

    with st.expander("Advanced / Dangerous Actions", expanded=False):
        st.warning("Training does not run inside Streamlit Cloud. Use Colab for GPU-backed jobs.")
        if st.button("Open Colab Training Workflow"):
            st.info("Add your Colab notebook link here when ready.")

    with st.expander("Model Status", expanded=False):
        model_status_card()
