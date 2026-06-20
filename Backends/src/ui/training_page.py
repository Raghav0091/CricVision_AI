import streamlit as st

from Backends.src.ui.ui_components import (
    feature_card,
    info_panel,
    page_header,
    section_header,
    workflow_step,
)


def show_training_page():
    page_header(
        "Model Training",
        "Train and validate cricket vision models in Google Colab, then deploy updated weights to CricVision AI.",
    )

    section_header("Training Platform")
    info_panel(
        "<strong>Model training happens in Google Colab.</strong> The Streamlit app loads trained weights "
        "from the Models folder. No training runs inside this deployment to keep Streamlit Cloud compatible."
    )

    platform_cols = st.columns(3)
    with platform_cols[0]:
        feature_card(
            "Colab Notebooks",
            "Use GPU-backed notebooks for YOLO training without heavy local setup.",
            "☁️",
        )
    with platform_cols[1]:
        feature_card(
            "Validated Weights",
            "Validate on real bowling clips before replacing production model files.",
            "✅",
        )
    with platform_cols[2]:
        feature_card(
            "Drop-in Deployment",
            "Replace best.pt files in Models/ without changing detection logic or paths.",
            "🚀",
        )

    section_header("Recommended Training Pipeline")
    pipeline_cols = st.columns(2)

    with pipeline_cols[0]:
        workflow_step(1, "Collect frames from live sessions and video analysis review exports.")
        workflow_step(2, "Label ball, stump, and difficult frames with consistent annotations.")
        workflow_step(3, "Train YOLO models in Google Colab with augmentation and early stopping.")

    with pipeline_cols[1]:
        workflow_step(4, "Validate on unseen clips and compare detection / tracking metrics.")
        workflow_step(5, "Replace model weights in Models/ball_detector or Models/cricket_objects.")
        workflow_step(6, "Re-run Video Analysis and Live Session to confirm improved tracking.")

    section_header("Current Status")
    st.info("First model to train: Cricket Ball Detector using YOLO.")

    if st.button("Open Colab Training Workflow"):
        st.warning(
            "Training code is managed in Google Colab. Add your notebook link here when ready."
        )
