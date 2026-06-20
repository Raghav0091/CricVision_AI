import streamlit as st
from pathlib import Path

from Backends.src.ui.ui_components import (
    feature_card,
    info_panel,
    metric_card,
    page_header,
    section_header,
    workflow_step,
)

REVIEW_FRAMES_DIR = Path("outputs/review_frames")


def show_datasets_page():
    page_header(
        "Datasets",
        "Manage training data, review exported frames, and prepare future model improvements.",
    )

    section_header("Dataset Guidance")
    info_panel(
        "<strong>Review frames</strong> are exported from Video Analysis and Live Session when detections "
        "need manual review. Use them to improve ball and stump labels before retraining."
    )

    guidance_cols = st.columns(3)
    with guidance_cols[0]:
        feature_card(
            "Collect Frames",
            "Export low-confidence, missed-ball, and poor-tracking frames from analysis runs.",
            "📸",
        )
    with guidance_cols[1]:
        feature_card(
            "Label in Colab",
            "Label exported frames with your preferred annotation workflow in Google Colab.",
            "🏷️",
        )
    with guidance_cols[2]:
        feature_card(
            "Retrain Models",
            "Replace model weights after validation without changing app logic or paths.",
            "🧠",
        )

    section_header("Review Frames Location")
    review_exists = REVIEW_FRAMES_DIR.exists()
    review_count = 0

    if review_exists:
        review_count = len(
            [
                path
                for path in REVIEW_FRAMES_DIR.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".csv"}
            ]
        )

    metric_cols = st.columns(3)
    with metric_cols[0]:
        metric_card("Review Folder", str(REVIEW_FRAMES_DIR), "Default export location")
    with metric_cols[1]:
        metric_card("Stored Files", str(review_count), "Images and metadata CSV")
    with metric_cols[2]:
        metric_card("Status", "Ready" if review_exists else "Missing", "Folder availability")

    if not review_exists:
        st.warning("Review frames folder not found yet. Export frames from analysis to populate it.")
    else:
        st.success(f"Review frames directory detected at `{REVIEW_FRAMES_DIR}`.")

    dataset_path = Path(r"C:\Dataset")
    section_header("Local Dataset Folders")

    if not dataset_path.exists():
        st.warning(f"Dataset folder not found at `{dataset_path}`.")
        info_panel(
            "When available, local dataset folders will appear here for ball and stump training data."
        )
        return

    st.write("Detected dataset folders:")
    folder_count = 0

    for item in dataset_path.iterdir():
        if item.is_dir():
            folder_count += 1
            st.markdown(
                f"""
                <div style="background:rgba(15,23,42,0.75);border:1px solid rgba(51,65,85,0.85);
                            border-radius:12px;padding:0.75rem 1rem;margin-bottom:0.5rem;color:#CBD5E1;">
                    📂 {item.name}
                </div>
                """,
                unsafe_allow_html=True,
            )

    if folder_count == 0:
        st.info("No subfolders found inside the dataset directory.")

    section_header("Future Training Workflow")
    workflow_cols = st.columns(2)
    with workflow_cols[0]:
        workflow_step(1, "Collect review frames from difficult detections during analysis.")
        workflow_step(2, "Label ball and stump boxes with consistent class names.")
        workflow_step(3, "Split into train / validation sets.")
    with workflow_cols[1]:
        workflow_step(4, "Train YOLO models in Google Colab.")
        workflow_step(5, "Validate on held-out clips and compare tracking quality.")
        workflow_step(6, "Replace model weights in the Models folder after validation.")
