import streamlit as st

from Backends.src.config.paths import DATASETS_DIR, REVIEW_FRAMES_DIR
from Backends.src.ui.components import metric_grid, model_status_card
from Backends.src.ui.theme import render_page_header, render_section_title, render_feature_card

def show_datasets_page():
    render_page_header(
        "Datasets",
        "Training data, review exports, and dataset readiness.",
    )

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

    metric_grid(
        [
            ("Review Frames", str(review_count), "Exported from analysis"),
            ("Dataset Status", "Ready" if review_exists else "Empty", "Review folder availability"),
            ("Import Ready", "Yes" if review_count else "No", "Frames available for labeling"),
        ],
        columns=3,
    )

    render_section_title("Dataset Cards")
    cards = st.columns(3)
    with cards[0]:
        render_feature_card(
            "Review Exports",
            "Low-confidence and missed detections saved during analysis for relabeling.",
            "📸",
        )
    with cards[1]:
        render_feature_card(
            "Labeling Workflow",
            "Label exported frames in Colab before retraining.",
            "🏷️",
        )
    with cards[2]:
        render_feature_card(
            "Validation Clips",
            "Hold out real bowling clips to compare tracking quality after retraining.",
            "✅",
        )

    with st.expander("Import / Upload", expanded=False):
        st.file_uploader("Upload labeled dataset archive", type=["zip"], key="dataset_upload")
        st.caption("Upload support is visual only until dataset ingestion is connected.")

    with st.expander("Advanced Paths", expanded=False):
        st.write(f"Review frames folder: `{REVIEW_FRAMES_DIR}`")
        dataset_path = DATASETS_DIR
        st.write(f"Local dataset root: `{dataset_path}`")
        if dataset_path.exists():
            folders = [item.name for item in dataset_path.iterdir() if item.is_dir()]
            if folders:
                for name in folders:
                    st.markdown(f"- `{name}`")
            else:
                st.caption("No subfolders found.")
        else:
            st.caption("Local dataset folder not found on this machine.")

    with st.expander("Model Status", expanded=False):
        model_status_card()
