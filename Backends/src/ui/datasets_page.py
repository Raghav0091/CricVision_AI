import streamlit as st
from pathlib import Path


def show_datasets_page():
    st.title("📁 Dataset Overview")

    dataset_path = Path("C:\Dataset")

    if not dataset_path.exists():
        st.warning("Dataset folder not found.")
        return

    st.write("Detected dataset folders:")

    for item in dataset_path.iterdir():
        if item.is_dir():
            st.write(f"📂 {item.name}")