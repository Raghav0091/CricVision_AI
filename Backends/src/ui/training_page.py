import streamlit as st


def show_training_page():
    st.title("🧠 Model Training")

    st.write("This page will later train the YOLO cricket ball detection model.")

    st.info(
        "First model to train: Cricket Ball Detector using YOLO."
    )

    if st.button("Train Ball Detector"):
        st.warning("Training code not added yet.")