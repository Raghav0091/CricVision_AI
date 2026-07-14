# CricVision Pro worker scaffold

This directory reserves the future background-processing boundary. It is not connected to a queue, FastAPI job dispatch, or the current Streamlit analysis pipeline. Its pipeline methods return explicit unavailable/empty results and must not be treated as completed ML features.
