# Backends: current Streamlit implementation

This directory contains the active CricVision Streamlit UI, model adapters, calibration, tracking, video pipeline, reports, and local session storage.

It remains active until the Next.js/FastAPI/worker path reaches tested feature parity. Avoid adding new large architecture layers here; changes should support the current application or safely prepare a measured migration.

Run from the repository root:

```powershell
streamlit run main.py
```
