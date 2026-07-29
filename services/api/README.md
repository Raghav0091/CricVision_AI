# CricVision Pro API

This FastAPI service is the future control plane for calibration, sessions, deliveries, and analysis jobs. It currently uses process-local metadata stores and local calibration-frame storage; it is not production-ready or connected to the ML worker.

## Windows development

Use Command Prompt from the repository root:

```bat
cd /d C:\CricVision_AI
.venv\Scripts\activate
python -m uvicorn services.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Only `.venv` is required. `.venv_pose` is not used. Physics Engine V1,
Virtual Pitch V1, calibration, detection, tracking, and replay are modules
inside this single FastAPI application. Stop it with `Ctrl+C`. If port 8000 is
occupied, close the old backend terminal before starting another backend.

Health: `http://127.0.0.1:8000/health`

OpenAPI: `http://127.0.0.1:8000/docs`

Calibration never fakes success. Until a dedicated stump detector is connected, `/calibration/solve` returns `stump_detector_missing`.
