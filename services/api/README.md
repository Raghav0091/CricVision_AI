# CricVision Pro API

This FastAPI service is the future control plane for calibration, sessions, deliveries, and analysis jobs. It currently uses process-local metadata stores and local calibration-frame storage; it is not production-ready or connected to the ML worker.

From the repository root:

```powershell
uv pip install -r services\api\requirements.txt
python -m uvicorn services.api.main:app --reload
```

Health: `http://localhost:8000/health`

OpenAPI: `http://localhost:8000/docs`

Calibration never fakes success. Until a dedicated stump detector is connected, `/calibration/solve` returns `stump_detector_missing`.
