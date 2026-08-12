import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routes import (
    analysis,
    calibration,
    device_calibration,
    health,
    quick_test,
    sessions,
    video_analysis,
)


def _development_cors_origins() -> list[str]:
    configured = os.environ.get("CRICVISION_WEB_ORIGIN", "").strip()
    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return ["http://localhost:3000", "http://127.0.0.1:3000"]


app = FastAPI(title="CricVision Pro API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_development_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(calibration.router)
app.include_router(sessions.router)
app.include_router(analysis.router)
app.include_router(video_analysis.router)
app.include_router(quick_test.router)
app.include_router(device_calibration.router)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
app.mount(
    "/static/delivery-clips",
    StaticFiles(
        directory=str(PROJECT_ROOT / "outputs" / "delivery_clips"),
        check_dir=False,
    ),
    name="delivery-clips-static",
)
app.mount(
    "/static/video-analysis",
    StaticFiles(
        directory=str(PROJECT_ROOT / "outputs" / "video_analysis"),
        check_dir=False,
    ),
    name="video-analysis-static",
)
