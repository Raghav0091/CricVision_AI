from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import analysis, calibration, deliveries, health, sessions


app = FastAPI(title="CricVision Pro API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(calibration.router)
app.include_router(sessions.router)
app.include_router(deliveries.router)
app.include_router(analysis.router)
