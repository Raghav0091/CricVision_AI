from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    delivery_id: str = Field(min_length=1)


class AnalysisJob(AnalysisRequest):
    id: str
    created_at: datetime
    status: Literal["queued"] = "queued"
    message: str = "Analysis worker integration is not available yet."


class BallDetectionClipResponse(BaseModel):
    success: bool
    status: Literal[
        "processing",
        "ready",
        "failed",
        "ball_detector_missing",
        "invalid_upload",
        "upload_too_large",
        "video_processing_failed",
        "model_inference_failed",
        "video_writer_failed",
    ]
    delivery_index: int | None = None
    session_id: str | None = None
    job_id: str | None = None
    progress: int = 0
    model_path_used: str | None = None
    frame_count: int = 0
    processed_frames: int = 0
    frames_with_ball: int = 0
    best_confidence: float = 0.0
    average_confidence: float = 0.0
    processed_video_url: str | None = None
    message: str
