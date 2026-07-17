from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    name: str = Field(default="Live bowling session", min_length=1, max_length=120)
    source: Literal["live", "upload"] = "live"
    session_type: Literal["standard", "experimental_delivery_test"] = "standard"


class SessionDeliveryRecord(BaseModel):
    delivery_index: int = Field(ge=1, le=6)
    raw_video_url: str | None = None
    job_id: str | None = None
    analysis_status: Literal["queued", "processing", "ready", "failed"] = "queued"
    progress: int = Field(default=0, ge=0, le=100)
    processed_video_url: str | None = None
    frames_processed: int = 0
    frames_with_ball: int = 0
    best_confidence: float = 0.0
    average_confidence: float = 0.0
    model_path_used: str | None = None
    error_message: str | None = None


class SessionRecord(SessionCreate):
    id: str
    created_at: datetime
    updated_at: datetime
    status: Literal["created", "capturing", "complete"] = "created"
    capture_status: Literal["recording", "capture_complete"] = "recording"
    analysis_status: Literal["not_started", "processing", "partially_ready", "ready", "failed"] = "not_started"
    delivery_count: int = 0
    deliveries: list[SessionDeliveryRecord] = Field(default_factory=list)
    delivery_ids: list[str] = Field(default_factory=list)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
