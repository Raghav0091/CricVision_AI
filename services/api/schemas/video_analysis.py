from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class VideoAnalysisPreparedResponse(BaseModel):
    success: bool
    analysis_id: str
    status: Literal["prepared"]
    original_filename: str
    stored_filename: str
    file_size_bytes: int
    created_at: datetime
    duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int
    codec: str | None = None
    reference_frame_index: int
    original_video_url: str
    reference_frame_url: str
    message: str
