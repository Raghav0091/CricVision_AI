from typing import Any, Literal

from pydantic import BaseModel, Field


class AlignmentBox(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)


class CalibrationRequest(BaseModel):
    frame_data_url: str
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    box_layout: dict[str, AlignmentBox]


class CalibrationResponse(BaseModel):
    success: bool
    quality: Literal["Unavailable", "Poor", "Partial", "Good"]
    reason: str
    message: str
    calibration_frame_path: str | None = None
    environment_context: dict[str, Any] | None = None
