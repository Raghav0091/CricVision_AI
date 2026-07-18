from datetime import datetime
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictGeometryModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)


class NormalizedPoint(StrictGeometryModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class NormalizedBox(StrictGeometryModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> "NormalizedBox":
        values = (self.x, self.y, self.width, self.height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Wicket box coordinates must be finite.")
        if self.x + self.width > 1.000001 or self.y + self.height > 1.000001:
            raise ValueError("Wicket box must stay inside the reference image.")
        return self


class WicketCandidate(StrictGeometryModel):
    candidate_id: str
    confidence: float = Field(ge=0, le=1)
    class_name: str
    box: NormalizedBox
    center: NormalizedPoint
    bottom_center: NormalizedPoint


class WicketCalibration(StrictGeometryModel):
    label: Literal["striker", "non_striker"]
    source: Literal["detected", "adjusted", "manual"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    box: NormalizedBox
    center: NormalizedPoint
    bottom_center: NormalizedPoint


class WicketCalibrationInput(StrictGeometryModel):
    label: Literal["striker", "non_striker"]
    source: Literal["detected", "adjusted", "manual"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    box: NormalizedBox


class PitchGeometry(StrictGeometryModel):
    axis_start: NormalizedPoint
    axis_end: NormalizedPoint
    corridor: list[NormalizedPoint] = Field(min_length=4, max_length=4)
    near_end_label: Literal["striker", "non_striker"]
    far_end_label: Literal["striker", "non_striker"]
    geometry_type: Literal["approximate_2d"]
    corridor_width_multiplier: float = Field(ge=0.7, le=1.5)


class VideoAnalysisPreparedResponse(BaseModel):
    success: bool
    analysis_id: str
    status: Literal["prepared", "calibrated"]
    original_filename: str
    stored_filename: str
    file_size_bytes: int
    created_at: datetime
    updated_at: datetime | None = None
    duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int
    codec: str | None = None
    reference_frame_index: int
    original_video_url: str
    reference_frame_url: str
    calibration_status: Literal["confirmed"] | None = None
    calibration_url: str | None = None
    calibration_overlay_url: str | None = None
    message: str


class VideoCalibrationDetectionResponse(BaseModel):
    success: bool
    status: Literal[
        "candidates_ready",
        "manual_required",
        "stump_detector_missing",
        "stump_detector_error",
    ]
    analysis_id: str
    reference_frame_url: str
    image_width: int
    image_height: int
    candidates: list[WicketCandidate] = Field(default_factory=list)
    provisional_striker_wicket: WicketCalibration | None = None
    provisional_non_striker_wicket: WicketCalibration | None = None
    pitch_geometry: PitchGeometry | None = None
    model_path_used: str
    warning: str | None = None
    message: str


class VideoCalibrationConfirmationRequest(BaseModel):
    analysis_id: str
    striker_wicket: WicketCalibrationInput
    non_striker_wicket: WicketCalibrationInput
    corridor_width_multiplier: float = Field(default=1.0, ge=0.7, le=1.5)
    user_note: str | None = Field(default=None, max_length=1000)


class ConfirmedVideoCalibrationResponse(BaseModel):
    success: bool
    status: Literal["calibrated"]
    analysis_id: str
    created_at: datetime
    updated_at: datetime
    reference_frame_index: int
    reference_frame_url: str
    calibration_url: str
    calibration_overlay_url: str
    image_width: int
    image_height: int
    model_path_used: str | None = None
    striker_wicket: WicketCalibration
    non_striker_wicket: WicketCalibration
    pitch_geometry: PitchGeometry
    user_note: str | None = None
    message: str
