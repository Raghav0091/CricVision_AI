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
    ball_detection_status: Literal[
        "detection_queued",
        "detecting_ball",
        "detection_complete",
        "detection_failed",
    ] | None = None
    ball_detection_job_id: str | None = None
    ball_detection_started_at: datetime | None = None
    ball_detection_completed_at: datetime | None = None
    detection_summary_url: str | None = None
    detection_overlay_url: str | None = None
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


VideoBallDetectionJobStatus = Literal[
    "queued",
    "loading_model",
    "processing",
    "writing_video",
    "saving_results",
    "ready",
    "failed",
    "ball_detector_missing",
]


class VideoBallDetectionResultLinks(BaseModel):
    processed_video_url: str
    detections_json_url: str
    detections_csv_url: str
    detection_summary_url: str


class VideoBallDetectionStartResponse(BaseModel):
    success: bool
    status: Literal["queued"]
    analysis_id: str
    job_id: str
    progress: int = Field(ge=0, le=100)
    current_frame: int = Field(ge=0)
    total_frames: int = Field(gt=0)
    message: str


class VideoBallDetectionJobResponse(BaseModel):
    success: bool
    status: VideoBallDetectionJobStatus
    analysis_id: str
    job_id: str
    progress: int = Field(ge=0, le=100)
    current_frame: int = Field(ge=0)
    total_frames: int = Field(gt=0)
    created_at: datetime
    updated_at: datetime
    model_path_used: str | None = None
    error_message: str | None = None
    result: VideoBallDetectionResultLinks | None = None
    message: str


class PixelPoint(StrictGeometryModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)


class BallCandidate(StrictGeometryModel):
    candidate_id: str
    class_id: int
    class_name: str
    confidence: float = Field(ge=0, le=1)
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4)
    bbox_normalized: NormalizedBox
    center: PixelPoint
    center_normalized: NormalizedPoint
    width_pixels: float = Field(gt=0)
    height_pixels: float = Field(gt=0)
    area_pixels: float = Field(gt=0)
    inside_pitch_corridor: bool | None = None


class FrameDetectionRecord(BaseModel):
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    processed: Literal[True]
    detections: list[BallCandidate] = Field(default_factory=list)


class VideoBallDetectionSettings(BaseModel):
    frame_stride: Literal[1]
    imgsz: Literal[960]
    confidence_threshold: Literal[0.15]
    max_det: Literal[20]


class VideoBallDetectionsDocument(BaseModel):
    analysis_id: str
    model_path_used: str
    model_class_names: list[str]
    settings: VideoBallDetectionSettings
    frames: list[FrameDetectionRecord]


class VideoBallDetectionSummary(BaseModel):
    analysis_id: str
    status: Literal["ready"]
    created_at: datetime
    completed_at: datetime
    original_video_url: str
    processed_video_url: str
    detections_json_url: str
    detections_csv_url: str
    detection_summary_url: str
    model_path_used: str
    model_warning: str | None = None
    model_class_names: list[str]
    device_used: str
    imgsz: Literal[960]
    confidence_threshold: Literal[0.15]
    frame_stride: Literal[1]
    max_det: Literal[20]
    total_frames: int = Field(gt=0)
    frames_processed: int = Field(gt=0)
    frames_with_candidates: int = Field(ge=0)
    frames_without_candidates: int = Field(ge=0)
    total_candidates: int = Field(ge=0)
    frames_with_multiple_candidates: int = Field(ge=0)
    candidates_inside_pitch_corridor: int = Field(ge=0)
    candidates_outside_pitch_corridor: int = Field(ge=0)
    candidates_without_corridor_information: int = Field(ge=0)
    best_confidence: float = Field(ge=0, le=1)
    average_confidence: float = Field(ge=0, le=1)
    average_candidates_per_detected_frame: float = Field(ge=0)
    processing_duration_seconds: float = Field(ge=0)
    output_video_frame_count: int = Field(gt=0)
    input_fps: float = Field(gt=0)
    output_fps: float = Field(gt=0)
    input_duration_seconds: float = Field(gt=0)
    output_duration_seconds: float = Field(gt=0)
    message: str


class VideoBallDetectionResultResponse(BaseModel):
    success: Literal[True]
    status: Literal["ready"]
    analysis_id: str
    summary: VideoBallDetectionSummary
    frame_candidate_counts: list[int]
    message: str
