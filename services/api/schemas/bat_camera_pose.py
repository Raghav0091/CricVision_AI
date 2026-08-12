from typing import Literal

from pydantic import BaseModel, Field

from .calibration import AlignmentBoxLayout, DetectionBoundingBox
from .delivery_physics import CameraCalibration


class ManualBatBoxes(BaseModel):
    """User-drawn boxes used directly as the bat bounding box, skipping detection."""

    striker: DetectionBoundingBox
    non_striker: DetectionBoundingBox


class BatPoseCalibrationRequest(BaseModel):
    frame_data_url: str
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    box_layout: AlignmentBoxLayout | None = None
    bat_height_m: float = Field(default=0.86, gt=0.3, lt=1.2)
    manual_boxes: ManualBatBoxes | None = None
    bat_detector_model_key: str = "cricshot10k"


class BatDetection(BaseModel):
    found: bool
    confidence: float
    bbox: DetectionBoundingBox | None
    source_box: Literal["striker", "non_striker"]
    class_name: str | None = None


class BatPoseCalibrationResponse(BaseModel):
    success: bool
    status: Literal[
        "invalid_calibration_frame",
        "bat_detector_missing",
        "bat_detector_error",
        "bats_not_found",
        "solve_failed",
        "calibrated",
    ]
    analysis_id: str
    message: str
    detections: dict[str, BatDetection] | None = None
    calibration: CameraCalibration | None = None
