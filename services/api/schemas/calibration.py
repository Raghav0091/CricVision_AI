from typing import Any, Literal

from pydantic import BaseModel, Field


class AlignmentBox(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)


class AlignmentBoxLayout(BaseModel):
    striker: AlignmentBox
    non_striker: AlignmentBox


class CalibrationRequest(BaseModel):
    frame_data_url: str
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    box_layout: AlignmentBoxLayout


class DetectionBoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class StumpDetection(BaseModel):
    found: bool
    confidence: float
    bbox: DetectionBoundingBox | None
    source_box: Literal["striker", "non_striker"]
    class_name: str | None = None


class Point(BaseModel):
    x: int
    y: int


class VirtualStump(BaseModel):
    name: Literal["left", "middle", "right"]
    top: Point
    base: Point


class VirtualBails(BaseModel):
    name: Literal["bails"]
    left: Point
    right: Point


class VirtualStumpEnd(BaseModel):
    geometry_type: Literal["estimated_from_bbox"]
    stumps: list[VirtualStump]
    bails: VirtualBails


class VirtualStumpGeometry(BaseModel):
    striker: VirtualStumpEnd | None
    non_striker: VirtualStumpEnd | None


class CalibrationDebugFiles(BaseModel):
    original: str
    overlay: str | None = None


class CalibrationResponse(BaseModel):
    success: bool
    status: Literal[
        "invalid_calibration_frame",
        "stump_detector_missing",
        "stump_detector_error",
        "stumps_not_found",
        "setup_complete",
    ]
    quality: Literal["Unavailable", "Poor", "Partial", "Good"]
    reason: str
    message: str
    calibration_frame_path: str | None = None
    model_path: str | None = None
    detections: dict[str, StumpDetection] | None = None
    virtual_stumps: VirtualStumpGeometry | None = None
    environment_context: dict[str, Any] | None = None
    debug_files: CalibrationDebugFiles | None = None
