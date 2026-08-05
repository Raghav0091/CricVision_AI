"""Normalized OpenCV camera contract for developer renderer bridges."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .real_pitch_registration import RealProjectedPitchGeometry
from .virtual_pitch import ProjectedPitchGeometry


CameraBridgeSource = Literal[
    "SYNTHETIC_VIRTUAL_PITCH",
    "ACCEPTED_SCENE_CALIBRATION",
    "ACCEPTED_WICKET_BOX_CALIBRATION",
    "REFINED_SCENE_CALIBRATION_CANDIDATE",
    "REAL_PITCH_REGISTRATION_CANDIDATE",
]
DistortionMode = Literal[
    "ZERO_DISTORTION",
    "PREUNDISTORTED_FRAME",
    "NONZERO_DISTORTION_UNSUPPORTED",
]


class CameraBridgeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CameraBridgeDistortion(CameraBridgeModel):
    mode: DistortionMode
    coefficients: list[float] = Field(min_length=4, max_length=14)
    coefficient_order: str
    maximum_absolute_coefficient: float = Field(ge=0)
    frame_preundistorted: bool
    exact_pinhole_rendering_supported: bool
    warning: str | None = None


class CameraBridgeSetupFrame(CameraBridgeModel):
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    image_url: str
    media_type: Literal["image/jpeg", "image/png"]


class CameraBridgeInput(CameraBridgeModel):
    source: CameraBridgeSource
    source_version: str
    analysis_id: str | None = None
    candidate_id: str
    accepted: bool
    classification: str
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    camera_matrix: list[list[float]]
    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float
    cy: float
    skew: float
    distortion: CameraBridgeDistortion
    rotation_representation: Literal["OPENCV_RODRIGUES_AND_MATRIX"] = (
        "OPENCV_RODRIGUES_AND_MATRIX"
    )
    rotation_vector: list[float] = Field(min_length=3, max_length=3)
    rotation_matrix: list[list[float]]
    translation_vector: list[float] = Field(min_length=3, max_length=3)
    camera_world_position: list[float] = Field(min_length=3, max_length=3)
    extrinsic_convention: Literal["X_CAMERA = R * X_CRICVISION_WORLD + T"] = (
        "X_CAMERA = R * X_CRICVISION_WORLD + T"
    )
    world_coordinate_system: Literal[
        "CRICVISION_X_RIGHT_Y_BOWLER_TO_STRIKER_Z_UP_METRES"
    ] = "CRICVISION_X_RIGHT_Y_BOWLER_TO_STRIKER_Z_UP_METRES"
    camera_coordinate_system: Literal["OPENCV_X_RIGHT_Y_DOWN_Z_FORWARD"] = (
        "OPENCV_X_RIGHT_Y_DOWN_Z_FORWARD"
    )
    near_m: float = Field(gt=0)
    far_m: float = Field(gt=0)
    setup_frame: CameraBridgeSetupFrame | None = None
    warnings: list[str] = Field(default_factory=list)


class CameraBridgePixelBounds(CameraBridgeModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class CameraBridgePixelPoint(CameraBridgeModel):
    x: float
    y: float


class ConfirmedWicketBoxEvidence(CameraBridgeModel):
    bounds: CameraBridgePixelBounds
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    role: Literal["NEAR_WICKET", "FAR_WICKET"]
    source: Literal["DETECTOR", "MANUAL", "DETECTOR_ADJUSTED"]
    detector_confidence: float | None = Field(default=None, ge=0, le=1)
    bottom_left: CameraBridgePixelPoint
    bottom_right: CameraBridgePixelPoint
    bottom_centre: CameraBridgePixelPoint
    top_centre: CameraBridgePixelPoint
    box_centre: CameraBridgePixelPoint


class WicketProjectionFitMetrics(CameraBridgeModel):
    observed_bounds: CameraBridgePixelBounds
    projected_bounds: CameraBridgePixelBounds | None = None
    centre_error_px: float | None = Field(default=None, ge=0)
    width_error_px: float | None = Field(default=None, ge=0)
    height_error_px: float | None = Field(default=None, ge=0)
    base_error_px: float | None = Field(default=None, ge=0)
    width_error_ratio: float | None = Field(default=None, ge=0)
    height_error_ratio: float | None = Field(default=None, ge=0)
    box_iou: float | None = Field(default=None, ge=0, le=1)


class ConfirmedWicketFitValidation(CameraBridgeModel):
    status: Literal["FIT_READY", "FIT_APPROXIMATE", "FIT_FAILED"]
    fit_score: float = Field(ge=0, le=1)
    native_image_width: int = Field(gt=0)
    native_image_height: int = Field(gt=0)
    near_wicket_evidence: ConfirmedWicketBoxEvidence
    far_wicket_evidence: ConfirmedWicketBoxEvidence
    near_wicket: WicketProjectionFitMetrics
    far_wicket: WicketProjectionFitMetrics
    reasons: list[str] = Field(default_factory=list)


class CameraBridgeResponse(CameraBridgeModel):
    bridge_version: Literal["opencv_three_camera_bridge_v1"] = (
        "opencv_three_camera_bridge_v1"
    )
    status: Literal["AVAILABLE", "UNAVAILABLE"]
    camera: CameraBridgeInput | None = None
    projected_pitch_geometry: (
        ProjectedPitchGeometry | RealProjectedPitchGeometry | None
    ) = None
    fit_status: Literal["FIT_READY", "FIT_APPROXIMATE", "FIT_FAILED"] | None = None
    fit_validation: ConfirmedWicketFitValidation | None = None
    metrics_unlocked: Literal[False] = False
    developer_only: Literal[True] = True
    warnings: list[str] = Field(default_factory=list)
    message: str


class ConfirmedWicketPixelBox(CameraBridgeModel):
    x_min: float = Field(ge=0)
    y_min: float = Field(ge=0)
    x_max: float = Field(gt=0)
    y_max: float = Field(gt=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    role: Literal["NEAR_WICKET", "FAR_WICKET"]
    source: Literal["DETECTOR", "MANUAL", "DETECTOR_ADJUSTED"]
    detector_confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> "ConfirmedWicketPixelBox":
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("Wicket box maximums must exceed minimums.")
        if self.x_max > self.frame_width or self.y_max > self.frame_height:
            raise ValueError("Wicket box must remain inside the native frame.")
        if not math.isclose(self.width, self.x_max - self.x_min, abs_tol=1e-6):
            raise ValueError("Wicket box width must match x_max - x_min.")
        if not math.isclose(self.height, self.y_max - self.y_min, abs_tol=1e-6):
            raise ValueError("Wicket box height must match y_max - y_min.")
        return self


class ConfirmedWicketCameraFitRequest(CameraBridgeModel):
    preset_id: str = Field(default="STANDARD_REAR_WICKET_NET_V1", min_length=1)
    near_wicket: ConfirmedWicketPixelBox
    far_wicket: ConfirmedWicketPixelBox

    @model_validator(mode="after")
    def validate_pair(self) -> "ConfirmedWicketCameraFitRequest":
        if self.near_wicket.role != "NEAR_WICKET" or self.far_wicket.role != "FAR_WICKET":
            raise ValueError("Near and far wicket roles must match their fields.")
        if (
            self.near_wicket.frame_width != self.far_wicket.frame_width
            or self.near_wicket.frame_height != self.far_wicket.frame_height
        ):
            raise ValueError("Confirmed wicket boxes must share one native frame.")
        return self
