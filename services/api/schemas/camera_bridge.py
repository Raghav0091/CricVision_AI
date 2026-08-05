"""Normalized OpenCV camera contract for developer renderer bridges."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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


class CameraBridgeResponse(CameraBridgeModel):
    bridge_version: Literal["opencv_three_camera_bridge_v1"] = (
        "opencv_three_camera_bridge_v1"
    )
    status: Literal["AVAILABLE", "UNAVAILABLE"]
    camera: CameraBridgeInput | None = None
    projected_pitch_geometry: (
        ProjectedPitchGeometry | RealProjectedPitchGeometry | None
    ) = None
    metrics_unlocked: Literal[False] = False
    developer_only: Literal[True] = True
    warnings: list[str] = Field(default_factory=list)
    message: str
