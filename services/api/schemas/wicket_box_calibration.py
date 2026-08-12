"""Wicket-box scene-setup contracts for Virtual Pitch Replay."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .video_analysis import CricketPitchGeometry


WicketBoxRole = Literal["NEAR", "FAR"]
WicketBoxValidationStatus = Literal[
    "PENDING",
    "VALID",
    "INVALID_DIMENSIONS",
    "OUT_OF_BOUNDS",
    "OVERLAP",
    "ROLE_ORDER_INVALID",
    "MISSING_ROLE",
]
StumpIdentity = Literal["LEFT", "MIDDLE", "RIGHT"]
StumpLandmarkProvenance = Literal["AUTOMATIC", "USER_CORRECTED"]
StumpLandmarkVisibility = Literal["VISIBLE", "PARTIAL", "OCCLUDED", "UNAVAILABLE"]
WicketBoxCalibrationStatus = Literal[
    "NOT_STARTED",
    "DETECTED",
    "REGISTERED",
    "ACCEPTED",
    "FAILED",
    "NOT_IMPLEMENTED",
]
WicketBoxCalibrationStageStatus = Literal[
    "NOT_IMPLEMENTED",
    "PENDING",
    "READY",
    "FAILED",
]


class WicketBoxModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PixelPoint(WicketBoxModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)


class WicketBox(WicketBoxModel):
    """Axis-aligned wicket region in original video pixel coordinates."""

    role: WicketBoxRole
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    source_image_width: int = Field(gt=0)
    source_image_height: int = Field(gt=0)
    calibration_frame_index: int = Field(ge=0)
    validation_status: WicketBoxValidationStatus = "PENDING"

    @model_validator(mode="after")
    def validate_geometry(self) -> "WicketBox":
        values = (self.x, self.y, self.width, self.height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Wicket box coordinates must be finite.")
        if self.x + self.width > self.source_image_width + 0.001:
            raise ValueError("Wicket box exceeds source image width.")
        if self.y + self.height > self.source_image_height + 0.001:
            raise ValueError("Wicket box exceeds source image height.")
        return self


class StumpLandmark(WicketBoxModel):
    wicket_role: WicketBoxRole
    stump_identity: StumpIdentity
    base: PixelPoint
    top: PixelPoint
    centre: PixelPoint
    visibility: StumpLandmarkVisibility = "VISIBLE"
    confidence: float = Field(ge=0, le=1)
    provenance: StumpLandmarkProvenance = "AUTOMATIC"


class ReprojectionDiagnostic(WicketBoxModel):
    landmark_id: str
    observed_pixel_x: float = Field(ge=0)
    observed_pixel_y: float = Field(ge=0)
    reprojected_pixel_x: float
    reprojected_pixel_y: float
    error_px: float = Field(ge=0)


class WicketBoxCalibrationCandidateSummary(WicketBoxModel):
    candidate_id: str
    assignment_hypothesis: Literal["A", "B"]
    near_semantic_end: Literal["bowler", "striker"]
    far_semantic_end: Literal["bowler", "striker"]
    reprojection_rmse_px: float | None = Field(default=None, ge=0)
    near_wicket_error_px: float | None = Field(default=None, ge=0)
    far_wicket_error_px: float | None = Field(default=None, ge=0)
    camera_height_m: float | None = None
    focal_length_px: float | None = Field(default=None, gt=0)
    stability_score: float | None = Field(default=None, ge=0, le=1)
    physically_valid: bool
    rejection_reasons: list[str] = Field(default_factory=list)


class WicketBoxRegistrationSummary(WicketBoxModel):
    recommended: WicketBoxCalibrationCandidateSummary | None = None
    alternative: WicketBoxCalibrationCandidateSummary | None = None
    rejected: list[WicketBoxCalibrationCandidateSummary] = Field(default_factory=list)
    auto_selected: bool = False
    orientation_ambiguous: bool = False
    user_message: str = ""


class CalibrationResult(WicketBoxModel):
    """Full wicket-box calibration persistence — not a boolean flag."""

    status: WicketBoxCalibrationStatus
    analysis_id: str = Field(min_length=1)
    accepted_at: datetime | None = None
    # Persisted so a re-solve from stored landmarks reaches for the same lens
    # profile the accepted pose was solved with.
    device_id: str | None = Field(default=None, max_length=128)
    # The pitch the accepted pose was solved against. Tracking, physics and
    # replay must read this rather than assuming regulation, or a pose solved
    # on a short rig gets measured against 20.12 m.
    pitch_geometry: CricketPitchGeometry | None = None
    calibration_frame_index: int = Field(ge=0)
    source_image_width: int = Field(gt=0)
    source_image_height: int = Field(gt=0)
    near_wicket_box: WicketBox | None = None
    far_wicket_box: WicketBox | None = None
    stump_landmarks: list[StumpLandmark] = Field(default_factory=list)
    automatic_stump_landmarks: list[StumpLandmark] = Field(default_factory=list)
    camera_matrix: list[list[float]] | None = None
    rotation_matrix: list[list[float]] | None = None
    translation_vector: list[float] | None = None
    distortion_coefficients: list[float] | None = None
    reprojection_rmse_px: float | None = Field(default=None, ge=0)
    reprojection_diagnostics: list[ReprojectionDiagnostic] = Field(
        default_factory=list
    )
    validation_status: WicketBoxValidationStatus = "PENDING"
    warnings: list[str] = Field(default_factory=list)
    registration_summary: WicketBoxRegistrationSummary | None = None
    message: str


class WicketBoxPairValidationResult(WicketBoxModel):
    valid: bool
    near_box: WicketBox | None = None
    far_box: WicketBox | None = None
    validation_status: WicketBoxValidationStatus
    overlap_fraction: float | None = Field(default=None, ge=0, le=1)
    role_order_valid: bool
    messages: list[str] = Field(default_factory=list)


class WicketBoxCalibrationDetectResponse(WicketBoxModel):
    success: bool
    status: WicketBoxCalibrationStageStatus
    analysis_id: str
    calibration_frame_index: int | None = Field(default=None, ge=0)
    source_image_width: int | None = Field(default=None, gt=0)
    source_image_height: int | None = Field(default=None, gt=0)
    near_wicket_box: WicketBox | None = None
    far_wicket_box: WicketBox | None = None
    stump_landmarks: list[StumpLandmark] = Field(default_factory=list)
    message: str


class WicketBoxCalibrationRegisterRequest(WicketBoxModel):
    analysis_id: str = Field(min_length=1)
    calibration_frame_index: int = Field(ge=0)
    source_image_width: int = Field(gt=0)
    source_image_height: int = Field(gt=0)
    near_wicket_box: WicketBox
    far_wicket_box: WicketBox
    stump_landmarks: list[StumpLandmark] = Field(default_factory=list)
    # Identifies a solved lens profile, which removes focal length from the PnP
    # unknowns. Absent means fall back to the bounded focal sweep.
    device_id: str | None = Field(default=None, max_length=128)
    # The pitch the operator declares they are actually standing on. Absent
    # means regulation, so a full-size net keeps solving exactly as before.
    pitch_geometry: CricketPitchGeometry | None = None

    @model_validator(mode="after")
    def validate_roles_and_ids(self) -> "WicketBoxCalibrationRegisterRequest":
        if self.near_wicket_box.role != "NEAR":
            raise ValueError("near_wicket_box.role must be NEAR.")
        if self.far_wicket_box.role != "FAR":
            raise ValueError("far_wicket_box.role must be FAR.")
        if self.analysis_id.strip() == "":
            raise ValueError("analysis_id is required.")
        return self


class WicketBoxCalibrationRegisterResponse(WicketBoxModel):
    success: bool
    status: WicketBoxCalibrationStageStatus
    analysis_id: str
    validation: WicketBoxPairValidationResult
    calibration: CalibrationResult | None = None
    message: str


class WicketBoxCalibrationAcceptRequest(WicketBoxModel):
    analysis_id: str = Field(min_length=1)
    accept_registered_calibration: bool = True
    user_note: str | None = Field(default=None, max_length=1000)


class WicketBoxCalibrationAcceptResponse(WicketBoxModel):
    success: bool
    status: WicketBoxCalibrationStageStatus
    analysis_id: str
    calibration: CalibrationResult | None = None
    message: str


def boxes_overlap(box_a: WicketBox, box_b: WicketBox) -> bool:
    """Return True when two axis-aligned boxes intersect."""
    if (
        box_a.source_image_width != box_b.source_image_width
        or box_a.source_image_height != box_b.source_image_height
    ):
        return False
    horizontal_gap = max(box_a.x, box_b.x) - min(
        box_a.x + box_a.width,
        box_b.x + box_b.width,
    )
    vertical_gap = max(box_a.y, box_b.y) - min(
        box_a.y + box_a.height,
        box_b.y + box_b.height,
    )
    return horizontal_gap < 0 and vertical_gap < 0


def overlap_fraction(box_a: WicketBox, box_b: WicketBox) -> float:
    if not boxes_overlap(box_a, box_b):
        return 0.0
    overlap_width = min(box_a.x + box_a.width, box_b.x + box_b.width) - max(
        box_a.x,
        box_b.x,
    )
    overlap_height = min(box_a.y + box_a.height, box_b.y + box_b.height) - max(
        box_a.y,
        box_b.y,
    )
    overlap_area = max(0.0, overlap_width) * max(0.0, overlap_height)
    smaller_area = min(box_a.width * box_a.height, box_b.width * box_b.height)
    if smaller_area <= 0:
        return 0.0
    return overlap_area / smaller_area


def validate_wicket_box_pair(
    near_box: WicketBox,
    far_box: WicketBox,
    *,
    require_role_order: bool = True,
) -> WicketBoxPairValidationResult:
    """Validate NEAR/FAR wicket boxes without silently swapping roles."""
    messages: list[str] = []

    if near_box.role != "NEAR":
        messages.append("near_wicket_box.role must remain NEAR.")
    if far_box.role != "FAR":
        messages.append("far_wicket_box.role must remain FAR.")

    if (
        near_box.source_image_width != far_box.source_image_width
        or near_box.source_image_height != far_box.source_image_height
    ):
        messages.append("Both wicket boxes must share source image dimensions.")

    role_order_valid = True
    if require_role_order:
        # ponytail: NEAR wicket is typically lower/larger in frame; reject swap attempts.
        near_centre_y = near_box.y + near_box.height / 2
        far_centre_y = far_box.y + far_box.height / 2
        near_area = near_box.width * near_box.height
        far_area = far_box.width * far_box.height
        if near_centre_y < far_centre_y and near_area < far_area:
            role_order_valid = False
            messages.append(
                "NEAR/FAR role order appears inverted relative to image perspective."
            )

    overlap = overlap_fraction(near_box, far_box)
    if overlap > 0:
        messages.append(f"Wicket boxes overlap (fraction={overlap:.3f}).")

    validation_status: WicketBoxValidationStatus = "VALID"
    if messages:
        if overlap > 0:
            validation_status = "OVERLAP"
        elif not role_order_valid:
            validation_status = "ROLE_ORDER_INVALID"
        else:
            validation_status = "INVALID_DIMENSIONS"

    return WicketBoxPairValidationResult(
        valid=not messages,
        near_box=near_box,
        far_box=far_box,
        validation_status=validation_status,
        overlap_fraction=overlap if overlap > 0 else None,
        role_order_valid=role_order_valid,
        messages=messages,
    )
