"""Contracts for non-accepted real-camera pitch registration candidates."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .virtual_pitch import (
    ProjectedLandmark,
    ProjectedLineSegment,
    ProjectedPolygon,
    ProjectedStump,
    ProjectionDiagnostics,
    VirtualCamera,
    WorldPoint3D,
)
from .wicket_observation import PixelBox, PixelPoint, SetupFrameCandidate


RegistrationStatus = Literal[
    "METRIC_3D_CANDIDATE",
    "GROUND_PLANE_CANDIDATE",
    "VISUAL_ONLY",
    "AMBIGUOUS",
    "REGISTRATION_FAILED",
    "NOT_ATTEMPTED",
]


class RegistrationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class RegistrationCorrespondence(RegistrationModel):
    correspondence_id: str
    observed_wicket_role: Literal["near", "far"]
    observed_semantic_id: str
    virtual_semantic_id: str | None = None
    mapping_type: Literal[
        "COARSE_POINTLIKE",
        "DETAILED_EXACT_POINT",
        "TOP_LINE",
        "BASE_LINE",
        "OUTER_AXIS",
        "WICKET_ENVELOPE",
        "WICKET_CENTRE",
    ]
    constraint_category: Literal[
        "EXACT_OR_POINTLIKE_ANCHOR",
        "SOFT_GEOMETRIC_CONSTRAINT",
    ]
    exactness: Literal["EXACT", "POINTLIKE", "SOFT"]
    observed_pixel: PixelPoint | None = None
    observed_line_start: PixelPoint | None = None
    observed_line_end: PixelPoint | None = None
    observed_bbox: PixelBox | None = None
    world_point: WorldPoint3D | None = None
    virtual_line_start: WorldPoint3D | None = None
    virtual_line_end: WorldPoint3D | None = None
    confidence: float = Field(ge=0, le=1)
    uncertainty_px: float = Field(ge=0)
    registration_weight: float = Field(ge=0)
    source_frames: list[int] = Field(default_factory=list)
    status: Literal["USED", "SOFT_ONLY", "REJECTED", "UNAVAILABLE"]
    rejection_reason: str | None = None


class CameraIntrinsicsCandidate(RegistrationModel):
    candidate_id: str
    focal_length_x_px: float = Field(gt=0)
    focal_length_y_px: float = Field(gt=0)
    principal_point_x_px: float
    principal_point_y_px: float
    distortion_coefficients: list[float] = Field(min_length=5, max_length=14)
    source: Literal[
        "trusted_clip_metadata",
        "trusted_device_profile",
        "video_focal_metadata",
        "bounded_image_hypothesis",
    ]
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    horizontal_fov_degrees: float = Field(gt=1, lt=179)
    lower_focal_bound_px: float = Field(gt=0)
    upper_focal_bound_px: float = Field(gt=0)
    focal_bound_reached: bool = False
    principal_point_refined: Literal[False] = False
    distortion_assumption: str


class ReprojectionResidual(RegistrationModel):
    correspondence_id: str
    observed_pixel: PixelPoint
    projected_pixel: PixelPoint
    residual_px: float = Field(ge=0)
    weighted_residual: float = Field(ge=0)
    inlier: bool


class RefinementDiagnostics(RegistrationModel):
    attempted: bool
    converged: bool
    method: str
    robust_loss: str
    initial_cost: float | None = Field(default=None, ge=0)
    final_cost: float | None = Field(default=None, ge=0)
    iterations: int = Field(default=0, ge=0)
    parameters_changed: list[str] = Field(default_factory=list)
    parameters_reaching_bounds: list[str] = Field(default_factory=list)
    residual_breakdown: dict[str, float] = Field(default_factory=dict)
    failure_reason: str | None = None


class PlausibilityCheck(RegistrationModel):
    check_id: str
    passed: bool
    value: float | str | bool | None = None
    reason: str


class IndependentValidation(RegistrationModel):
    anchor_fit_score: float = Field(ge=0, le=1)
    independent_scene_score: float = Field(ge=0, le=1)
    geometry_plausibility_score: float = Field(ge=0, le=1)
    projected_wicket_envelope_score: float = Field(ge=0, le=1)
    crease_edge_support_score: float | None = Field(default=None, ge=0, le=1)
    perspective_convergence_score: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TemporalValidation(RegistrationModel):
    supporting_frame_count: int = Field(ge=0)
    evaluated_frame_ids: list[int] = Field(default_factory=list)
    mean_wicket_alignment_iou: float | None = Field(default=None, ge=0, le=1)
    minimum_wicket_alignment_iou: float | None = Field(default=None, ge=0, le=1)
    stability_score: float = Field(ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)


class PoseUncertainty(RegistrationModel):
    perturbation_count: int = Field(ge=0)
    deterministic_seed: int
    camera_position_spread_m: float | None = Field(default=None, ge=0)
    rotation_spread_degrees: float | None = Field(default=None, ge=0)
    focal_length_spread_px: float | None = Field(default=None, ge=0)
    maximum_overlay_movement_px: float | None = Field(default=None, ge=0)
    projected_bounce_location_sensitivity_px: float | None = Field(
        default=None, ge=0
    )
    stable_for_future_metric_use: bool
    warnings: list[str] = Field(default_factory=list)


class CameraPoseCandidate(RegistrationModel):
    candidate_id: str
    assignment_hypothesis: Literal["A", "B"]
    near_semantic_end: Literal["bowler", "striker"]
    far_semantic_end: Literal["bowler", "striker"]
    lateral_mapping: Literal["image_left_to_world_left", "image_left_to_world_right"]
    setup_frame_index: int = Field(ge=0)
    intrinsics: CameraIntrinsicsCandidate
    anchor_subset_id: str
    attempted: bool
    solver_success: bool
    pnp_method: str
    refinement: RefinementDiagnostics
    rotation_vector: list[float] | None = Field(
        default=None, min_length=3, max_length=3
    )
    translation_vector: list[float] | None = Field(
        default=None, min_length=3, max_length=3
    )
    rotation_matrix: list[list[float]] | None = None
    camera_world_position: list[float] | None = Field(
        default=None, min_length=3, max_length=3
    )
    inlier_correspondence_ids: list[str] = Field(default_factory=list)
    outlier_correspondence_ids: list[str] = Field(default_factory=list)
    reprojection_residuals: list[ReprojectionResidual] = Field(default_factory=list)
    reprojection_rmse_px: float | None = Field(default=None, ge=0)
    median_reprojection_error_px: float | None = Field(default=None, ge=0)
    maximum_inlier_error_px: float | None = Field(default=None, ge=0)
    plausibility_checks: list[PlausibilityCheck] = Field(default_factory=list)
    independent_validation: IndependentValidation | None = None
    temporal_validation: TemporalValidation | None = None
    uncertainty: PoseUncertainty | None = None
    score: float = Field(ge=0, le=1)
    classification: RegistrationStatus
    eligible_for_selection: bool
    failure_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RealProjectedPitchGeometry(RegistrationModel):
    virtual_pitch_model_version: Literal["v1"] = "v1"
    source_camera: VirtualCamera
    projected_landmarks: list[ProjectedLandmark]
    projected_line_segments: list[ProjectedLineSegment]
    projected_stumps: list[ProjectedStump]
    projected_polygons: list[ProjectedPolygon]
    projected_bails: list[ProjectedLineSegment]
    diagnostics: ProjectionDiagnostics
    registered_to_real_setup_frame: Literal[True] = True


class RegistrationDiagnostics(RegistrationModel):
    setup_frame_image_url: str | None = None
    projected_overlay_url: str | None = None
    anchor_residual_overlay_url: str | None = None
    alternate_assignment_overlay_url: str | None = None
    result_json_url: str | None = None
    focal_candidate_count: int = Field(default=0, ge=0)
    pose_candidate_count: int = Field(default=0, ge=0)
    eligibility_reasons: list[str] = Field(default_factory=list)
    rejected_correspondence_count: int = Field(default=0, ge=0)


class RealPitchRegistrationResult(RegistrationModel):
    real_pitch_registration_version: Literal["v1"] = "v1"
    analysis_id: str
    status: RegistrationStatus
    attempted: bool
    setup_frame: SetupFrameCandidate | None = None
    supporting_frames: list[SetupFrameCandidate] = Field(default_factory=list)
    wicket_observation_source: str
    virtual_pitch_version: Literal["v1"] = "v1"
    correspondences: list[RegistrationCorrespondence] = Field(default_factory=list)
    candidates: list[CameraPoseCandidate] = Field(default_factory=list)
    selected_candidate: CameraPoseCandidate | None = None
    competing_candidate: CameraPoseCandidate | None = None
    ambiguity_score: float = Field(ge=0, le=1)
    projected_pitch_geometry: RealProjectedPitchGeometry | None = None
    competing_projected_pitch_geometry: RealProjectedPitchGeometry | None = None
    warnings: list[str] = Field(default_factory=list)
    metrics_locked: Literal[True] = True
    acceptance_required: Literal[True] = True
    failure_reasons: list[str] = Field(default_factory=list)
    diagnostics: RegistrationDiagnostics
    message: str
    developer_only: Literal[True] = True
