"""Unified assisted scene-calibration contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .real_pitch_registration import (
    CameraPoseCandidate,
    RealProjectedPitchGeometry,
)
from .wicket_observation import PixelPoint, SetupFrameCandidate


SceneCalibrationStage = Literal[
    "NOT_STARTED",
    "DETECTING_WICKETS",
    "OBSERVING_WICKETS",
    "GENERATING_POSE",
    "NEEDS_ADJUSTMENT",
    "ORIENTATION_REQUIRED",
    "GROUND_PLANE_READY",
    "METRIC_3D_READY",
    "INSUFFICIENT_EVIDENCE",
    "FAILED",
]
CalibrationLevel = Literal[
    "UNAVAILABLE",
    "VISUAL_ONLY",
    "GROUND_PLANE_READY",
    "METRIC_3D_READY",
]
AnchorSource = Literal[
    "automatic",
    "manually_adjusted",
    "manually_added",
]
AnchorKind = Literal["wicket", "crease", "pitch_edge"]
ImageLeftMapping = Literal[
    "IMAGE_LEFT_IS_PITCH_LEFT",
    "IMAGE_LEFT_IS_PITCH_RIGHT",
]
OrientationSelection = ImageLeftMapping | Literal["NOT_SURE"]
CameraEnd = Literal["bowler", "striker", "unknown"]
OrientationEvidenceType = Literal[
    "USER_CONFIRMED_LATERAL_ORIENTATION",
    "SAVED_CAMERA_ORIENTATION_PRESET",
    "SEMANTIC_PITCH_EDGE_POINT",
    "SEMANTIC_CREASE_ENDPOINT",
    "TRUSTED_CAMERA_END",
    "TRUSTED_SESSION_DIRECTION",
    "FUTURE_AUTOMATIC_ASYMMETRIC_EVIDENCE",
]
OrientationEvidenceSource = Literal[
    "user",
    "saved_preset",
    "manual_anchor",
    "trusted_session",
    "future_automatic",
]


class SceneCalibrationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SceneCalibrationStageEvent(SceneCalibrationModel):
    stage: SceneCalibrationStage
    at: datetime
    message: str


class SceneCalibrationAnchor(SceneCalibrationModel):
    semantic_id: str
    kind: AnchorKind
    wicket_role: Literal["near", "far"] | None = None
    video_point: PixelPoint | None = None
    source: AnchorSource
    original_automatic_point: PixelPoint | None = None
    confidence: float = Field(ge=0, le=1)
    uncertainty_px: float = Field(ge=0)
    adjustment_distance_px: float = Field(ge=0)
    frame_index: int = Field(ge=0)
    valid: bool
    used_for_refinement: bool
    used_for_validation: bool
    validation_messages: list[str] = Field(default_factory=list)


class SceneCalibrationAnchorInput(SceneCalibrationModel):
    semantic_id: str
    video_point: PixelPoint | None = None
    source: AnchorSource
    used_for_refinement: bool = True
    used_for_validation: bool = True


class SceneCalibrationAnchorUpdateRequest(SceneCalibrationModel):
    anchor_version: int = Field(ge=0)
    anchors: list[SceneCalibrationAnchorInput] = Field(
        min_length=1,
        max_length=10,
    )


class SceneCalibrationRefineRequest(SceneCalibrationModel):
    anchor_version: int = Field(ge=1)


class SceneCalibrationActionRequest(SceneCalibrationModel):
    anchor_version: int = Field(ge=0)
    candidate_id: str | None = None


class SceneCalibrationDetectionSummary(SceneCalibrationModel):
    detector_model: str | None = None
    sampled_frame_count: int = Field(ge=0)
    raw_detection_count: int = Field(ge=0)
    rejected_detection_count: int = Field(ge=0)
    reused_persisted_result: bool


class SceneCalibrationObservationSummary(SceneCalibrationModel):
    status: str
    setup_frame_index: int | None = Field(default=None, ge=0)
    supporting_frame_count: int = Field(ge=0)
    near_wicket_available: bool
    far_wicket_available: bool
    available_anchor_count: int = Field(ge=0)
    result_url: str | None = None


class SceneCalibrationRegistrationSummary(SceneCalibrationModel):
    status: str
    attempted: bool
    selected_candidate_id: str | None = None
    assignment_hypothesis: Literal["A", "B"] | None = None
    focal_length_px: float | None = Field(default=None, gt=0)
    reprojection_rmse_px: float | None = Field(default=None, ge=0)
    median_reprojection_error_px: float | None = Field(default=None, ge=0)
    maximum_inlier_error_px: float | None = Field(default=None, ge=0)
    inlier_count: int = Field(ge=0)
    outlier_count: int = Field(ge=0)
    wicket_envelope_score: float | None = Field(default=None, ge=0, le=1)
    temporal_stability_score: float | None = Field(default=None, ge=0, le=1)
    independent_scene_score: float | None = Field(default=None, ge=0, le=1)
    ambiguity_score: float = Field(ge=0, le=1)
    result_url: str | None = None


class CalibrationThresholdResult(SceneCalibrationModel):
    threshold_id: str
    passed: bool
    value: float | int | bool | str | None
    requirement: str
    reason: str


class OrientationEvidence(SceneCalibrationModel):
    evidence_id: str
    evidence_type: OrientationEvidenceType
    source: OrientationEvidenceSource
    frame_index: int | None = Field(default=None, ge=0)
    native_pixel_coordinate: PixelPoint | None = None
    semantic_label: str
    confidence: float = Field(ge=0, le=1)
    uncertainty: float = Field(default=0.0, ge=0)
    authoritative: bool = False
    supports_candidate_ids: list[str] = Field(default_factory=list)
    rejects_candidate_ids: list[str] = Field(default_factory=list)
    explanation: str
    created_at: datetime
    user_confirmed: bool = False


class OrientationResolution(SceneCalibrationModel):
    required: bool
    resolved: bool
    image_left_mapping: ImageLeftMapping | None = None
    camera_end: CameraEnd | None = None
    ambiguity_before: float = Field(ge=0, le=1)
    ambiguity_after: float = Field(ge=0, le=1)
    selected_candidate_id: str | None = None
    rejected_candidate_ids: list[str] = Field(default_factory=list)
    consistent_candidate_ids: list[str] = Field(default_factory=list)
    evidence_applied: list[OrientationEvidence] = Field(default_factory=list)
    symmetric_evidence_insufficient: list[str] = Field(default_factory=list)
    remaining_failures: list[str] = Field(default_factory=list)


class SceneCalibrationOrientationRequest(SceneCalibrationModel):
    anchor_version: int = Field(ge=0)
    image_left_mapping: OrientationSelection
    camera_end: CameraEnd = "unknown"
    create_preset: bool = False
    preset_name: str | None = Field(default=None, max_length=80)
    user_confirmed_same_fixed_setup: bool = False


class CameraOrientationPreset(SceneCalibrationModel):
    preset_id: str
    preset_name: str
    version: Literal["v1"] = "v1"
    created_at: datetime
    updated_at: datetime
    source_analysis_id: str
    native_width: int = Field(gt=0)
    native_height: int = Field(gt=0)
    rotation_metadata: str | None = None
    camera_device_identifier: str | None = None
    lens_or_focal_metadata: str | None = None
    camera_end: CameraEnd
    image_left_mapping: ImageLeftMapping
    virtual_pitch_version: Literal["v1"] = "v1"
    confidence: float = Field(ge=0, le=1)
    user_confirmed: bool
    compatible: bool = False
    compatibility_reasons: list[str] = Field(default_factory=list)


class SceneCalibrationPresetResponse(SceneCalibrationModel):
    analysis_id: str
    compatible_presets: list[CameraOrientationPreset] = Field(default_factory=list)
    rejected_presets: list[CameraOrientationPreset] = Field(default_factory=list)


class SceneCalibrationPresetRequest(SceneCalibrationModel):
    anchor_version: int = Field(ge=0)
    preset_id: str
    user_confirmed_same_fixed_setup: bool


class SceneCalibrationValidation(SceneCalibrationModel):
    eligible_level: CalibrationLevel
    checks: list[CalibrationThresholdResult] = Field(default_factory=list)
    accepted_anchor_count: int = Field(ge=0)
    manually_adjusted_anchor_count: int = Field(ge=0)
    manually_added_anchor_count: int = Field(ge=0)
    all_required_checks_passed: bool
    failure_reasons: list[str] = Field(default_factory=list)


class AcceptedCalibrationSummary(SceneCalibrationModel):
    revision: int = Field(ge=1)
    accepted_by_user: Literal[True] = True
    accepted_at: datetime
    accepted_level: Literal["GROUND_PLANE_READY", "METRIC_3D_READY"]
    accepted_candidate_id: str
    anchor_version: int = Field(ge=1)
    virtual_pitch_version: Literal["v1"] = "v1"
    registration_version: Literal["v1"] = "v1"
    snapshot_url: str
    image_left_mapping: ImageLeftMapping | None = None
    orientation_preset_id: str | None = None


class AcceptedSceneCalibrationSnapshot(SceneCalibrationModel):
    scene_calibration_version: Literal["v1"] = "v1"
    analysis_id: str
    revision: int = Field(ge=1)
    accepted_at: datetime
    calibration_level: Literal["GROUND_PLANE_READY", "METRIC_3D_READY"]
    candidate_id: str
    setup_frame: SetupFrameCandidate
    supporting_frames: list[SetupFrameCandidate] = Field(default_factory=list)
    setup_frame_image_url: str | None = None
    raw_wicket_overlay_url: str | None = None
    anchors: list[SceneCalibrationAnchor]
    optional_crease_anchors: list[SceneCalibrationAnchor] = Field(
        default_factory=list
    )
    camera_matrix: list[list[float]]
    distortion_coefficients: list[float]
    rotation_vector: list[float]
    rotation_matrix: list[list[float]]
    translation_vector: list[float]
    projection_matrix: list[list[float]]
    camera_world_position: list[float]
    image_to_pitch_homography: list[list[float]] | None = None
    pitch_to_image_homography: list[list[float]] | None = None
    end_assignment: Literal["A", "B"]
    reprojection_rmse_px: float = Field(ge=0)
    median_reprojection_error_px: float = Field(ge=0)
    maximum_inlier_error_px: float = Field(ge=0)
    correspondence_count: int = Field(ge=0)
    uncertainty: dict[str, object]
    validation: SceneCalibrationValidation
    orientation_evidence: list[OrientationEvidence] = Field(default_factory=list)
    image_left_mapping: ImageLeftMapping | None = None
    camera_end: CameraEnd | None = None
    ambiguity_before_resolution: float | None = Field(default=None, ge=0, le=1)
    ambiguity_after_resolution: float | None = Field(default=None, ge=0, le=1)
    selected_mirror_candidate: str | None = None
    rejected_mirror_candidate: str | None = None
    orientation_preset_id: str | None = None
    semantic_anchor_version: int | None = None
    user_confirmation_timestamp: datetime | None = None
    virtual_pitch_version: Literal["v1"] = "v1"
    registration_version: Literal["v1"] = "v1"


class SceneCalibrationResult(SceneCalibrationModel):
    scene_calibration_version: Literal["v1"] = "v1"
    analysis_id: str
    workflow: Literal["ASSISTED_SCENE_CALIBRATION_V1"] = (
        "ASSISTED_SCENE_CALIBRATION_V1"
    )
    stage: SceneCalibrationStage
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime
    stage_history: list[SceneCalibrationStageEvent] = Field(default_factory=list)
    setup_frame: SetupFrameCandidate | None = None
    supporting_frames: list[SetupFrameCandidate] = Field(default_factory=list)
    setup_frame_image_url: str | None = None
    raw_wicket_overlay_url: str | None = None
    raw_stump_detection_summary: SceneCalibrationDetectionSummary | None = None
    wicket_observation_summary: SceneCalibrationObservationSummary | None = None
    automatic_registration_summary: SceneCalibrationRegistrationSummary | None = None
    refined_registration_summary: SceneCalibrationRegistrationSummary | None = None
    current_anchor_set: list[SceneCalibrationAnchor] = Field(default_factory=list)
    optional_crease_anchors: list[SceneCalibrationAnchor] = Field(
        default_factory=list
    )
    anchor_version: int = Field(default=0, ge=0)
    orientation_required: bool = False
    image_left_mapping: ImageLeftMapping | None = None
    camera_end: CameraEnd | None = None
    orientation_evidence: list[OrientationEvidence] = Field(default_factory=list)
    orientation_resolution: OrientationResolution | None = None
    available_orientation_presets: list[CameraOrientationPreset] = Field(
        default_factory=list
    )
    orientation_preset_id: str | None = None
    selected_candidate: CameraPoseCandidate | None = None
    competing_candidate: CameraPoseCandidate | None = None
    projected_pitch_geometry: RealProjectedPitchGeometry | None = None
    competing_projected_pitch_geometry: RealProjectedPitchGeometry | None = None
    validation: SceneCalibrationValidation | None = None
    accepted_calibration: AcceptedCalibrationSummary | None = None
    calibration_level: CalibrationLevel = "UNAVAILABLE"
    metrics_unlocked: list[str] = Field(default_factory=list)
    metrics_locked_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)
    developer_diagnostics_available: bool = False
    legacy_fallback_available: bool = True
    visual_overlay_enabled: bool = False
    message: str
