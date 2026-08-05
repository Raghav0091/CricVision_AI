"""Strict contracts for Pitch-Space Delivery Analysis Lab V1 setup and fit."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PitchSpaceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ImagePoint(PitchSpaceModel):
    x: float
    y: float


class PitchPoint(PitchSpaceModel):
    x_m: float
    y_m: float


class FrameWicketBox(PitchSpaceModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    confidence: float = Field(ge=0, le=1)
    clipped: bool = False
    source: str


class SetupFrameEvaluation(PitchSpaceModel):
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    decoded: bool
    image_width: int = Field(ge=0)
    image_height: int = Field(ge=0)
    sharpness: float = Field(ge=0)
    brightness: float = Field(ge=0, le=255)
    near_wicket: FrameWicketBox | None = None
    far_wicket: FrameWicketBox | None = None
    suitable: bool
    quality_score: float = Field(ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)


class SetupFrameDecision(PitchSpaceModel):
    preferred_frame_attempted: Literal[True] = True
    preferred_frame_index: Literal[0] = 0
    preferred_frame_passed: bool
    selected_frame_index: int | None = Field(default=None, ge=0)
    selected_timestamp_seconds: float | None = Field(default=None, ge=0)
    fallback_used: bool
    fallback_candidates: list[int] = Field(default_factory=list)
    evaluations: list[SetupFrameEvaluation] = Field(default_factory=list)
    selection_reasons: list[str] = Field(default_factory=list)
    quality_score: float = Field(ge=0, le=1)


class StableWicketBox(PitchSpaceModel):
    perspective_role: Literal["NEAR", "FAR"]
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    confidence: float = Field(ge=0, le=1)
    frame_support: int = Field(ge=1)
    supporting_frame_indices: list[int]
    centre_spread_px: float = Field(ge=0)
    size_spread_ratio: float = Field(ge=0)
    clipped: bool
    source: Literal["persisted", "newly_generated"]


class PitchFitCorrespondence(PitchSpaceModel):
    semantic_id: str
    image_point: ImagePoint
    pitch_point: PitchPoint


class ProjectedPitchPrimitive(PitchSpaceModel):
    primitive_id: str
    primitive_type: Literal["LINE", "POLYGON", "WICKET_BASE"]
    image_points: list[ImagePoint]


class PitchFitResult(PitchSpaceModel):
    status: Literal["READY", "PITCH_FIT_FAILED"]
    selected_hypothesis: str | None = None
    near_semantic_end: Literal["bowler", "striker"] | None = None
    image_left_is_pitch_left: bool | None = None
    image_to_pitch_homography: list[list[float]] | None = None
    pitch_to_image_homography: list[list[float]] | None = None
    correspondences: list[PitchFitCorrespondence] = Field(default_factory=list)
    projected_pitch: list[ProjectedPitchPrimitive] = Field(default_factory=list)
    determinant: float | None = None
    condition_number: float | None = None
    reprojection_rmse_px: float | None = None
    projected_pitch_area_px2: float | None = None
    fit_score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)


class CameraStabilityResult(PitchSpaceModel):
    status: Literal["FIXED_CAMERA", "MINOR_DRIFT", "UNSTABLE_CAMERA", "UNAVAILABLE"]
    frames_checked: list[int] = Field(default_factory=list)
    maximum_centre_drift_ratio: float | None = Field(default=None, ge=0)
    maximum_scale_change_ratio: float | None = Field(default=None, ge=0)
    confidence: float = Field(ge=0, le=1)
    reliable_until_frame: int | None = Field(default=None, ge=0)
    warnings: list[str] = Field(default_factory=list)


class ImageSpaceTrackPoint(PitchSpaceModel):
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    image_x_px: float
    image_y_px: float
    detection_confidence: float = Field(ge=0, le=1)
    provenance: str
    track_valid: bool = True


class PitchSpaceTrackPoint(PitchSpaceModel):
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    image_x_px: float
    image_y_px: float
    pitch_x_m: float
    pitch_y_m: float
    detection_confidence: float = Field(ge=0, le=1)
    pitch_fit_confidence: float = Field(ge=0, le=1)
    combined_confidence: float = Field(ge=0, le=1)
    provenance: str
    in_pitch_bounds: bool
    bounce_phase: Literal["PRE_BOUNCE", "BOUNCE", "POST_BOUNCE", "UNKNOWN"]
    warnings: list[str] = Field(default_factory=list)


class BounceAlternative(PitchSpaceModel):
    frame_index: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)


class PitchSpaceBounceResult(PitchSpaceModel):
    status: Literal["DETECTED", "UNCERTAIN", "UNAVAILABLE"]
    bounce_frame: int | None = Field(default=None, ge=0)
    bounce_timestamp_seconds: float | None = Field(default=None, ge=0)
    pitch_x_m: float | None = None
    pitch_y_m: float | None = None
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)
    alternative_candidates: list[BounceAlternative] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PitchSpaceLineLengthResult(PitchSpaceModel):
    status: Literal["AVAILABLE", "UNAVAILABLE"]
    line: Literal["MIDDLE", "PITCH_LEFT", "PITCH_RIGHT", "UNAVAILABLE"]
    length: Literal["YORKER", "FULL", "GOOD_LENGTH", "BACK_OF_A_LENGTH", "SHORT", "UNAVAILABLE"]
    lateral_offset_from_middle_m: float | None = None
    distance_from_striker_wicket_m: float | None = None
    distance_from_striker_popping_crease_m: float | None = None
    confidence: float = Field(ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)


class EstimatedPlanarSpeedResult(PitchSpaceModel):
    status: Literal["AVAILABLE", "UNAVAILABLE"]
    label: Literal["ESTIMATED_PLANAR_SPEED"]
    speed_mps: float | None = Field(default=None, ge=0)
    speed_kmh: float | None = Field(default=None, ge=0)
    frames_used: list[int] = Field(default_factory=list)
    interval_seconds: float | None = Field(default=None, ge=0)
    method: Literal["ROBUST_LONGITUDINAL_REGRESSION"]
    confidence: Literal["HIGH", "MEDIUM", "LOW", "INSUFFICIENT_EVIDENCE"]
    confidence_score: float = Field(ge=0, le=1)
    estimated_range_kmh: list[float] | None = Field(default=None, min_length=2, max_length=2)
    warnings: list[str] = Field(default_factory=list)


class EstimatedLateralMovementResult(PitchSpaceModel):
    status: Literal["AVAILABLE", "UNAVAILABLE"]
    label: Literal["ESTIMATED_LATERAL_MOVEMENT"]
    direction: Literal["PITCH_LEFT", "PITCH_RIGHT", "STRAIGHT", "UNAVAILABLE"]
    movement_m: float | None = Field(default=None, ge=0)
    signed_movement_m: float | None = None
    maximum_movement_m: float | None = Field(default=None, ge=0)
    frames_used: list[int] = Field(default_factory=list)
    confidence: Literal["HIGH", "MEDIUM", "LOW", "INSUFFICIENT_EVIDENCE"]
    confidence_score: float = Field(ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)


class StageTimings(PitchSpaceModel):
    upload_ms: float | None = Field(default=None, ge=0)
    observation_load_or_run_ms: float | None = Field(default=None, ge=0)
    setup_selection_ms: float | None = Field(default=None, ge=0)
    box_stabilisation_ms: float | None = Field(default=None, ge=0)
    pitch_fit_ms: float | None = Field(default=None, ge=0)
    overlay_ms: float | None = Field(default=None, ge=0)
    ball_tracking_load_ms: float | None = Field(default=None, ge=0)
    pitch_space_conversion_ms: float | None = Field(default=None, ge=0)
    bounce_ms: float | None = Field(default=None, ge=0)
    speed_ms: float | None = Field(default=None, ge=0)
    movement_ms: float | None = Field(default=None, ge=0)
    replay_preparation_ms: float | None = Field(default=None, ge=0)
    persistence_ms: float | None = Field(default=None, ge=0)
    total_ms: float = Field(ge=0)


class PitchSpaceDeliveryAnalysisV1(PitchSpaceModel):
    version: Literal["pitch_space_delivery_analysis_v1"] = "pitch_space_delivery_analysis_v1"
    analysis_id: str
    status: Literal[
        "NO_VIDEO",
        "UPLOAD_FAILED",
        "FRAME_ZERO_UNUSABLE",
        "INSUFFICIENT_WICKETS",
        "PITCH_FIT_FAILED",
        "UNSTABLE_CAMERA",
        "BALL_TRACK_UNAVAILABLE",
        "BOUNCE_UNAVAILABLE",
        "SPEED_UNAVAILABLE",
        "MOVEMENT_UNAVAILABLE",
        "PARTIAL",
        "COMPLETE",
        "FAILED",
    ]
    source_video_url: str
    source_filename: str
    native_width: int = Field(gt=0)
    native_height: int = Field(gt=0)
    fps: float = Field(gt=0)
    frame_count: int = Field(ge=1)
    setup_frame_decision: SetupFrameDecision
    wicket_observation_source: Literal["persisted", "newly_generated"]
    stable_near_wicket: StableWicketBox | None = None
    stable_far_wicket: StableWicketBox | None = None
    pitch_fit: PitchFitResult
    camera_stability: CameraStabilityResult
    image_space_track: list[ImageSpaceTrackPoint] = Field(default_factory=list)
    pitch_space_track: list[PitchSpaceTrackPoint] = Field(default_factory=list)
    bounce: PitchSpaceBounceResult | None = None
    line: PitchSpaceLineLengthResult | None = None
    length: PitchSpaceLineLengthResult | None = None
    estimated_planar_speed: EstimatedPlanarSpeedResult | None = None
    estimated_lateral_movement: EstimatedLateralMovementResult | None = None
    overlay_url: str | None = None
    overall_confidence: float = Field(ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)
    unavailable_metrics: list[str] = Field(default_factory=list)
    downstream_statuses: list[str] = Field(default_factory=list)
    production_accepted: Literal[False] = False
    metrics_unlocked: list[str] = Field(default_factory=list, max_length=0)
    airborne_3d_available: Literal[False] = False
    stage_timings: StageTimings


class RecentPitchSpaceAnalysis(PitchSpaceModel):
    analysis_id: str
    status: str
    source_filename: str
    report_url: str


class RecentPitchSpaceAnalyses(PitchSpaceModel):
    items: list[RecentPitchSpaceAnalysis] = Field(default_factory=list)
