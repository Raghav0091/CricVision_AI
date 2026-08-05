"""Physics Engine V1 contracts for one tracked cricket delivery."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from packages.cricket_vision.calibration.cricket_pitch_geometry import (
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
)


ConfidenceGrade = Literal[
    "HIGH",
    "MEDIUM",
    "LOW",
    "INSUFFICIENT_EVIDENCE",
]
CalibrationMode = Literal[
    "METRIC_3D",
    "METRIC_GROUND_PLANE",
    "IMAGE_SPACE_ONLY",
]
CalibrationConfidence = Literal["HIGH", "MEDIUM", "LOW", "UNAVAILABLE"]
WorldCoordinateSystem = Literal["CRICVISION_PITCH_V1", "CALIBRATION_V2"]
GeometryValidity = Literal[
    "VALID_METRIC_3D",
    "INVALID_REPROJECTION",
    "OUTSIDE_PITCH_GEOMETRY",
    "IMAGE_SPACE_ONLY",
]
PhysicsStatus = Literal[
    "SUCCESS",
    "PARTIAL",
    "IMAGE_SPACE_ONLY",
    "INSUFFICIENT_EVIDENCE",
    "FAILED",
]
PhysicsProvenance = Literal["OBSERVED", "RECONSTRUCTED", "PROJECTED"]


class PhysicsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ObservedBallPoint(PhysicsModel):
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    pixel_x: float = Field(ge=0)
    pixel_y: float = Field(ge=0)
    detector_confidence: float = Field(ge=0, le=1)
    tracker_confidence: float = Field(ge=0, le=1)
    source: str
    candidate_id: str | None = None
    bbox_xyxy: list[float] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
    )


class RejectedObservation(PhysicsModel):
    frame_index: int = Field(ge=0)
    candidate_id: str | None = None
    reason: str
    residual_px: float | None = Field(default=None, ge=0)


class CameraCalibration(PhysicsModel):
    mode: CalibrationMode
    confidence: CalibrationConfidence
    world_coordinate_system: WorldCoordinateSystem = "CRICVISION_PITCH_V1"
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    camera_matrix: list[list[float]] | None = None
    distortion_coefficients: list[float] | None = None
    rotation_vector: list[float] | None = None
    rotation_matrix: list[list[float]] | None = None
    translation_vector: list[float] | None = None
    projection_matrix: list[list[float]] | None = None
    image_to_pitch_homography: list[list[float]] | None = None
    pitch_to_image_homography: list[list[float]] | None = None
    correspondences_used: int = Field(default=0, ge=0)
    reprojection_error_px: float | None = Field(default=None, ge=0)
    calibration_confidence: float = Field(default=0.0, ge=0, le=1)
    failure_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)


class TrajectorySample(PhysicsModel):
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    world_x_m: float | None = None
    world_y_m: float | None = None
    world_z_m: float | None = None
    pixel_x: float
    pixel_y: float
    velocity_x_mps: float | None = None
    velocity_y_mps: float | None = None
    velocity_z_mps: float | None = None
    speed_mps: float | None = Field(default=None, ge=0)
    provenance: PhysicsProvenance
    confidence: float = Field(ge=0, le=1)
    nearest_observation_frame: int | None = Field(default=None, ge=0)
    nearest_observation_delta_frames: int | None = Field(default=None, ge=0)
    nearest_observation_delta_seconds: float | None = Field(default=None, ge=0)


class FittedTrajectoryParameters(PhysicsModel):
    selected_model: Literal[
        "IMAGE_QUADRATIC",
        "BALLISTIC",
        "BALLISTIC_LATERAL",
        "BALLISTIC_LATERAL_DECELERATION",
    ]
    origin_timestamp_seconds: float
    x0_m: float | None = None
    y0_m: float | None = None
    z0_m: float | None = None
    vx0_mps: float | None = None
    vy0_mps: float | None = None
    vz0_mps: float | None = None
    lateral_acceleration_mps2: float | None = None
    forward_acceleration_mps2: float | None = None
    post_bounce_vx_mps: float | None = None
    post_bounce_vy_mps: float | None = None
    post_bounce_vz_mps: float | None = None
    parameter_bounds_reached: list[str] = Field(default_factory=list)


class BouncePhysicsResult(PhysicsModel):
    status: Literal["DETECTED", "ESTIMATED", "INSUFFICIENT_EVIDENCE"]
    frame_index: int | None = Field(default=None, ge=0)
    timestamp_seconds: float | None = Field(default=None, ge=0)
    world_x_m: float | None = None
    world_y_m: float | None = None
    pixel_x: float | None = None
    pixel_y: float | None = None
    distance_from_striker_wicket_m: float | None = Field(default=None, ge=0)
    lateral_offset_m: float | None = None
    confidence: ConfidenceGrade
    confidence_score: float = Field(ge=0, le=1)
    uncertainty_frames: int | None = Field(default=None, ge=0)
    uncertainty_m: float | None = Field(default=None, ge=0)
    directly_supported: bool = False
    evidence: list[str] = Field(default_factory=list)


class SpeedAnalytics(PhysicsModel):
    earliest_measured_speed_kmh: float | None = Field(default=None, ge=0)
    average_pre_bounce_speed_kmh: float | None = Field(default=None, ge=0)
    speed_at_bounce_kmh: float | None = Field(default=None, ge=0)
    average_post_bounce_speed_kmh: float | None = Field(default=None, ge=0)
    observed_temporal_span_seconds: float = Field(default=0.0, ge=0)
    confidence: ConfidenceGrade
    uncertainty_kmh: float | None = Field(default=None, ge=0)
    unavailable_reason: str | None = None


class StumpPlaneCrossing(PhysicsModel):
    timestamp_seconds: float = Field(ge=0)
    world_x_m: float
    world_y_m: float
    world_z_m: float = Field(ge=0)


OverallStumpToStumpStatus = Literal[
    "MEASURED",
    "PARTIALLY_PROJECTED",
    "UNAVAILABLE",
]


class OverallStumpToStumpSpeed(PhysicsModel):
    """Average speed between non-striker and striker wicket plane crossings."""

    speed_mps: float | None = Field(default=None, ge=0)
    speed_kph: float | None = Field(default=None, ge=0)
    start_time_seconds: float | None = Field(default=None, ge=0)
    end_time_seconds: float | None = Field(default=None, ge=0)
    travelled_distance_m: float | None = Field(default=None, ge=0)
    start_crossing: StumpPlaneCrossing | None = None
    end_crossing: StumpPlaneCrossing | None = None
    observed_fraction: float = Field(default=0.0, ge=0, le=1)
    recovered_fraction: float = Field(default=0.0, ge=0, le=1)
    projected_fraction: float = Field(default=0.0, ge=0, le=1)
    confidence: ConfidenceGrade = "INSUFFICIENT_EVIDENCE"
    status: OverallStumpToStumpStatus = "UNAVAILABLE"
    unavailable_reason: str | None = None


class LateralMovementResult(PhysicsModel):
    movement_m: float | None = None
    movement_cm: float | None = None
    direction: Literal[
        "toward_positive_x",
        "toward_negative_x",
        "negligible",
        "unavailable",
    ]
    lateral_acceleration_mps2: float | None = None
    observed_fraction: float = Field(default=0.0, ge=0, le=1)
    confidence: ConfidenceGrade
    uncertainty_m: float | None = Field(default=None, ge=0)
    unavailable_reason: str | None = None


class PostBounceMovementResult(PhysicsModel):
    status: Literal["MEASURED", "PROJECTED", "UNAVAILABLE"]
    lateral_turn_mps: float | None = None
    lateral_turn_cm_at_last_observation: float | None = None
    speed_loss_kmh: float | None = None
    bounce_angle_change_degrees: float | None = None
    observed_points: int = Field(default=0, ge=0)
    confidence: ConfidenceGrade
    unavailable_reason: str | None = None


class LineLengthResult(PhysicsModel):
    line: Literal[
        "outside pitch left",
        "pitch-left channel",
        "middle",
        "pitch-right channel",
        "outside pitch right",
        "unavailable",
    ]
    length: Literal[
        "yorker",
        "full",
        "good length",
        "short",
        "very short",
        "unavailable",
    ]
    bounce_distance_from_striker_m: float | None = Field(default=None, ge=0)
    lateral_offset_from_middle_m: float | None = None
    unavailable_reason: str | None = None


class FitDiagnostics(PhysicsModel):
    converged: bool
    selected_model: str
    optimizer_status: str
    iterations: int = Field(default=0, ge=0)
    inlier_frames: list[int] = Field(default_factory=list)
    outlier_frames: list[int] = Field(default_factory=list)
    weighted_reprojection_rmse_px: float | None = Field(default=None, ge=0)
    median_reprojection_error_px: float | None = Field(default=None, ge=0)
    maximum_inlier_error_px: float | None = Field(default=None, ge=0)
    parameter_bounds_reached: list[str] = Field(default_factory=list)
    processing_duration_seconds: float = Field(default=0.0, ge=0)


class DeliveryInterval(PhysicsModel):
    start_frame: int | None = Field(default=None, ge=0)
    end_frame: int | None = Field(default=None, ge=0)
    first_observed_frame: int | None = Field(default=None, ge=0)
    last_observed_frame: int | None = Field(default=None, ge=0)
    terminal_reason: str


class MetricAvailability(PhysicsModel):
    metric: str
    available: bool
    reason: str | None = None


class PhysicsPitchGeometry(PhysicsModel):
    pitch_length_m: float = PITCH_LENGTH_M
    pitch_width_m: float = PITCH_WIDTH_M


class GeometryValidationResult(PhysicsModel):
    validity: GeometryValidity
    mean_reprojection_px: float | None = Field(default=None, ge=0)
    median_reprojection_px: float | None = Field(default=None, ge=0)
    p95_reprojection_px: float | None = Field(default=None, ge=0)
    max_reprojection_px: float | None = Field(default=None, ge=0)
    in_pitch_fraction: float | None = Field(default=None, ge=0, le=1)
    threshold_px: float | None = Field(default=None, ge=0)
    reason: str | None = None


class DeliveryPhysicsResult(PhysicsModel):
    physics_engine_version: Literal["v1"] = "v1"
    status: PhysicsStatus
    analysis_id: str
    coordinate_system: str
    pitch_geometry: PhysicsPitchGeometry = Field(
        default_factory=PhysicsPitchGeometry
    )
    geometry_validation: GeometryValidationResult | None = None
    calibration: CameraCalibration
    fitted_parameters: FittedTrajectoryParameters | None = None
    trajectory_samples: list[TrajectorySample] = Field(default_factory=list)
    accepted_observations: list[ObservedBallPoint] = Field(default_factory=list)
    rejected_observations: list[RejectedObservation] = Field(default_factory=list)
    delivery_interval: DeliveryInterval
    bounce: BouncePhysicsResult
    speed: SpeedAnalytics
    overall_stump_to_stump: OverallStumpToStumpSpeed = Field(
        default_factory=OverallStumpToStumpSpeed
    )
    pre_bounce_lateral_movement: LateralMovementResult
    post_bounce_movement: PostBounceMovementResult
    line_and_length: LineLengthResult
    fit_diagnostics: FitDiagnostics
    confidence: ConfidenceGrade
    confidence_score: float = Field(ge=0, le=1)
    uncertainty_method: str
    exact_spin_rpm: None = None
    exact_spin_rpm_unavailable_reason: str = (
        "Ball surface rotation is not directly observable."
    )
    unavailable_metrics: list[MetricAvailability] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    physics_result_url: str | None = None
