"""Versioned contracts for preset-constrained automatic pitch registration."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .camera_bridge import CameraBridgeInput, DistortionMode
from .real_pitch_registration import CameraPoseCandidate, RealProjectedPitchGeometry
from .wicket_observation import SetupFrameCandidate


class PresetContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, use_enum_values=True)


class NativeVideoOrientation(str, Enum):
    PORTRAIT = "PORTRAIT"
    LANDSCAPE = "LANDSCAPE"
    SQUARE = "SQUARE"
    PORTRAIT_OR_LANDSCAPE = "PORTRAIT_OR_LANDSCAPE"


class PresetCameraEnd(str, Enum):
    BOWLER = "bowler"
    STRIKER = "striker"


class PresetImageLeftMapping(str, Enum):
    IMAGE_LEFT_IS_PITCH_LEFT = "IMAGE_LEFT_IS_PITCH_LEFT"
    IMAGE_LEFT_IS_PITCH_RIGHT = "IMAGE_LEFT_IS_PITCH_RIGHT"


class PresetDistortionPolicy(str, Enum):
    ZERO_DISTORTION = "ZERO_DISTORTION"


class PresetCompatibilityStatus(str, Enum):
    COMPATIBLE = "COMPATIBLE"
    COMPATIBLE_WITH_WARNINGS = "COMPATIBLE_WITH_WARNINGS"
    INCOMPATIBLE = "INCOMPATIBLE"


class PresetCompatibilitySeverity(str, Enum):
    WARNING = "WARNING"
    ERROR = "ERROR"


class PresetCompatibilityReasonCode(str, Enum):
    INVALID_NATIVE_DIMENSIONS = "INVALID_NATIVE_DIMENSIONS"
    UNSUPPORTED_ORIENTATION = "UNSUPPORTED_ORIENTATION"
    ASPECT_RATIO_OUT_OF_RANGE = "ASPECT_RATIO_OUT_OF_RANGE"
    UNSUPPORTED_ROTATION = "UNSUPPORTED_ROTATION"
    UNSUPPORTED_DISTORTION = "UNSUPPORTED_DISTORTION"
    CAMERA_END_MISMATCH = "CAMERA_END_MISMATCH"
    BOTH_WICKETS_REQUIRED = "BOTH_WICKETS_REQUIRED"
    SETUP_FRAME_UNAVAILABLE = "SETUP_FRAME_UNAVAILABLE"
    INSUFFICIENT_FRAME_SUPPORT = "INSUFFICIENT_FRAME_SUPPORT"
    INVALID_WICKET_OBSERVATIONS = "INVALID_WICKET_OBSERVATIONS"
    SEVERE_WICKET_CLIPPING = "SEVERE_WICKET_CLIPPING"
    NESTED_FALSE_WICKET_EVIDENCE = "NESTED_FALSE_WICKET_EVIDENCE"
    UNSUPPORTED_CROP_OR_ROTATION = "UNSUPPORTED_CROP_OR_ROTATION"


class AutoRegistrationStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    PRESET_INCOMPATIBLE = "PRESET_INCOMPATIBLE"
    INSUFFICIENT_WICKETS = "INSUFFICIENT_WICKETS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    FITTING = "FITTING"
    AUTO_REGISTRATION_READY = "AUTO_REGISTRATION_READY"
    VISUAL_OVERLAY_READY = "VISUAL_OVERLAY_READY"
    NEEDS_ASSISTANCE = "NEEDS_ASSISTANCE"
    FAILED = "FAILED"


class AutoRegistrationGeometricClassification(str, Enum):
    METRIC_3D_CANDIDATE = "METRIC_3D_CANDIDATE"
    GROUND_PLANE_CANDIDATE = "GROUND_PLANE_CANDIDATE"
    VISUAL_ONLY = "VISUAL_ONLY"
    REGISTRATION_FAILED = "REGISTRATION_FAILED"


class AutoRegistrationObservationSource(str, Enum):
    PERSISTED_WICKET_OBSERVATION_V1 = "PERSISTED_WICKET_OBSERVATION_V1"
    NEW_WICKET_OBSERVATION_V1 = "NEW_WICKET_OBSERVATION_V1"
    UNAVAILABLE = "UNAVAILABLE"


class AutoRegistrationCandidateSource(str, Enum):
    PRESET_NOMINAL = "PRESET_NOMINAL"
    PRESET_PERTURBATION = "PRESET_PERTURBATION"
    EXISTING_PNP_CANDIDATE = "EXISTING_PNP_CANDIDATE"
    EXISTING_REFINED_CANDIDATE = "EXISTING_REFINED_CANDIDATE"


class MetreBounds(PresetContractModel):
    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, use_enum_values=True, frozen=True
    )

    minimum_m: float
    maximum_m: float

    @model_validator(mode="after")
    def validate_order(self) -> "MetreBounds":
        if self.minimum_m >= self.maximum_m:
            raise ValueError("Metre bounds must be strictly increasing.")
        return self

    @property
    def minimum(self) -> float:
        return self.minimum_m

    @property
    def maximum(self) -> float:
        return self.maximum_m


class DegreeBounds(PresetContractModel):
    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, use_enum_values=True, frozen=True
    )

    minimum_deg: float
    maximum_deg: float

    @model_validator(mode="after")
    def validate_order(self) -> "DegreeBounds":
        if self.minimum_deg >= self.maximum_deg:
            raise ValueError("Degree bounds must be strictly increasing.")
        return self

    @property
    def minimum(self) -> float:
        return self.minimum_deg

    @property
    def maximum(self) -> float:
        return self.maximum_deg


class AspectRatioBounds(PresetContractModel):
    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, use_enum_values=True, frozen=True
    )

    minimum_long_edge_to_short_edge_ratio: float = Field(ge=1)
    maximum_long_edge_to_short_edge_ratio: float = Field(ge=1)

    @model_validator(mode="after")
    def validate_order(self) -> "AspectRatioBounds":
        if (
            self.minimum_long_edge_to_short_edge_ratio
            >= self.maximum_long_edge_to_short_edge_ratio
        ):
            raise ValueError("Aspect-ratio bounds must be strictly increasing.")
        return self


class CameraSetupPreset(PresetContractModel):
    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, use_enum_values=True, frozen=True
    )

    preset_id: str = Field(min_length=1)
    preset_name: str = Field(min_length=1)
    version: Literal["v1"] = "v1"
    description: str
    intended_use: str
    camera_end: PresetCameraEnd
    pitch_profile: str = Field(min_length=1)
    native_orientation: NativeVideoOrientation
    expected_aspect_ratio_range: AspectRatioBounds
    nominal_camera_height_m: float = Field(gt=0)
    camera_height_bounds_m: MetreBounds
    nominal_distance_behind_wicket_m: float = Field(gt=0)
    distance_bounds_m: MetreBounds
    nominal_lateral_offset_m: float
    lateral_offset_bounds_m: MetreBounds
    nominal_yaw_deg: float
    yaw_bounds_deg: DegreeBounds
    nominal_pitch_deg: float
    pitch_bounds_deg: DegreeBounds
    nominal_roll_deg: float
    roll_bounds_deg: DegreeBounds
    nominal_horizontal_fov_deg: float = Field(gt=1, lt=179)
    horizontal_fov_bounds_deg: DegreeBounds
    image_left_mapping: PresetImageLeftMapping
    distortion_policy: PresetDistortionPolicy
    both_wickets_required: bool
    minimum_frame_support: int = Field(ge=1)
    minimum_wicket_confidence: float = Field(ge=0, le=1)
    source: str = Field(min_length=1)
    development_only: bool
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_nominals_within_bounds(self) -> "CameraSetupPreset":
        checks = (
            ("nominal_camera_height_m", self.nominal_camera_height_m, self.camera_height_bounds_m.minimum_m, self.camera_height_bounds_m.maximum_m),
            ("nominal_distance_behind_wicket_m", self.nominal_distance_behind_wicket_m, self.distance_bounds_m.minimum_m, self.distance_bounds_m.maximum_m),
            ("nominal_lateral_offset_m", self.nominal_lateral_offset_m, self.lateral_offset_bounds_m.minimum_m, self.lateral_offset_bounds_m.maximum_m),
            ("nominal_yaw_deg", self.nominal_yaw_deg, self.yaw_bounds_deg.minimum_deg, self.yaw_bounds_deg.maximum_deg),
            ("nominal_pitch_deg", self.nominal_pitch_deg, self.pitch_bounds_deg.minimum_deg, self.pitch_bounds_deg.maximum_deg),
            ("nominal_roll_deg", self.nominal_roll_deg, self.roll_bounds_deg.minimum_deg, self.roll_bounds_deg.maximum_deg),
            ("nominal_horizontal_fov_deg", self.nominal_horizontal_fov_deg, self.horizontal_fov_bounds_deg.minimum_deg, self.horizontal_fov_bounds_deg.maximum_deg),
        )
        outside = [name for name, value, low, high in checks if not low <= value <= high]
        if outside:
            raise ValueError(
                "Preset nominal values must lie within their bounds: "
                + ", ".join(outside)
            )
        return self


class PresetCompatibilityReason(PresetContractModel):
    reason_code: PresetCompatibilityReasonCode
    severity: PresetCompatibilitySeverity
    message: str


class PresetCompatibilityInput(PresetContractModel):
    native_video_width_px: int = Field(gt=0)
    native_video_height_px: int = Field(gt=0)
    native_orientation: NativeVideoOrientation
    rotation_metadata_deg: int | None = None
    distortion_mode: DistortionMode
    camera_end: PresetCameraEnd | None = None
    near_wicket_available: bool
    far_wicket_available: bool
    setup_frame_available: bool
    supporting_frame_count: int = Field(ge=0)
    minimum_observed_wicket_confidence: float | None = Field(
        default=None, ge=0, le=1
    )
    wicket_observations_valid: bool
    severe_clipping_detected: bool
    nested_false_wicket_evidence_detected: bool
    unsupported_crop_or_rotation_detected: bool

    @model_validator(mode="after")
    def validate_native_orientation(self) -> "PresetCompatibilityInput":
        if self.native_orientation == "PORTRAIT_OR_LANDSCAPE":
            raise ValueError("Native orientation must describe one input video.")
        return self


class PresetCompatibility(PresetContractModel):
    status: PresetCompatibilityStatus
    native_video_width_px: int = Field(gt=0)
    native_video_height_px: int = Field(gt=0)
    detected_orientation: NativeVideoOrientation
    long_edge_to_short_edge_aspect_ratio: float = Field(ge=1)
    rotation_metadata_deg: int | None = None
    distortion_mode: DistortionMode
    camera_end: PresetCameraEnd | None = None
    both_wickets_present: bool
    setup_frame_available: bool
    supporting_frame_count: int = Field(ge=0)
    wicket_observations_valid: bool
    severe_clipping_detected: bool
    nested_false_wicket_evidence_detected: bool
    unsupported_crop_or_rotation_detected: bool
    reasons: list[PresetCompatibilityReason] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status_reasons(self) -> "PresetCompatibility":
        if self.detected_orientation == "PORTRAIT_OR_LANDSCAPE":
            raise ValueError("Detected orientation must describe one native video.")
        has_error = any(reason.severity == "ERROR" for reason in self.reasons)
        has_warning = any(reason.severity == "WARNING" for reason in self.reasons)
        if self.status == "INCOMPATIBLE" and not has_error:
            raise ValueError("INCOMPATIBLE compatibility requires an ERROR reason.")
        if self.status == "COMPATIBLE" and (has_error or has_warning):
            raise ValueError("COMPATIBLE compatibility cannot contain reasons.")
        if self.status == "COMPATIBLE_WITH_WARNINGS" and (has_error or not has_warning):
            raise ValueError("COMPATIBLE_WITH_WARNINGS requires warnings and no errors.")
        return self


class PresetAutoRegistrationRunRequest(PresetContractModel):
    preset_id: str = Field(min_length=1)
    reuse_existing_observations: bool = True
    force_redetect: bool = False
    development_diagnostics: bool = False


class CameraSetupPresetListResponse(PresetContractModel):
    preset_contract_version: Literal["v1"] = "v1"
    presets: list[CameraSetupPreset]


class AutoRegistrationParameters(PresetContractModel):
    camera_height_m: float = Field(gt=0)
    distance_behind_wicket_m: float = Field(gt=0)
    lateral_offset_m: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    horizontal_fov_deg: float = Field(gt=1, lt=179)
    principal_point_offset_x_px: float = 0.0
    principal_point_offset_y_px: float = 0.0


class AutoRegistrationParameterChange(PresetContractModel):
    parameter_name: str
    unit: Literal["m", "deg", "px"]
    initial_value: float
    fitted_value: float
    delta: float


class AutoRegistrationActiveBound(PresetContractModel):
    parameter_name: str
    bound: Literal["MINIMUM", "MAXIMUM"]
    value: float
    unit: Literal["m", "deg", "px"]
    critical: bool


class AutoRegistrationCandidateAttempt(PresetContractModel):
    candidate_id: str
    source: AutoRegistrationCandidateSource
    deterministic_order: int = Field(ge=0)
    initial_parameters: AutoRegistrationParameters
    attempted: bool
    converged: bool
    eligible_for_selection: bool
    robust_loss: str | None = None
    final_cost: float | None = Field(default=None, ge=0)
    score: float | None = Field(default=None, ge=0, le=1)
    rejection_reasons: list[str] = Field(default_factory=list)


class AutoRegistrationAnchorMetrics(PresetContractModel):
    exact_anchor_count: int = Field(ge=0)
    pointlike_anchor_count: int = Field(ge=0)
    soft_constraint_count: int = Field(ge=0)
    inlier_count: int = Field(ge=0)
    outlier_count: int = Field(ge=0)
    reprojection_rmse_px: float | None = Field(default=None, ge=0)
    median_reprojection_error_px: float | None = Field(default=None, ge=0)
    maximum_inlier_error_px: float | None = Field(default=None, ge=0)


class AutoRegistrationEnvelopeMetrics(PresetContractModel):
    near_wicket_iou: float | None = Field(default=None, ge=0, le=1)
    far_wicket_iou: float | None = Field(default=None, ge=0, le=1)
    near_centre_residual_px: float | None = Field(default=None, ge=0)
    far_centre_residual_px: float | None = Field(default=None, ge=0)
    near_width_residual_px: float | None = Field(default=None, ge=0)
    far_width_residual_px: float | None = Field(default=None, ge=0)
    near_height_residual_px: float | None = Field(default=None, ge=0)
    far_height_residual_px: float | None = Field(default=None, ge=0)


class AutoRegistrationTemporalMetrics(PresetContractModel):
    frame_count: int = Field(ge=0)
    successful_frame_count: int = Field(ge=0)
    median_near_wicket_iou: float | None = Field(default=None, ge=0, le=1)
    median_far_wicket_iou: float | None = Field(default=None, ge=0, le=1)
    median_centre_residual_px: float | None = Field(default=None, ge=0)
    median_width_residual_px: float | None = Field(default=None, ge=0)
    median_height_residual_px: float | None = Field(default=None, ge=0)
    scale_consistency_score: float | None = Field(default=None, ge=0, le=1)
    temporal_stability_score: float = Field(ge=0, le=1)
    worst_supporting_frame_index: int | None = Field(default=None, ge=0)


class AutoRegistrationPhysicalCheck(PresetContractModel):
    check_id: str
    passed: bool
    value: float | str | bool | None = None
    reason: str


class AutoRegistrationUncertainty(PresetContractModel):
    perturbation_count: int = Field(ge=0)
    deterministic_seed: int
    camera_position_spread_m: float | None = Field(default=None, ge=0)
    camera_rotation_spread_deg: float | None = Field(default=None, ge=0)
    horizontal_fov_spread_deg: float | None = Field(default=None, ge=0)
    projected_wicket_movement_px: float | None = Field(default=None, ge=0)
    projected_pitch_corner_movement_px: float | None = Field(default=None, ge=0)
    projected_bounce_location_sensitivity_px: float | None = Field(default=None, ge=0)
    candidate_ordering_stability: float | None = Field(default=None, ge=0, le=1)
    stable: bool
    warnings: list[str] = Field(default_factory=list)


class AutoRegistrationAmbiguity(PresetContractModel):
    score: float = Field(ge=0, le=1)
    competing_solution_plausible: bool
    selected_candidate_id: str | None = None
    competing_candidate_id: str | None = None
    reasons: list[str] = Field(default_factory=list)


class AutoRegistrationStageTimings(PresetContractModel):
    observation_load_ms: float | None = Field(default=None, ge=0)
    candidate_generation_ms: float | None = Field(default=None, ge=0)
    optimisation_ms: float | None = Field(default=None, ge=0)
    temporal_validation_ms: float | None = Field(default=None, ge=0)
    total_ms: float | None = Field(default=None, ge=0)


class PresetAutoRegistrationResult(PresetContractModel):
    preset_auto_registration_version: Literal["v1"] = "v1"
    analysis_id: str
    status: AutoRegistrationStatus
    geometric_classification: AutoRegistrationGeometricClassification
    preset: CameraSetupPreset
    preset_compatibility: PresetCompatibility
    setup_frame: SetupFrameCandidate | None = None
    supporting_frames: list[SetupFrameCandidate] = Field(default_factory=list)
    observation_source: AutoRegistrationObservationSource
    detection_reused: bool
    candidates_attempted: list[AutoRegistrationCandidateAttempt] = Field(default_factory=list)
    selected_candidate: CameraPoseCandidate | None = None
    competing_candidate: CameraPoseCandidate | None = None
    fitted_parameters: AutoRegistrationParameters | None = None
    initial_parameters: AutoRegistrationParameters | None = None
    parameter_changes: list[AutoRegistrationParameterChange] = Field(default_factory=list)
    active_bounds: list[AutoRegistrationActiveBound] = Field(default_factory=list)
    anchor_metrics: AutoRegistrationAnchorMetrics | None = None
    envelope_metrics: AutoRegistrationEnvelopeMetrics | None = None
    temporal_metrics: AutoRegistrationTemporalMetrics | None = None
    physical_checks: list[AutoRegistrationPhysicalCheck] = Field(default_factory=list)
    uncertainty: AutoRegistrationUncertainty | None = None
    ambiguity: AutoRegistrationAmbiguity | None = None
    projected_pitch: RealProjectedPitchGeometry | None = None
    bridge_camera: CameraBridgeInput | None = None
    stage_timings: AutoRegistrationStageTimings = Field(default_factory=AutoRegistrationStageTimings)
    warnings: list[str] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)
    manual_assistance_available: bool
    production_accepted: Literal[False] = False
    metrics_unlocked: list[str] = Field(default_factory=list, max_length=0)

    @model_validator(mode="after")
    def validate_result_invariants(self) -> "PresetAutoRegistrationResult":
        if self.status == "PRESET_INCOMPATIBLE" and self.preset_compatibility.status != "INCOMPATIBLE":
            raise ValueError("PRESET_INCOMPATIBLE requires incompatible preset compatibility.")
        if self.status in {"AUTO_REGISTRATION_READY", "VISUAL_OVERLAY_READY"}:
            required = {
                "selected_candidate": self.selected_candidate,
                "fitted_parameters": self.fitted_parameters,
                "projected_pitch": self.projected_pitch,
                "bridge_camera": self.bridge_camera,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError("Ready automatic registration is missing: " + ", ".join(missing))
        return self


STANDARD_REAR_WICKET_NET_V1 = CameraSetupPreset(
    preset_id="STANDARD_REAR_WICKET_NET_V1",
    preset_name="Standard rear-wicket practice net",
    description="Development preset for a fixed tripod behind the bowler-end wicket with both wicket sets visible.",
    intended_use="Constrained FullTrack-style practice-net video in portrait or landscape native orientation.",
    camera_end=PresetCameraEnd.BOWLER,
    pitch_profile="official_core",
    native_orientation=NativeVideoOrientation.PORTRAIT_OR_LANDSCAPE,
    expected_aspect_ratio_range=AspectRatioBounds(
        minimum_long_edge_to_short_edge_ratio=1.25,
        maximum_long_edge_to_short_edge_ratio=2.25,
    ),
    nominal_camera_height_m=1.5,
    camera_height_bounds_m=MetreBounds(minimum_m=0.75, maximum_m=3.0),
    nominal_distance_behind_wicket_m=8.0,
    distance_bounds_m=MetreBounds(minimum_m=3.0, maximum_m=15.0),
    nominal_lateral_offset_m=0.0,
    lateral_offset_bounds_m=MetreBounds(minimum_m=-2.5, maximum_m=2.5),
    nominal_yaw_deg=0.0,
    yaw_bounds_deg=DegreeBounds(minimum_deg=-15.0, maximum_deg=15.0),
    nominal_pitch_deg=-4.0,
    pitch_bounds_deg=DegreeBounds(minimum_deg=-18.0, maximum_deg=8.0),
    nominal_roll_deg=0.0,
    roll_bounds_deg=DegreeBounds(minimum_deg=-8.0, maximum_deg=8.0),
    nominal_horizontal_fov_deg=45.0,
    horizontal_fov_bounds_deg=DegreeBounds(minimum_deg=25.0, maximum_deg=80.0),
    image_left_mapping=PresetImageLeftMapping.IMAGE_LEFT_IS_PITCH_LEFT,
    distortion_policy=PresetDistortionPolicy.ZERO_DISTORTION,
    both_wickets_required=True,
    minimum_frame_support=3,
    minimum_wicket_confidence=0.35,
    source="CricVision development assumption informed by the existing strongest rear-wicket clip; not measured camera metadata.",
    development_only=True,
    warnings=[
        "Camera height, distance, lateral offset, orientation, and FOV are bounded priors, not measured facts.",
        "The preset does not compensate for arbitrary crops, rotations, or nonzero lens distortion.",
        "Automatic readiness does not accept calibration or unlock metric analytics.",
    ],
)


CAMERA_SETUP_PRESETS_BY_ID: Mapping[str, CameraSetupPreset] = MappingProxyType(
    {STANDARD_REAR_WICKET_NET_V1.preset_id: STANDARD_REAR_WICKET_NET_V1}
)

# Canonical public catalogue name consumed by orchestration and API layers.
CAMERA_SETUP_PRESETS = CAMERA_SETUP_PRESETS_BY_ID


def get_camera_setup_preset(preset_id: str) -> CameraSetupPreset | None:
    preset = CAMERA_SETUP_PRESETS_BY_ID.get(preset_id)
    return preset.model_copy(deep=True) if preset is not None else None


def list_camera_setup_presets() -> list[CameraSetupPreset]:
    return [
        CAMERA_SETUP_PRESETS_BY_ID[preset_id].model_copy(deep=True)
        for preset_id in sorted(CAMERA_SETUP_PRESETS_BY_ID)
    ]
