from datetime import datetime
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from packages.cricket_vision.calibration.cricket_pitch_geometry import (
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    POPPING_CREASE_OFFSET_M,
    STUMP_DIAMETER_MAX_M,
    STUMP_HEIGHT_M,
    WICKET_WIDTH_M,
    CricketPitchDimensions,
)

from .delivery_physics import DeliveryPhysicsResult


class StrictGeometryModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)


class NormalizedPoint(StrictGeometryModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class NormalizedBox(StrictGeometryModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> "NormalizedBox":
        values = (self.x, self.y, self.width, self.height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Wicket box coordinates must be finite.")
        if self.x + self.width > 1.000001 or self.y + self.height > 1.000001:
            raise ValueError("Wicket box must stay inside the reference image.")
        return self


WicketDetectionPass = Literal["full_frame", "far_roi", "near_roi", "guide_roi"]


class WicketCandidate(StrictGeometryModel):
    candidate_id: str
    confidence: float = Field(ge=0, le=1)
    class_name: str
    box: NormalizedBox
    center: NormalizedPoint
    bottom_center: NormalizedPoint
    detection_pass: WicketDetectionPass | None = None


class WicketCalibration(StrictGeometryModel):
    label: Literal["striker", "non_striker"]
    source: Literal["detected", "adjusted", "manual"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    box: NormalizedBox
    center: NormalizedPoint
    bottom_center: NormalizedPoint
    # Honest alias for the approximate bbox bottom-centre used as a soft base.
    approximate_wicket_base_reference: NormalizedPoint | None = None
    # Which robust detector pass produced this box (not user edit source).
    detection_pass: WicketDetectionPass | None = None


class WicketCalibrationInput(StrictGeometryModel):
    label: Literal["striker", "non_striker"]
    source: Literal["detected", "adjusted", "manual"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    box: NormalizedBox
    detection_pass: WicketDetectionPass | None = None


class PitchGeometry(StrictGeometryModel):
    axis_start: NormalizedPoint
    axis_end: NormalizedPoint
    corridor: list[NormalizedPoint] = Field(min_length=4, max_length=4)
    near_end_label: Literal["striker", "non_striker"]
    far_end_label: Literal["striker", "non_striker"]
    geometry_type: Literal["approximate_2d"]
    corridor_width_multiplier: float = Field(ge=0.7, le=1.5)


VisualCalibrationQuality = Literal["READY", "WEAK", "FAILED"]
VisualCalibrationMode = Literal["automatic_visual"]


class VideoAnalysisPreparedResponse(BaseModel):
    success: bool
    analysis_id: str
    status: Literal["prepared", "calibrated"]
    original_filename: str
    stored_filename: str
    file_size_bytes: int
    created_at: datetime
    updated_at: datetime | None = None
    duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int
    codec: str | None = None
    reference_frame_index: int
    reference_frame_selection: dict[str, object] | None = None
    original_video_url: str
    reference_frame_url: str
    calibration_status: Literal["confirmed"] | None = None
    calibration_url: str | None = None
    calibration_overlay_url: str | None = None
    visual_calibration_quality: VisualCalibrationQuality | None = None
    visual_calibration_mode: VisualCalibrationMode | None = None
    calibration_v2_status: Literal[
        "confirmed",
        "ready",
        "weak",
        "unstable",
        "insufficient_geometry",
    ] | None = None
    calibration_v2_url: str | None = None
    calibration_v2_overlay_url: str | None = None
    calibration_v2_quality_grade: Literal[
        "excellent",
        "good",
        "usable",
        "weak",
        "poor",
        "insufficient_geometry",
    ] | None = None
    calibration_v2_reprojection_rmse_px: float | None = None
    camera_pose_status: Literal[
        "ready",
        "usable",
        "weak",
        "unstable",
        "insufficient_landmarks",
        "solver_failed",
        "implausible_pose",
    ] | None = None
    camera_pose_quality: float | None = Field(default=None, ge=0, le=1)
    camera_pose_url: str | None = None
    camera_pose_overlay_url: str | None = None
    camera_intrinsics_source: Literal[
        "calibrated_device_profile",
        "metadata_estimated",
        "heuristic_estimated",
        "manually_provided",
    ] | None = None
    camera_pose_reprojection_rmse_px: float | None = Field(
        default=None,
        ge=0,
    )
    calibration_mode_used: Literal[
        "ground_plane",
        "wicket_camera_pose",
    ] | None = None
    ball_detection_status: Literal[
        "detection_queued",
        "detecting_ball",
        "detection_complete",
        "detection_failed",
    ] | None = None
    ball_detection_job_id: str | None = None
    ball_detection_started_at: datetime | None = None
    ball_detection_completed_at: datetime | None = None
    detection_summary_url: str | None = None
    detection_overlay_url: str | None = None
    tracking_status: Literal[
        "tracking_queued",
        "tracking_ball",
        "tracking_complete",
        "tracking_failed",
        "tracking_no_reliable_track",
    ] | None = None
    tracking_job_id: str | None = None
    tracking_started_at: datetime | None = None
    tracking_completed_at: datetime | None = None
    tracking_summary_url: str | None = None
    tracking_video_url: str | None = None
    message: str


class VisualCalibrationDetectionDebug(BaseModel):
    """Developer-only diagnostics for robust two-wicket detection."""

    pass_count: int = 0
    passes: list[dict[str, object]] = Field(default_factory=list)
    rejected: list[dict[str, object]] = Field(default_factory=list)
    rois: dict[str, NormalizedBox] = Field(default_factory=dict)
    selected: dict[str, object] | None = None
    debug_overlay_url: str | None = None
    debug_json_url: str | None = None


class VideoCalibrationDetectionRequest(BaseModel):
    """Optional guided search regions for two-wicket detection."""

    striker_guide: NormalizedBox | None = None
    non_striker_guide: NormalizedBox | None = None


class VideoCalibrationDetectionResponse(BaseModel):
    success: bool
    status: Literal[
        "candidates_ready",
        "manual_required",
        "detection_incomplete",
        "stump_detector_missing",
        "stump_detector_error",
    ]
    analysis_id: str
    reference_frame_index: int
    reference_frame_url: str
    image_width: int
    image_height: int
    candidates: list[WicketCandidate] = Field(default_factory=list)
    provisional_striker_wicket: WicketCalibration | None = None
    provisional_non_striker_wicket: WicketCalibration | None = None
    pitch_geometry: PitchGeometry | None = None
    striker_guide: NormalizedBox | None = None
    non_striker_guide: NormalizedBox | None = None
    failed_ends: list[Literal["striker", "non_striker"]] = Field(
        default_factory=list
    )
    model_path_used: str
    mode: VisualCalibrationMode = "automatic_visual"
    quality: VisualCalibrationQuality = "FAILED"
    quality_reasons: list[str] = Field(default_factory=list)
    assignment_warning: str | None = None
    warning: str | None = None
    message: str
    detection_debug: VisualCalibrationDetectionDebug | None = None


class VideoCalibrationConfirmationRequest(BaseModel):
    analysis_id: str
    striker_wicket: WicketCalibrationInput
    non_striker_wicket: WicketCalibrationInput
    corridor_width_multiplier: float = Field(default=1.0, ge=0.7, le=1.5)
    user_note: str | None = Field(default=None, max_length=1000)
    # ponytail: optional for older clients; guided UI always sends both.
    striker_guide: NormalizedBox | None = None
    non_striker_guide: NormalizedBox | None = None


class ConfirmedVideoCalibrationResponse(BaseModel):
    success: bool
    status: Literal["calibrated"]
    analysis_id: str
    created_at: datetime
    updated_at: datetime
    reference_frame_index: int
    reference_frame_url: str
    calibration_url: str
    calibration_overlay_url: str
    scene_overlay_url: str | None = None
    scene_overlay_status: Literal["ready", "failed", "skipped"] | None = None
    image_width: int
    image_height: int
    model_path_used: str | None = None
    mode: VisualCalibrationMode = "automatic_visual"
    quality: VisualCalibrationQuality = "READY"
    quality_reasons: list[str] = Field(default_factory=list)
    assignment_warning: str | None = None
    striker_wicket: WicketCalibration
    non_striker_wicket: WicketCalibration
    pitch_geometry: PitchGeometry
    striker_guide: NormalizedBox | None = None
    non_striker_guide: NormalizedBox | None = None
    user_note: str | None = None
    message: str


CalibrationLandmarkSource = Literal[
    "detected",
    "inferred",
    "manually_adjusted",
    "manual",
]


class CricketPitchGeometry(StrictGeometryModel):
    pitch_length_m: float = Field(default=PITCH_LENGTH_M, gt=0, le=40)
    wicket_width_m: float = Field(default=WICKET_WIDTH_M, gt=0, le=1)
    wicket_height_m: float = Field(default=STUMP_HEIGHT_M, gt=0, le=2)
    stump_diameter_m: float = Field(
        default=STUMP_DIAMETER_MAX_M,
        gt=0,
        le=0.1,
    )
    pitch_width_m: float = Field(default=PITCH_WIDTH_M, gt=0, le=10)
    popping_crease_distance_m: float = Field(
        default=POPPING_CREASE_OFFSET_M,
        gt=0,
        le=5,
    )

    @model_validator(mode="after")
    def validate_pitch_dimensions(self) -> "CricketPitchGeometry":
        if self.pitch_width_m < self.wicket_width_m:
            raise ValueError("Pitch width must exceed wicket width.")
        if self.stump_diameter_m >= self.wicket_width_m / 2:
            raise ValueError("Stump diameter is invalid for the wicket width.")
        if self.popping_crease_distance_m >= self.pitch_length_m / 2:
            raise ValueError("Popping crease distance is invalid.")
        return self

    @computed_field
    @property
    def stump_lateral_positions_m(self) -> dict[str, float]:
        outer_centre = (self.wicket_width_m - self.stump_diameter_m) / 2
        return {
            "left": -outer_centre,
            "middle": 0.0,
            "right": outer_centre,
        }

    def to_dimensions(self) -> CricketPitchDimensions:
        """Adapt the wire model onto the geometry dataclass the solver takes.

        Crease lengths are not declarable over the wire, so they keep their
        regulation defaults.
        """
        return CricketPitchDimensions(
            pitch_length_m=self.pitch_length_m,
            wicket_width_m=self.wicket_width_m,
            wicket_height_m=self.wicket_height_m,
            stump_diameter_m=self.stump_diameter_m,
            pitch_width_m=self.pitch_width_m,
            popping_crease_distance_m=self.popping_crease_distance_m,
        )


class CalibrationCoordinateSystem(BaseModel):
    units: Literal["metres"]
    origin: Literal["bowler_wicket_centre"]
    x_axis: Literal["toward_striker"]
    y_axis: Literal["lateral"]
    z_axis: Literal["up"]
    left_right_convention: str
    image_left_right_convention: Literal[
        "image_left_is_world_left",
        "image_left_is_world_right",
    ]


class CalibrationLandmarkInput(StrictGeometryModel):
    id: str = Field(min_length=1, max_length=100)
    label: str = Field(min_length=1, max_length=120)
    wicket_end: Literal["bowler", "striker", "ground"]
    landmark_type: Literal["stump_base", "ground_control"]
    normalized_x: float = Field(ge=0, le=1)
    normalized_y: float = Field(ge=0, le=1)
    source: CalibrationLandmarkSource
    confidence: float | None = Field(default=None, ge=0, le=1)
    world_x_m: float | None = None
    world_y_m: float | None = None
    world_z_m: float | None = None


class CalibrationLandmark(CalibrationLandmarkInput):
    pixel_x: float = Field(ge=0)
    pixel_y: float = Field(ge=0)
    world_x_m: float
    world_y_m: float
    world_z_m: float


class CalibrationLandmarkSet(BaseModel):
    primary_stump_bases: list[CalibrationLandmark] = Field(
        min_length=6,
        max_length=6,
    )
    optional_ground_landmarks: list[CalibrationLandmark] = Field(
        default_factory=list,
        max_length=32,
    )


class CalibrationV2ConfirmRequest(BaseModel):
    analysis_id: str
    landmarks: list[CalibrationLandmarkInput] = Field(
        min_length=6,
        max_length=38,
    )
    pitch_geometry: CricketPitchGeometry
    image_left_right_convention: Literal[
        "image_left_is_world_left",
        "image_left_is_world_right",
    ]
    landmark_semantics_confirmed: bool = False
    ground_reference_mode: Literal["use", "skip"] = "use"
    user_note: str | None = Field(default=None, max_length=1000)


class ReprojectionDiagnostic(StrictGeometryModel):
    landmark_id: str
    landmark_source: CalibrationLandmarkSource = "manual"
    used_for_homography: bool = True
    ransac_inlier: bool | None = None
    observed_pixel_x: float = Field(ge=0)
    observed_pixel_y: float = Field(ge=0)
    reprojected_pixel_x: float
    reprojected_pixel_y: float
    error_px: float = Field(ge=0)


class GroundHomographyResult(StrictGeometryModel):
    transform_available: bool
    image_to_ground_homography: list[list[float]] | None = None
    ground_to_image_homography: list[list[float]] | None = None
    determinant: float | None = None
    condition_number: float | None = Field(default=None, ge=0)
    estimation_method: Literal["none", "direct", "ransac"] = "none"
    ransac_reprojection_threshold_px: float | None = Field(
        default=None,
        gt=0,
    )
    ransac_inlier_count: int | None = Field(default=None, ge=0)
    ransac_inlier_landmark_ids: list[str] = Field(default_factory=list)
    round_trip_image_rmse_px: float | None = Field(default=None, ge=0)
    round_trip_ground_rmse_m: float | None = Field(default=None, ge=0)
    image_convention: Literal["pixel_uv"]
    ground_convention: Literal["pitch_xy_metres_z0"]


class CalibrationQualityV2(StrictGeometryModel):
    landmark_coverage: float = Field(ge=0, le=1)
    usable_landmarks: int = Field(ge=0)
    metric_correspondence_count: int = Field(default=0, ge=0)
    additional_metric_ground_landmark_count: int = Field(default=0, ge=0)
    landmark_spread_score: float = Field(default=0, ge=0, le=1)
    world_coverage: float = Field(default=0, ge=0, le=1)
    reprojection_rmse_px: float | None = Field(default=None, ge=0)
    max_reprojection_error_px: float | None = Field(default=None, ge=0)
    median_reprojection_error_px: float | None = Field(default=None, ge=0)
    normalized_reprojection_rmse: float | None = Field(default=None, ge=0)
    geometry_condition: Literal[
        "well_conditioned",
        "weak",
        "unstable",
        "insufficient",
    ]
    homography_condition_number: float | None = Field(default=None, ge=0)
    image_coverage: float = Field(ge=0, le=1)
    wicket_order_valid: bool
    transform_available: bool
    full_pitch_projection_allowed: bool = False
    projection_outside_fraction: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    manual_adjustment_count: int = Field(ge=0)
    used_landmark_ids: list[str] = Field(default_factory=list)
    ignored_landmark_ids: list[str] = Field(default_factory=list)
    landmark_sources: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    quality_grade: Literal[
        "excellent",
        "good",
        "usable",
        "weak",
        "poor",
        "insufficient_geometry",
    ]
    overall_confidence: float = Field(ge=0, le=1)
    reprojection_diagnostics: list[ReprojectionDiagnostic] = Field(
        default_factory=list
    )


class GroundPoint2D(StrictGeometryModel):
    x_m: float
    y_m: float


class ImagePixelPoint(StrictGeometryModel):
    x: float
    y: float


class ProjectedPitchLine(BaseModel):
    id: str
    label: str
    ground_points: list[GroundPoint2D] = Field(min_length=2)
    image_points: list[ImagePixelPoint] = Field(min_length=2)


class VirtualPitchOverlayGeometry(BaseModel):
    projected_lines: list[ProjectedPitchLine] = Field(default_factory=list)
    projection_mode: Literal[
        "full_pitch",
        "local_debug",
        "landmarks_only",
    ] = "landmarks_only"


class CalibrationV2FutureCameraPoseFields(BaseModel):
    camera_intrinsics: list[list[float]] | None = None
    distortion_coefficients: list[float] | None = None
    camera_rotation: list[list[float]] | None = None
    camera_translation: list[float] | None = None
    camera_position_world: list[float] | None = None
    solvepnp_reprojection_rmse: float | None = None
    stump_top_landmarks: list[CalibrationLandmark] | None = None


class CalibrationV2InitialiseResponse(BaseModel):
    success: Literal[True]
    status: Literal["initialised"]
    analysis_id: str
    reference_frame_url: str
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    pitch_geometry: CricketPitchGeometry
    landmarks: list[CalibrationLandmark] = Field(min_length=6, max_length=6)
    image_left_right_convention: Literal[
        "image_left_is_world_left",
        "image_left_is_world_right",
    ]
    warnings: list[str] = Field(default_factory=list)
    message: str


class CalibrationV2Result(BaseModel):
    success: bool
    status: Literal[
        "confirmed",
        "ready",
        "weak",
        "unstable",
        "insufficient_geometry",
    ]
    schema_version: Literal["2.0", "2.1"]
    analysis_id: str
    calibration_mode: Literal["ground_plane"]
    coordinate_system: CalibrationCoordinateSystem
    pitch_geometry: CricketPitchGeometry
    landmark_set: CalibrationLandmarkSet
    homography: GroundHomographyResult
    quality: CalibrationQualityV2
    virtual_pitch_overlay_geometry: VirtualPitchOverlayGeometry
    calibration_v2_url: str
    calibration_v2_overlay_url: str
    reference_frame_url: str
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    landmark_semantics_confirmed: bool = False
    ground_reference_mode: Literal["use", "skip"] = "use"
    ground_transform_reason: str | None = None
    created_at: datetime
    updated_at: datetime
    user_note: str | None = None
    future_camera_pose: CalibrationV2FutureCameraPoseFields
    message: str


WicketLandmarkVisibility = Literal[
    "visible",
    "uncertain",
    "occluded",
    "unavailable",
]
CameraIntrinsicsSource = Literal[
    "calibrated_device_profile",
    "metadata_estimated",
    "heuristic_estimated",
    "manually_provided",
]
CameraPoseStatus = Literal[
    "ready",
    "usable",
    "weak",
    "unstable",
    "insufficient_landmarks",
    "solver_failed",
    "implausible_pose",
]


class WicketPoseLandmarkInput(StrictGeometryModel):
    id: str = Field(min_length=1, max_length=100)
    label: str = Field(min_length=1, max_length=120)
    wicket_end: Literal["bowler", "striker"]
    stump_position: Literal["left", "middle", "right"]
    point_type: Literal["base", "top"]
    normalized_x: float = Field(ge=0, le=1)
    normalized_y: float = Field(ge=0, le=1)
    source: CalibrationLandmarkSource
    confidence: float | None = Field(default=None, ge=0, le=1)
    visibility: WicketLandmarkVisibility = "visible"


class WicketPoseLandmark(WicketPoseLandmarkInput):
    pixel_x: float = Field(ge=0)
    pixel_y: float = Field(ge=0)
    world_x_m: float
    world_y_m: float
    world_z_m: float


class CameraIntrinsics(StrictGeometryModel):
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float
    cy: float
    intrinsic_matrix: list[list[float]]
    distortion_coefficients: list[float] = Field(min_length=4, max_length=14)
    source: CameraIntrinsicsSource
    quality: Literal["calibrated", "estimated", "low"]
    device_profile_id: str | None = None
    camera_model: str | None = None
    lens_mode: str | None = None
    resolution_label: str | None = None
    assumed_horizontal_fov_degrees: float | None = Field(
        default=None,
        gt=1,
        lt=179,
    )
    distortion_model_source: Literal["calibrated", "not_calibrated"]
    assumptions: list[str] = Field(default_factory=list)


class CameraPoseReprojectionDiagnostic(StrictGeometryModel):
    landmark_id: str
    observed_pixel_x: float = Field(ge=0)
    observed_pixel_y: float = Field(ge=0)
    projected_pixel_x: float
    projected_pixel_y: float
    residual_px: float = Field(ge=0)
    camera_depth_m: float
    ransac_inlier: bool


class CameraPoseQualityComponents(StrictGeometryModel):
    landmark_quality: float = Field(ge=0, le=1)
    landmark_coverage: float = Field(ge=0, le=1)
    reprojection_quality: float = Field(ge=0, le=1)
    intrinsics_quality: float = Field(ge=0, le=1)
    geometry_condition: float = Field(ge=0, le=1)
    pose_plausibility: float = Field(ge=0, le=1)
    overall_pose_quality: float = Field(ge=0, le=1)


class CameraPoseSolution(StrictGeometryModel):
    solved: bool
    accepted: bool
    solver_method: str
    refinement_method: str | None = None
    rotation_vector: list[float] | None = None
    rotation_matrix: list[list[float]] | None = None
    translation_vector: list[float] | None = None
    camera_position_world: list[float] | None = None
    camera_forward_direction_world: list[float] | None = None
    camera_height_m: float | None = None
    landmark_count: int = Field(ge=0)
    used_landmark_ids: list[str] = Field(default_factory=list)
    unavailable_landmark_ids: list[str] = Field(default_factory=list)
    ransac_inlier_ids: list[str] = Field(default_factory=list)
    ransac_outlier_ids: list[str] = Field(default_factory=list)
    reprojection_rmse_px: float | None = Field(default=None, ge=0)
    reprojection_median_px: float | None = Field(default=None, ge=0)
    reprojection_max_px: float | None = Field(default=None, ge=0)
    normalized_reprojection_rmse: float | None = Field(default=None, ge=0)
    reprojection_diagnostics: list[CameraPoseReprojectionDiagnostic] = Field(
        default_factory=list
    )
    positive_depth_for_all_used_landmarks: bool | None = None
    both_wickets_in_front: bool | None = None
    camera_faces_pitch: bool | None = None
    wicket_order_plausible: bool | None = None
    warnings: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)


class WicketCameraPoseInitialiseResponse(BaseModel):
    success: Literal[True]
    status: Literal["initialised"]
    analysis_id: str
    reference_frame_url: str
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    pitch_geometry: CricketPitchGeometry
    landmarks: list[WicketPoseLandmark] = Field(min_length=12, max_length=12)
    camera_intrinsics: CameraIntrinsics
    warnings: list[str] = Field(default_factory=list)
    message: str


class WicketCameraPoseSolveRequest(BaseModel):
    analysis_id: str
    landmarks: list[WicketPoseLandmarkInput] = Field(
        min_length=12,
        max_length=12,
    )
    pitch_geometry: CricketPitchGeometry
    camera_intrinsics: CameraIntrinsics
    landmark_semantics_confirmed: bool = False
    user_note: str | None = Field(default=None, max_length=1000)


class WicketCameraPoseResult(BaseModel):
    success: bool
    status: CameraPoseStatus
    schema_version: Literal["2.2"]
    analysis_id: str
    calibration_mode: Literal["wicket_camera_pose"]
    coordinate_system: CalibrationCoordinateSystem
    pitch_geometry: CricketPitchGeometry
    stump_top_definition: Literal["top_of_stump_body_excluding_bails"]
    landmarks: list[WicketPoseLandmark] = Field(min_length=12, max_length=12)
    camera_intrinsics: CameraIntrinsics
    camera_pose: CameraPoseSolution
    quality: CameraPoseQualityComponents
    camera_pose_url: str
    camera_pose_overlay_url: str
    reference_frame_url: str
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    landmark_semantics_confirmed: bool = False
    created_at: datetime
    updated_at: datetime
    user_note: str | None = None
    message: str


VideoBallDetectionJobStatus = Literal[
    "queued",
    "loading_model",
    "processing",
    "writing_video",
    "saving_results",
    "ready",
    "failed",
    "ball_detector_missing",
]


class VideoBallDetectionResultLinks(BaseModel):
    processed_video_url: str
    detections_json_url: str
    detections_csv_url: str
    detection_summary_url: str


class BallDetectorModelOption(BaseModel):
    key: str
    display_name: str
    description: str
    available: bool


class BallDetectorModelsResponse(BaseModel):
    models: list[BallDetectorModelOption]
    default_key: Literal["automatic"] = "automatic"


class VideoBallDetectionStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ball_detector_model_key: str = "automatic"


class VideoBallDetectionStartResponse(BaseModel):
    success: bool
    status: Literal["queued"]
    analysis_id: str
    job_id: str
    progress: int = Field(ge=0, le=100)
    current_frame: int = Field(ge=0)
    total_frames: int = Field(gt=0)
    ball_detector_model_key: str
    ball_detector_model_name: str
    message: str


class VideoBallDetectionJobResponse(BaseModel):
    success: bool
    status: VideoBallDetectionJobStatus
    analysis_id: str
    job_id: str
    progress: int = Field(ge=0, le=100)
    current_frame: int = Field(ge=0)
    total_frames: int = Field(gt=0)
    created_at: datetime
    updated_at: datetime
    model_path_used: str | None = None
    ball_detector_model_key: str
    ball_detector_model_name: str
    error_message: str | None = None
    result: VideoBallDetectionResultLinks | None = None
    message: str


class PixelPoint(StrictGeometryModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)


class BallCandidate(StrictGeometryModel):
    candidate_id: str
    class_id: int
    class_name: str
    confidence: float = Field(ge=0, le=1)
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4)
    bbox_normalized: NormalizedBox
    center: PixelPoint
    center_normalized: NormalizedPoint
    width_pixels: float = Field(gt=0)
    height_pixels: float = Field(gt=0)
    area_pixels: float = Field(gt=0)
    inside_pitch_corridor: bool | None = None


class FrameDetectionRecord(BaseModel):
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    processed: Literal[True]
    detections: list[BallCandidate] = Field(default_factory=list)


# video_ball_detection_service.IMAGE_SIZE moved 960 -> 1280 to match the
# ball_only_* detectors' native training resolution, but these literals were
# left behind, so every detection run raised a ValidationError. Both values are
# allowed: 1280 is what runs now, 960 is what 136 stored detection documents
# already carry, and narrowing to 1280 alone would make those unloadable.
BallDetectionImageSize = Literal[960, 1280]


class VideoBallDetectionSettings(BaseModel):
    frame_stride: Literal[1]
    imgsz: BallDetectionImageSize
    confidence_threshold: Literal[0.15]
    max_det: Literal[20]


class BallDetectorResultMetadata(BaseModel):
    requested_key: str
    selected_key: str
    display_name: str
    model_file: str


class VideoBallDetectionsDocument(BaseModel):
    analysis_id: str
    detector: BallDetectorResultMetadata | None = None
    model_path_used: str
    model_class_names: list[str]
    settings: VideoBallDetectionSettings
    frames: list[FrameDetectionRecord]


class VideoBallDetectionSummary(BaseModel):
    analysis_id: str
    status: Literal["ready"]
    created_at: datetime
    completed_at: datetime
    original_video_url: str
    processed_video_url: str
    detections_json_url: str
    detections_csv_url: str
    detection_summary_url: str
    detector: BallDetectorResultMetadata | None = None
    model_path_used: str
    model_warning: str | None = None
    model_class_names: list[str]
    device_used: str
    imgsz: BallDetectionImageSize
    confidence_threshold: Literal[0.15]
    frame_stride: Literal[1]
    max_det: Literal[20]
    total_frames: int = Field(gt=0)
    frames_processed: int = Field(gt=0)
    frames_with_candidates: int = Field(ge=0)
    frames_without_candidates: int = Field(ge=0)
    total_candidates: int = Field(ge=0)
    frames_with_multiple_candidates: int = Field(ge=0)
    candidates_inside_pitch_corridor: int = Field(ge=0)
    candidates_outside_pitch_corridor: int = Field(ge=0)
    candidates_without_corridor_information: int = Field(ge=0)
    best_confidence: float = Field(ge=0, le=1)
    average_confidence: float = Field(ge=0, le=1)
    average_candidates_per_detected_frame: float = Field(ge=0)
    processing_duration_seconds: float = Field(ge=0)
    output_video_frame_count: int = Field(gt=0)
    input_fps: float = Field(gt=0)
    output_fps: float = Field(gt=0)
    input_duration_seconds: float = Field(gt=0)
    output_duration_seconds: float = Field(gt=0)
    message: str


class VideoBallDetectionResultResponse(BaseModel):
    success: Literal[True]
    status: Literal["ready"]
    analysis_id: str
    summary: VideoBallDetectionSummary
    frame_candidate_counts: list[int]
    frames: list[FrameDetectionRecord] | None = None
    message: str


VideoBallTrackingJobStatus = Literal[
    "queued",
    "loading_detections",
    "analysing_candidates",
    "building_track",
    "recovering_gaps",
    "fitting_physics",
    "rendering_video",
    "saving_results",
    "ready",
    "failed",
    "no_reliable_track",
]


class VideoBallTrackingResultLinks(BaseModel):
    tracking_video_url: str
    tracking_json_url: str
    tracking_csv_url: str
    tracking_summary_url: str
    delivery_replay_url: str | None = None
    physics_result_url: str | None = None


# Provenance for every final track point (never label all as "detected").
TrackingProvenance = Literal[
    "OBSERVED",
    "TRACKER_RECOVERED",
    "PHYSICS_RECONSTRUCTED",
    "PROJECTED",
]


class VideoBallTrackingStartResponse(BaseModel):
    success: Literal[True]
    status: Literal["queued"]
    analysis_id: str
    job_id: str
    progress: int = Field(ge=0, le=100)
    message: str


class VideoBallTrackingJobResponse(BaseModel):
    success: bool
    status: VideoBallTrackingJobStatus
    analysis_id: str
    job_id: str
    progress: int = Field(ge=0, le=100)
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None
    result: VideoBallTrackingResultLinks | None = None
    message: str


class TrackingPoint(StrictGeometryModel):
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    # Legacy debug alias kept for older UI colour maps; provenance is canonical.
    source: Literal["observed", "predicted", "recovered"]
    provenance: TrackingProvenance
    candidate_id: str | None = None
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    normalized_x: float = Field(ge=0, le=1)
    normalized_y: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    uncertainty: float = Field(default=0.0, ge=0, le=1)
    vx: float
    vy: float
    prediction_error: float | None = Field(default=None, ge=0)
    inside_pitch_corridor: bool | None = None


class TrackingCandidateScoreComponents(StrictGeometryModel):
    detector_confidence: float
    motion: float
    prediction_proximity: float
    direction: float
    size_consistency: float
    corridor: float
    static_penalty: float = Field(ge=0)
    jump_penalty: float = Field(ge=0)
    total: float


class TrackingCandidateDiagnostic(StrictGeometryModel):
    frame_index: int = Field(ge=0)
    candidate_id: str
    selected: bool
    selection_reason: str
    static_likelihood: float = Field(ge=0, le=1)
    score_components: TrackingCandidateScoreComponents | None = None


class PrimaryBounceResult(BaseModel):
    bounce_detected: bool | Literal["uncertain"]
    bounce_frame: int | None = Field(default=None, ge=0)
    bounce_timestamp_seconds: float | None = Field(default=None, ge=0)
    bounce_x: float | None = Field(default=None, ge=0)
    bounce_y: float | None = Field(default=None, ge=0)
    bounce_normalized_x: float | None = Field(default=None, ge=0, le=1)
    bounce_normalized_y: float | None = Field(default=None, ge=0, le=1)
    confidence: float = Field(default=0.0, ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class VideoBallTrackingSettings(BaseModel):
    motion_model: Literal["constant_velocity_recent_median"]
    max_recoverable_gap: int = Field(ge=1)
    minimum_observed_points: int = Field(ge=2)
    static_radius_normalized: float = Field(gt=0)
    base_gate_normalized: float = Field(gt=0)
    maximum_gate_normalized: float = Field(gt=0)
    history_points: int = Field(ge=1)
    beam_width: int = Field(default=4, ge=1, le=8)
    tracker_version: Literal["delivery_track_v2"] = "delivery_track_v2"


class VideoBallTrackingDocument(BaseModel):
    analysis_id: str
    status: Literal["ready", "no_reliable_track"]
    created_at: datetime
    completed_at: datetime
    settings: VideoBallTrackingSettings
    primary_track: list[TrackingPoint]
    # Raw association decisions before bidirectional refinement / outlier trim.
    raw_primary_track: list[TrackingPoint] = Field(default_factory=list)
    candidate_diagnostics: list[TrackingCandidateDiagnostic]
    bounce: PrimaryBounceResult | None = None
    physics: DeliveryPhysicsResult | None = None
    message: str


class VideoBallTrackingSummary(BaseModel):
    analysis_id: str
    tracking_job_id: str | None = None
    source_track_id: str | None = None
    track_source_consistent: bool | None = None
    status: Literal["ready", "no_reliable_track"]
    total_video_frames: int = Field(gt=0)
    raw_candidate_count: int = Field(ge=0)
    candidate_frames: int = Field(ge=0)
    track_start_frame: int | None = Field(default=None, ge=0)
    track_end_frame: int | None = Field(default=None, ge=0)
    # Earliest strong observation of the final hypothesis — not true release.
    first_supported_delivery_point: int | None = Field(default=None, ge=0)
    track_start_label: Literal["track_start", "unavailable"] = "unavailable"
    track_duration_frames: int = Field(ge=0)
    track_duration_seconds: float = Field(ge=0)
    observed_track_points: int = Field(ge=0)
    predicted_points: int = Field(ge=0)
    recovered_points: int = Field(ge=0)
    physics_reconstructed_points: int = Field(default=0, ge=0)
    projected_points: int = Field(default=0, ge=0)
    rejected_candidates: int = Field(ge=0)
    longest_gap_frames: int = Field(ge=0)
    observation_ratio: float = Field(default=0.0, ge=0, le=1)
    average_observed_confidence: float = Field(ge=0, le=1)
    consistency_score: float = Field(default=0.0, ge=0, le=1)
    track_confidence: float = Field(ge=0, le=1)
    track_quality: Literal["high", "medium", "low", "failed"]
    approximate_direction: str
    possible_bounce_transition_detected: bool | Literal["uncertain"]
    bounce_detected: bool | Literal["uncertain"] = "uncertain"
    bounce_frame: int | None = Field(default=None, ge=0)
    bounce_confidence: float = Field(default=0.0, ge=0, le=1)
    tracking_video_url: str
    delivery_replay_url: str | None = None
    replay_payload_url: str | None = None
    finalized_track_url: str | None = None
    physics_result_url: str | None = None
    physics_engine_version: Literal["v1"] | None = None
    physics_status: Literal[
        "SUCCESS",
        "PARTIAL",
        "IMAGE_SPACE_ONLY",
        "INSUFFICIENT_EVIDENCE",
        "FAILED",
    ] | None = None
    tracking_json_url: str
    tracking_csv_url: str
    tracking_summary_url: str
    processing_duration_seconds: float = Field(ge=0)
    message: str


class VideoBallTrackingResultResponse(BaseModel):
    success: bool
    status: Literal["ready", "no_reliable_track"]
    analysis_id: str
    summary: VideoBallTrackingSummary
    primary_track: list[TrackingPoint]
    render_track: list[TrackingPoint] = Field(default_factory=list)
    raw_primary_track: list[TrackingPoint] = Field(default_factory=list)
    candidate_diagnostics: list[TrackingCandidateDiagnostic] = Field(
        default_factory=list
    )
    bounce: PrimaryBounceResult | None = None
    physics: DeliveryPhysicsResult | None = None
    track_source_consistency_errors: list[str] = Field(default_factory=list)
    message: str
