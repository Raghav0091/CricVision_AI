from datetime import datetime
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class WicketCandidate(StrictGeometryModel):
    candidate_id: str
    confidence: float = Field(ge=0, le=1)
    class_name: str
    box: NormalizedBox
    center: NormalizedPoint
    bottom_center: NormalizedPoint


class WicketCalibration(StrictGeometryModel):
    label: Literal["striker", "non_striker"]
    source: Literal["detected", "adjusted", "manual"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    box: NormalizedBox
    center: NormalizedPoint
    bottom_center: NormalizedPoint


class WicketCalibrationInput(StrictGeometryModel):
    label: Literal["striker", "non_striker"]
    source: Literal["detected", "adjusted", "manual"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    box: NormalizedBox


class PitchGeometry(StrictGeometryModel):
    axis_start: NormalizedPoint
    axis_end: NormalizedPoint
    corridor: list[NormalizedPoint] = Field(min_length=4, max_length=4)
    near_end_label: Literal["striker", "non_striker"]
    far_end_label: Literal["striker", "non_striker"]
    geometry_type: Literal["approximate_2d"]
    corridor_width_multiplier: float = Field(ge=0.7, le=1.5)


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
    original_video_url: str
    reference_frame_url: str
    calibration_status: Literal["confirmed"] | None = None
    calibration_url: str | None = None
    calibration_overlay_url: str | None = None
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


class VideoCalibrationDetectionResponse(BaseModel):
    success: bool
    status: Literal[
        "candidates_ready",
        "manual_required",
        "stump_detector_missing",
        "stump_detector_error",
    ]
    analysis_id: str
    reference_frame_url: str
    image_width: int
    image_height: int
    candidates: list[WicketCandidate] = Field(default_factory=list)
    provisional_striker_wicket: WicketCalibration | None = None
    provisional_non_striker_wicket: WicketCalibration | None = None
    pitch_geometry: PitchGeometry | None = None
    model_path_used: str
    warning: str | None = None
    message: str


class VideoCalibrationConfirmationRequest(BaseModel):
    analysis_id: str
    striker_wicket: WicketCalibrationInput
    non_striker_wicket: WicketCalibrationInput
    corridor_width_multiplier: float = Field(default=1.0, ge=0.7, le=1.5)
    user_note: str | None = Field(default=None, max_length=1000)


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
    image_width: int
    image_height: int
    model_path_used: str | None = None
    striker_wicket: WicketCalibration
    non_striker_wicket: WicketCalibration
    pitch_geometry: PitchGeometry
    user_note: str | None = None
    message: str


CalibrationLandmarkSource = Literal[
    "detected",
    "inferred",
    "manually_adjusted",
    "manual",
]


class CricketPitchGeometry(StrictGeometryModel):
    pitch_length_m: float = Field(default=20.12, gt=0, le=40)
    wicket_width_m: float = Field(default=0.2286, gt=0, le=1)
    wicket_height_m: float = Field(default=0.7112, gt=0, le=2)
    pitch_width_m: float = Field(default=3.05, gt=0, le=10)
    popping_crease_distance_m: float = Field(default=1.22, gt=0, le=5)

    @model_validator(mode="after")
    def validate_pitch_dimensions(self) -> "CricketPitchGeometry":
        if self.pitch_width_m < self.wicket_width_m:
            raise ValueError("Pitch width must exceed wicket width.")
        if self.popping_crease_distance_m >= self.pitch_length_m / 2:
            raise ValueError("Popping crease distance is invalid.")
        return self


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
    created_at: datetime
    updated_at: datetime
    user_note: str | None = None
    future_camera_pose: CalibrationV2FutureCameraPoseFields
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


class VideoBallDetectionStartResponse(BaseModel):
    success: bool
    status: Literal["queued"]
    analysis_id: str
    job_id: str
    progress: int = Field(ge=0, le=100)
    current_frame: int = Field(ge=0)
    total_frames: int = Field(gt=0)
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


class VideoBallDetectionSettings(BaseModel):
    frame_stride: Literal[1]
    imgsz: Literal[960]
    confidence_threshold: Literal[0.15]
    max_det: Literal[20]


class VideoBallDetectionsDocument(BaseModel):
    analysis_id: str
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
    model_path_used: str
    model_warning: str | None = None
    model_class_names: list[str]
    device_used: str
    imgsz: Literal[960]
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
    message: str


VideoBallTrackingJobStatus = Literal[
    "queued",
    "loading_detections",
    "analysing_candidates",
    "building_track",
    "recovering_gaps",
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
    source: Literal["observed", "predicted", "recovered"]
    candidate_id: str | None = None
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    normalized_x: float = Field(ge=0, le=1)
    normalized_y: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
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


class VideoBallTrackingSettings(BaseModel):
    motion_model: Literal["constant_velocity_recent_median"]
    max_recoverable_gap: int = Field(ge=1)
    minimum_observed_points: int = Field(ge=2)
    static_radius_normalized: float = Field(gt=0)
    base_gate_normalized: float = Field(gt=0)
    maximum_gate_normalized: float = Field(gt=0)
    history_points: int = Field(ge=1)


class VideoBallTrackingDocument(BaseModel):
    analysis_id: str
    status: Literal["ready", "no_reliable_track"]
    created_at: datetime
    completed_at: datetime
    settings: VideoBallTrackingSettings
    primary_track: list[TrackingPoint]
    candidate_diagnostics: list[TrackingCandidateDiagnostic]
    message: str


class VideoBallTrackingSummary(BaseModel):
    analysis_id: str
    status: Literal["ready", "no_reliable_track"]
    total_video_frames: int = Field(gt=0)
    raw_candidate_count: int = Field(ge=0)
    candidate_frames: int = Field(ge=0)
    track_start_frame: int | None = Field(default=None, ge=0)
    track_end_frame: int | None = Field(default=None, ge=0)
    track_duration_frames: int = Field(ge=0)
    track_duration_seconds: float = Field(ge=0)
    observed_track_points: int = Field(ge=0)
    predicted_points: int = Field(ge=0)
    recovered_points: int = Field(ge=0)
    rejected_candidates: int = Field(ge=0)
    longest_gap_frames: int = Field(ge=0)
    average_observed_confidence: float = Field(ge=0, le=1)
    track_confidence: float = Field(ge=0, le=1)
    track_quality: Literal["low", "medium", "good", "strong"]
    approximate_direction: str
    possible_bounce_transition_detected: bool | Literal["uncertain"]
    tracking_video_url: str
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
    message: str
