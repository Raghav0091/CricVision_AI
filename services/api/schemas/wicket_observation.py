"""Contracts for evidence-backed real-video wicket observations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


QualityLevel = Literal["HIGH", "MEDIUM", "LOW", "UNAVAILABLE"]
StabilityLevel = Literal["STABLE", "PARTIALLY_STABLE", "UNSTABLE", "NOT_FOUND"]
PerspectiveRole = Literal[
    "NEAR_WICKET_CANDIDATE",
    "FAR_WICKET_CANDIDATE",
    "UNRESOLVED_WICKET",
]
RegistrationRole = Literal[
    "PRIMARY_ANCHOR",
    "SECONDARY_ANCHOR",
    "VALIDATION_ONLY",
    "DO_NOT_USE",
]
ObservationStatus = Literal[
    "READY_FOR_REGISTRATION_EXPERIMENT",
    "PARTIAL",
    "INSUFFICIENT_WICKETS",
    "INSUFFICIENT_LANDMARKS",
    "UNSTABLE",
    "FAILED",
]


class WicketObservationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PixelPoint(WicketObservationModel):
    x: float
    y: float


class PixelBox(WicketObservationModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class SetupFrameCandidate(WicketObservationModel):
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    score: float = Field(ge=0, le=1)
    sharpness: float = Field(ge=0)
    brightness: float = Field(ge=0, le=255)
    wicket_detection_count: int = Field(ge=0)
    mean_detector_confidence: float = Field(ge=0, le=1)
    detection_stability: float = Field(ge=0, le=1)
    obstruction_score: float = Field(ge=0, le=1)
    selected: bool
    rejection_reasons: list[str] = Field(default_factory=list)


class RawWicketDetection(WicketObservationModel):
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    bbox: PixelBox
    confidence: float = Field(ge=0, le=1)
    class_name: str
    source: str
    detector_model: str
    perspective_role: PerspectiveRole


class RoiMetadata(WicketObservationModel):
    source_frame_width: int = Field(gt=0)
    source_frame_height: int = Field(gt=0)
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    padding_x: int = Field(ge=0)
    padding_y: int = Field(ge=0)
    native_scale: Literal[1.0] = 1.0
    processing_variants: list[str] = Field(default_factory=list)


class WicketLineObservation(WicketObservationModel):
    start: PixelPoint
    end: PixelPoint


class WicketLandmarkObservation(WicketObservationModel):
    semantic_id: str
    geometry_type: Literal["POINT", "LINE"]
    pixel_x: float | None = None
    pixel_y: float | None = None
    line: WicketLineObservation | None = None
    confidence: float = Field(ge=0, le=1)
    uncertainty_px: float = Field(ge=0)
    extraction_method: str
    supporting_evidence: list[str] = Field(default_factory=list)
    supporting_frames: list[int] = Field(default_factory=list)
    registration_role: RegistrationRole
    quality: QualityLevel
    status: Literal["AVAILABLE", "UNAVAILABLE", "REJECTED"]
    rejection_reason: str | None = None


class WicketRegionObservation(WicketObservationModel):
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    bbox: PixelBox
    centre: PixelPoint
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    detector_confidence: float = Field(ge=0, le=1)
    detector_model: str
    source: str
    temporal_support: int = Field(ge=0)
    supporting_frame_ids: list[int] = Field(default_factory=list)
    centre_variation_px: float = Field(ge=0)
    size_variation_ratio: float = Field(ge=0)
    confidence_variation: float = Field(ge=0)
    perspective_role: PerspectiveRole
    stability: StabilityLevel
    quality: QualityLevel
    uncertainty_px: float = Field(ge=0)
    rejection_reason: str | None = None


class WicketObservation(WicketObservationModel):
    region: WicketRegionObservation
    roi: RoiMetadata
    coarse_landmarks: list[WicketLandmarkObservation]
    detailed_landmarks: list[WicketLandmarkObservation]
    detailed_landmarks_status: Literal[
        "AVAILABLE", "PARTIAL", "INSUFFICIENT_EVIDENCE"
    ]
    quality_score: float = Field(ge=0, le=1)
    quality_factors: dict[str, float]
    warnings: list[str] = Field(default_factory=list)
    roi_debug_urls: dict[str, str] = Field(default_factory=dict)


class AssignmentHypothesis(WicketObservationModel):
    hypothesis_id: Literal["A", "B"]
    near_semantic_end: Literal["bowler", "striker"]
    far_semantic_end: Literal["bowler", "striker"]
    finalised: Literal[False] = False
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)


class WicketObservationDiagnostics(WicketObservationModel):
    detector_model_path: str
    detector_class_labels: list[str]
    clean_source_video: str
    sampled_frame_ids: list[int]
    raw_detections: list[RawWicketDetection]
    rejected_detections: list[dict[str, object]] = Field(default_factory=list)
    setup_frame_image_url: str | None = None
    raw_detection_overlay_url: str | None = None
    landmark_overlay_url: str | None = None
    result_json_url: str | None = None


class WicketObservationResult(WicketObservationModel):
    version: Literal["wicket_observations_v1"] = "wicket_observations_v1"
    analysis_id: str
    status: ObservationStatus
    setup_frame: SetupFrameCandidate | None
    supporting_frames: list[SetupFrameCandidate] = Field(default_factory=list)
    frame_candidates: list[SetupFrameCandidate] = Field(default_factory=list)
    near_wicket: WicketObservation | None = None
    far_wicket: WicketObservation | None = None
    unresolved_regions: list[WicketRegionObservation] = Field(default_factory=list)
    assignment_hypotheses: list[AssignmentHypothesis]
    warnings: list[str] = Field(default_factory=list)
    diagnostics: WicketObservationDiagnostics
    future_registration_readiness: ObservationStatus
    message: str
    developer_only: Literal[True] = True
