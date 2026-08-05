"""Versioned native-pixel wicket landmark evidence contracts."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .wicket_observation import PixelBox


class EvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, use_enum_values=True)


class EvidenceSemanticType(str, Enum):
    EXACT = "EXACT"
    POINTLIKE = "POINTLIKE"
    LINE = "LINE"
    SOFT = "SOFT"
    UNAVAILABLE = "UNAVAILABLE"


class EvidenceStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    REJECTED = "REJECTED"


class WicketEvidencePoint(EvidenceModel):
    semantic_id: str = Field(min_length=1)
    x_px: float | None = None
    y_px: float | None = None
    confidence: float = Field(ge=0, le=1)
    uncertainty_x_px: float | None = Field(default=None, ge=0)
    uncertainty_y_px: float | None = Field(default=None, ge=0)
    supporting_frame_count: int = Field(ge=0)
    supporting_frame_ids: list[int] = Field(default_factory=list)
    extraction_method: str = Field(min_length=1)
    semantic_type: EvidenceSemanticType
    status: EvidenceStatus
    correlation_family: str | None = None
    rejection_reason: str | None = None

    @model_validator(mode="after")
    def validate_availability(self) -> "WicketEvidencePoint":
        available = self.status == "AVAILABLE"
        coordinates = (self.x_px, self.y_px)
        uncertainties = (self.uncertainty_x_px, self.uncertainty_y_px)
        if available and self.semantic_type == "UNAVAILABLE":
            raise ValueError("Available point cannot use UNAVAILABLE semantics.")
        if available and (any(value is None for value in coordinates + uncertainties)):
            raise ValueError("Available point requires coordinates and uncertainty.")
        if not available and any(value is not None for value in coordinates):
            raise ValueError("Unavailable or rejected point cannot contain coordinates.")
        if self.semantic_type == "UNAVAILABLE" and available:
            raise ValueError("UNAVAILABLE semantics cannot be marked available.")
        if self.supporting_frame_count != len(set(self.supporting_frame_ids)):
            raise ValueError("Point supporting_frame_count must match unique frame IDs.")
        return self


class NormalizedLineEquation(EvidenceModel):
    a: float
    b: float
    c: float

    @model_validator(mode="after")
    def validate_normalized(self) -> "NormalizedLineEquation":
        magnitude = (self.a * self.a + self.b * self.b) ** 0.5
        if not 0.999 <= magnitude <= 1.001:
            raise ValueError("Line equation normal must have unit length.")
        return self


class WicketEvidenceLine(EvidenceModel):
    semantic_id: str = Field(min_length=1)
    start_x_px: float | None = None
    start_y_px: float | None = None
    end_x_px: float | None = None
    end_y_px: float | None = None
    normalized_line_equation: NormalizedLineEquation | None = None
    confidence: float = Field(ge=0, le=1)
    angular_uncertainty_deg: float | None = Field(default=None, ge=0)
    perpendicular_uncertainty_px: float | None = Field(default=None, ge=0)
    supporting_frame_count: int = Field(ge=0)
    supporting_frame_ids: list[int] = Field(default_factory=list)
    extraction_method: str = Field(min_length=1)
    semantic_type: EvidenceSemanticType
    status: EvidenceStatus
    correlation_family: str | None = None
    rejection_reason: str | None = None

    @model_validator(mode="after")
    def validate_availability(self) -> "WicketEvidenceLine":
        values = (self.start_x_px, self.start_y_px, self.end_x_px, self.end_y_px)
        available = self.status == "AVAILABLE"
        if available and self.semantic_type == "UNAVAILABLE":
            raise ValueError("Available line cannot use UNAVAILABLE semantics.")
        if available and (
            any(value is None for value in values)
            or self.normalized_line_equation is None
            or self.angular_uncertainty_deg is None
            or self.perpendicular_uncertainty_px is None
        ):
            raise ValueError("Available line requires endpoints, equation, and uncertainty.")
        if not available and any(value is not None for value in values):
            raise ValueError("Unavailable or rejected line cannot contain endpoints.")
        if self.supporting_frame_count != len(set(self.supporting_frame_ids)):
            raise ValueError("Line supporting_frame_count must match unique frame IDs.")
        return self


class NativeRoi(EvidenceModel):
    box: PixelBox
    padding_left_px: int = Field(default=0, ge=0)
    padding_top_px: int = Field(default=0, ge=0)
    padding_right_px: int = Field(default=0, ge=0)
    padding_bottom_px: int = Field(default=0, ge=0)
    clipped: bool = False


class WicketEvidenceQuality(EvidenceModel):
    detailed_axis_count: int = Field(ge=0, le=3)
    top_point_count: int = Field(ge=0, le=3)
    base_point_count: int = Field(ge=0, le=3)
    line_count: int = Field(ge=0)
    independent_constraint_count: int = Field(ge=0)
    temporal_support: int = Field(ge=0)
    mean_confidence: float = Field(ge=0, le=1)
    median_uncertainty_px: float | None = Field(default=None, ge=0)
    severe_clipping: bool
    false_line_risk: float = Field(ge=0, le=1)
    evidence_grade: Literal["DETAILED", "PARTIAL", "COARSE", "INSUFFICIENT"]


class WicketLandmarkDebugMedia(EvidenceModel):
    native_roi_image_url: str | None = None
    temporal_consensus_image_url: str | None = None
    accepted_evidence_overlay_url: str | None = None

    @field_validator(
        "native_roi_image_url",
        "temporal_consensus_image_url",
        "accepted_evidence_overlay_url",
    )
    @classmethod
    def validate_analysis_media_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if (
            not value.startswith("/static/video-analysis/")
            or "/calibration/wicket_landmarks_v1/" not in value
            or ".." in value
            or "\\" in value
        ):
            raise ValueError("Debug media must use a safe analysis-owned URL.")
        return value


class WicketLandmarkSet(EvidenceModel):
    role: Literal["near", "far"]
    source_consensus_box: PixelBox
    native_roi: NativeRoi
    supporting_frame_ids: list[int] = Field(default_factory=list)
    crop_quality: float = Field(ge=0, le=1)
    alignment_quality: float = Field(ge=0, le=1)
    axes: list[WicketEvidenceLine] = Field(default_factory=list)
    points: list[WicketEvidencePoint] = Field(default_factory=list)
    lines: list[WicketEvidenceLine] = Field(default_factory=list)
    outer_envelope: PixelBox | None = None
    evidence_completeness: WicketEvidenceQuality
    confidence: float = Field(ge=0, le=1)
    uncertainty_px: float | None = Field(default=None, ge=0)
    clipping: bool
    warnings: list[str] = Field(default_factory=list)
    debug_media: WicketLandmarkDebugMedia | None = None

    @model_validator(mode="after")
    def validate_independent_count(self) -> "WicketLandmarkSet":
        available = [
            item
            for item in [*self.points, *self.axes, *self.lines]
            if item.status == "AVAILABLE"
            and item.semantic_type in {"EXACT", "POINTLIKE", "LINE"}
        ]
        families = {
            item.correlation_family or item.semantic_id
            for item in available
        }
        if self.evidence_completeness.independent_constraint_count > len(families):
            raise ValueError(
                "Independent constraint count exceeds independent evidence families."
            )
        return self


class SupportingFrameEvidence(EvidenceModel):
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    quality_score: float = Field(ge=0, le=1)
    selected: bool
    rejection_reasons: list[str] = Field(default_factory=list)


class FrameSelectionSummary(EvidenceModel):
    frames_considered: int = Field(ge=0)
    frames_selected: int = Field(ge=0)
    deterministic: Literal[True] = True
    minimum_required: int = Field(ge=1)
    selection_method: str = Field(min_length=1)


class TemporalAlignmentSummary(EvidenceModel):
    method: str = Field(min_length=1)
    frames_attempted: int = Field(ge=0)
    frames_aligned: int = Field(ge=0)
    frames_rejected: int = Field(ge=0)
    median_normalized_residual: float | None = Field(default=None, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)


class WicketLandmarkEvidenceRunRequest(EvidenceModel):
    reuse_existing_observations: bool = True
    force_redetect: bool = False
    include_optional_scene_evidence: bool = False
    rerun_auto_registration: bool = False
    write_debug_media: bool = False
    preset_id: str = "STANDARD_REAR_WICKET_NET_V1"

    @model_validator(mode="after")
    def validate_detection_policy(self) -> "WicketLandmarkEvidenceRunRequest":
        if self.force_redetect and self.reuse_existing_observations:
            raise ValueError("force_redetect and reuse_existing_observations conflict.")
        if not self.force_redetect and not self.reuse_existing_observations:
            raise ValueError("An observation source must be selected.")
        return self


class WicketLandmarkEvidenceResult(EvidenceModel):
    wicket_landmark_evidence_version: Literal["v1"] = "v1"
    analysis_id: str = Field(min_length=1)
    source_observation_version: str
    created_at: str
    status: Literal["READY", "PARTIAL", "INSUFFICIENT_EVIDENCE", "FAILED"]
    native_image_width: int = Field(gt=0)
    native_image_height: int = Field(gt=0)
    rotation_applied: int = 0
    coordinate_space: Literal["NATIVE_ORIENTED_PIXELS"] = "NATIVE_ORIENTED_PIXELS"
    near_wicket: WicketLandmarkSet | None = None
    far_wicket: WicketLandmarkSet | None = None
    optional_scene_evidence: list[WicketEvidenceLine] = Field(default_factory=list)
    supporting_frames: list[SupportingFrameEvidence] = Field(default_factory=list)
    frame_selection: FrameSelectionSummary
    temporal_alignment: TemporalAlignmentSummary
    extraction_diagnostics: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict
    )
    warnings: list[str] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)
    detector_reused: bool
    production_accepted: Literal[False] = False
    metrics_unlocked: list[str] = Field(default_factory=list, max_length=0)
