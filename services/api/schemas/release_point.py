from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .video_analysis import PixelPoint, StrictGeometryModel


ReleaseJobStatus = Literal[
    "queued",
    "loading_inputs",
    "generating_candidates",
    "scoring_candidates",
    "saving_results",
    "ready",
    "unresolved",
    "failed",
]
ReleaseStatus = Literal["ready", "unresolved"]
ReleaseEvidenceMode = Literal[
    "observed_pose_ball_separation",
    "trajectory_pose_inferred",
    "fallback_trajectory_only",
    "unresolved",
]
ReleaseType = Literal["OBSERVED_RELEASE", "INFERRED_RELEASE", "UNRESOLVED"]


class ReleaseAnalysisInput(BaseModel):
    analysis_id: str
    raw_video_path: str
    fps: float = Field(gt=0)
    frame_count: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    detections_path: str
    tracking_path: str
    calibration_path: str
    calibration_v2_path: str | None = None
    camera_pose_path: str | None = None


class ReleaseFrameUncertainty(StrictGeometryModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class ReleaseResult(StrictGeometryModel):
    schema_version: Literal["1.0", "1.1", "1.2", "1.3"]
    analysis_id: str
    status: ReleaseStatus
    release_frame: int | None = Field(default=None, ge=0)
    release_time_seconds: float | None = Field(default=None, ge=0)
    release_point_px: PixelPoint | None = None
    confidence: float = Field(ge=0, le=1)
    frame_uncertainty: ReleaseFrameUncertainty | None = None
    method: str
    evidence_mode: ReleaseEvidenceMode
    release_type: ReleaseType
    evidence: dict[str, Any] = Field(default_factory=dict)
    quality_flags: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


class ReleaseCandidateScore(BaseModel):
    frame_index: int = Field(ge=0)
    score: float = Field(ge=0, le=1)
    method: ReleaseEvidenceMode
    release_type: ReleaseType
    observed: bool
    source: str
    features: dict[str, Any] = Field(default_factory=dict)
    score_components: dict[str, float] = Field(default_factory=dict)
    quality_flags: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class ReleaseResultDocument(BaseModel):
    schema_version: Literal["1.0", "1.1", "1.2", "1.3"]
    analysis_id: str
    created_at: datetime
    completed_at: datetime
    result: ReleaseResult
    candidate_scores: list[ReleaseCandidateScore]
    quality_summary: dict[str, Any] = Field(default_factory=dict)
    message: str


class VideoReleasePointResultLinks(BaseModel):
    release_json_url: str


class VideoReleasePointStartResponse(BaseModel):
    success: Literal[True]
    status: Literal["queued"]
    analysis_id: str
    job_id: str
    progress: int = Field(ge=0, le=100)
    message: str


class VideoReleasePointJobResponse(BaseModel):
    success: bool
    status: ReleaseJobStatus
    analysis_id: str
    job_id: str
    progress: int = Field(ge=0, le=100)
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None
    result: VideoReleasePointResultLinks | None = None
    message: str


class VideoReleasePointResultResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    status: ReleaseStatus
    analysis_id: str
    release_json_url: str
    result: ReleaseResult
    candidate_scores: list[ReleaseCandidateScore] = Field(default_factory=list)
    quality_summary: dict[str, Any] = Field(default_factory=dict)
    message: str
