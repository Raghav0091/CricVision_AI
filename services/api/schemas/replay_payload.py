"""Virtual Pitch Replay payload contracts (schema version 1.0)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


REPLAY_SCHEMA_VERSION = "1.0"
REPLAY_COORDINATE_SYSTEM = "CRICVISION_PITCH_V1"
REPLAY_DISTANCE_UNIT = "metre"
REPLAY_TIME_UNIT = "second"
REPLAY_EXPORT_WIDTH = 1920
REPLAY_EXPORT_HEIGHT = 1080

MeasurementValidity = Literal[
    "CALIBRATED",
    "IMAGE_SPACE_ONLY",
    "VISUALIZATION_ONLY",
    "INSUFFICIENT_EVIDENCE",
]
GeometryValidity = Literal[
    "VALID_METRIC_3D",
    "INVALID_REPROJECTION",
    "OUTSIDE_PITCH_GEOMETRY",
    "IMAGE_SPACE_ONLY",
]
MetricAvailabilityStatus = Literal[
    "AVAILABLE",
    "UNAVAILABLE",
    "NOT_COMPUTED",
    "NOT_IMPLEMENTED",
]
TrajectoryProvenance = Literal["OBSERVED", "RECOVERED", "PHYSICS_FITTED"]
ReplayPayloadStatus = Literal[
    "READY",
    "PENDING",
    "NOT_IMPLEMENTED",
    "INSUFFICIENT_EVIDENCE",
]
CameraSource = Literal[
    "CALIBRATED",
    "PRESET_VISUALIZATION",
    "UNAVAILABLE",
]
ReplayCalibrationSource = Literal[
    "ACCEPTED_SCENE_CALIBRATION",
    "ACCEPTED_WICKET_BOX_CALIBRATION",
]


class ReplayModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class WorldPoint3D(ReplayModel):
    x_m: float
    y_m: float
    z_m: float


class ImagePoint2D(ReplayModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)


class ReplayMetric(ReplayModel):
    """Metric wrapper — never use zero for missing evidence."""

    value: float | None = None
    unit: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    method: str | None = None
    status: MetricAvailabilityStatus = "NOT_COMPUTED"
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def validate_availability(self) -> "ReplayMetric":
        if self.status in {"UNAVAILABLE", "NOT_COMPUTED", "NOT_IMPLEMENTED"}:
            if self.value is not None:
                raise ValueError(
                    "Metric value must remain null when status is unavailable."
                )
        if self.status == "AVAILABLE" and self.value is None:
            raise ValueError("Available metrics must include a value.")
        return self


class ReplayTrajectorySample(ReplayModel):
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    world_position: WorldPoint3D | None = None
    image_position: ImagePoint2D | None = None
    provenance: TrajectoryProvenance
    confidence: float = Field(ge=0, le=1)


class ReplayCamera(ReplayModel):
    source: CameraSource = "UNAVAILABLE"
    calibration_source: ReplayCalibrationSource | None = None
    preset_name: str | None = None
    image_width: int | None = Field(default=None, gt=0)
    image_height: int | None = Field(default=None, gt=0)
    camera_matrix: list[list[float]] | None = None
    rotation_matrix: list[list[float]] | None = None
    translation_vector: list[float] | None = None
    distortion_coefficients: list[float] | None = None
    visualization_only: bool = False


class ReplayPlayback(ReplayModel):
    export_width: int = Field(default=REPLAY_EXPORT_WIDTH, gt=0)
    export_height: int = Field(default=REPLAY_EXPORT_HEIGHT, gt=0)
    landscape: Literal[True] = True
    fps: float | None = Field(default=None, gt=0)
    duration_seconds: float | None = Field(default=None, ge=0)
    start_frame_index: int | None = Field(default=None, ge=0)
    end_frame_index: int | None = Field(default=None, ge=0)


class ReplayBounce(ReplayModel):
    detected: bool | None = None
    frame_index: int | None = Field(default=None, ge=0)
    timestamp_seconds: float | None = Field(default=None, ge=0)
    world_position: WorldPoint3D | None = None
    image_position: ImagePoint2D | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    status: MetricAvailabilityStatus = "NOT_COMPUTED"
    unavailable_reason: str | None = None


class ReplayMetrics(ReplayModel):
    release_speed_kmh: ReplayMetric
    average_pre_bounce_speed_kmh: ReplayMetric
    speed_at_bounce_kmh: ReplayMetric
    overall_stump_to_stump_speed_kmh: ReplayMetric
    delivery_length_m: ReplayMetric
    # Product label: estimated lateral deviation (not swing).
    estimated_lateral_deviation_m: ReplayMetric


class ReplayDiagnostics(ReplayModel):
    status: ReplayPayloadStatus = "NOT_IMPLEMENTED"
    measurement_validity: MeasurementValidity = "INSUFFICIENT_EVIDENCE"
    geometry_validity: GeometryValidity | None = None
    source_track_id: str | None = None
    tracking_job_id: str | None = None
    calibration_id: str | None = None
    generated_at: str | None = None
    calibration_source: ReplayCalibrationSource | None = None
    mean_reprojection_px: float | None = Field(default=None, ge=0)
    median_reprojection_px: float | None = Field(default=None, ge=0)
    p95_reprojection_px: float | None = Field(default=None, ge=0)
    max_reprojection_px: float | None = Field(default=None, ge=0)
    in_pitch_fraction: float | None = Field(default=None, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)
    unavailable_reason: str | None = None


class ReplayPayloadV1(ReplayModel):
    schema_version: Literal["1.0"] = REPLAY_SCHEMA_VERSION
    analysis_id: str = Field(min_length=1)
    coordinate_system: Literal["CRICVISION_PITCH_V1"] = REPLAY_COORDINATE_SYSTEM
    distance_unit: Literal["metre"] = REPLAY_DISTANCE_UNIT
    time_unit: Literal["second"] = REPLAY_TIME_UNIT
    measurement_validity: MeasurementValidity = "INSUFFICIENT_EVIDENCE"
    camera: ReplayCamera
    playback: ReplayPlayback
    trajectory: list[ReplayTrajectorySample] = Field(default_factory=list)
    bounce: ReplayBounce
    metrics: ReplayMetrics
    diagnostics: ReplayDiagnostics

    @model_validator(mode="after")
    def enforce_measurement_validity_rules(self) -> "ReplayPayloadV1":
        if self.measurement_validity == "IMAGE_SPACE_ONLY":
            for sample in self.trajectory:
                if sample.world_position is not None:
                    raise ValueError(
                        "IMAGE_SPACE_ONLY replay must not expose world positions."
                    )
            if self.bounce.world_position is not None:
                raise ValueError(
                    "IMAGE_SPACE_ONLY replay must not expose bounce world position."
                )
            for metric in (
                self.metrics.release_speed_kmh,
                self.metrics.average_pre_bounce_speed_kmh,
                self.metrics.speed_at_bounce_kmh,
                self.metrics.overall_stump_to_stump_speed_kmh,
                self.metrics.delivery_length_m,
                self.metrics.estimated_lateral_deviation_m,
            ):
                if metric.value is not None:
                    raise ValueError(
                        "IMAGE_SPACE_ONLY replay must not expose world metrics."
                    )

        if self.measurement_validity == "VISUALIZATION_ONLY":
            if not self.camera.visualization_only:
                raise ValueError(
                    "VISUALIZATION_ONLY replay requires camera.visualization_only."
                )
            calibrated_metric_available = any(
                metric.status == "AVAILABLE"
                for metric in (
                    self.metrics.release_speed_kmh,
                    self.metrics.average_pre_bounce_speed_kmh,
                    self.metrics.speed_at_bounce_kmh,
                    self.metrics.delivery_length_m,
                    self.metrics.estimated_lateral_deviation_m,
                )
            )
            if calibrated_metric_available:
                raise ValueError(
                    "VISUALIZATION_ONLY replay cannot claim calibrated measurements."
                )

        return self


def unavailable_metric(unit: str, reason: str) -> ReplayMetric:
    return ReplayMetric(
        value=None,
        unit=unit,
        confidence=None,
        method=None,
        status="UNAVAILABLE",
        unavailable_reason=reason,
    )


def build_stage0_replay_payload(analysis_id: str) -> ReplayPayloadV1:
    """Honest Stage 0 placeholder — no fabricated calibration or metrics."""
    reason = "Virtual Pitch Replay payload is not implemented yet."
    return ReplayPayloadV1(
        analysis_id=analysis_id,
        measurement_validity="INSUFFICIENT_EVIDENCE",
        camera=ReplayCamera(source="UNAVAILABLE", visualization_only=False),
        playback=ReplayPlayback(),
        trajectory=[],
        bounce=ReplayBounce(status="NOT_IMPLEMENTED", unavailable_reason=reason),
        metrics=ReplayMetrics(
            release_speed_kmh=unavailable_metric("km/h", reason),
            average_pre_bounce_speed_kmh=unavailable_metric("km/h", reason),
            speed_at_bounce_kmh=unavailable_metric("km/h", reason),
            overall_stump_to_stump_speed_kmh=unavailable_metric("km/h", reason),
            delivery_length_m=unavailable_metric("m", reason),
            estimated_lateral_deviation_m=unavailable_metric("m", reason),
        ),
        diagnostics=ReplayDiagnostics(
            status="NOT_IMPLEMENTED",
            measurement_validity="INSUFFICIENT_EVIDENCE",
            unavailable_reason=reason,
            warnings=["Stage 0 contract only — replay pipeline not wired."],
        ),
    )
