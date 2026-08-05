"""Canonical finalized delivery track shared by every replay renderer."""

from __future__ import annotations

from datetime import datetime
import math
import statistics
from typing import Literal

from pydantic import BaseModel, Field

from ..schemas.delivery_physics import DeliveryPhysicsResult, TrajectorySample
from ..schemas.video_analysis import (
    TrackingPoint,
    TrackingProvenance,
)

FINALIZED_TRACK_FILENAME = "finalized_track.json"

# ponytail: pixel gate scales with local median step; caps wild physics reprojection.
MIN_JUMP_GATE_PX = 80.0
JUMP_GATE_MEDIAN_MULTIPLIER = 4.0
MAX_PROJECTED_EXTENSION_FRAMES = 12

_PROVENANCE_TO_SOURCE: dict[
    TrackingProvenance, Literal["observed", "predicted", "recovered"]
] = {
    "OBSERVED": "observed",
    "TRACKER_RECOVERED": "recovered",
    "PHYSICS_RECONSTRUCTED": "recovered",
    "PROJECTED": "predicted",
}

_MEASURED_PROVENANCE = frozenset({"OBSERVED", "TRACKER_RECOVERED"})
_EXTENSION_PROVENANCE = frozenset({"PHYSICS_RECONSTRUCTED", "PROJECTED"})

class FinalizedTrackPoint(BaseModel):
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    normalized_x: float = Field(ge=0, le=1)
    normalized_y: float = Field(ge=0, le=1)
    provenance: TrackingProvenance
    confidence: float = Field(ge=0, le=1)
    world_x_m: float | None = None
    world_y_m: float | None = None
    world_z_m: float | None = None
    candidate_id: str | None = None
    source: Literal["observed", "predicted", "recovered"]


class TrackSourceConsistency(BaseModel):
    speed_source_track_id: str
    main_video_source_track_id: str
    delivery_replay_source_track_id: str
    replay_payload_source_track_id: str
    consistent: bool
    errors: list[str] = Field(default_factory=list)


class RejectedExtensionPoint(BaseModel):
    frame_index: int
    reason: str
    provenance: TrackingProvenance
    jump_px: float | None = None


class TrackTermination(BaseModel):
    valid_end_frame: int | None = None
    termination_reason: str | None = None
    first_invalid_frame: int | None = None
    first_invalid_reason: str | None = None
    rejected_extension_points: list[RejectedExtensionPoint] = Field(
        default_factory=list
    )


class FinalizedDeliveryTrack(BaseModel):
    """Single canonical track object for debug, replay, overlay, and payload."""

    track_id: str
    tracking_job_id: str
    analysis_id: str
    generated_at: datetime
    observed: list[FinalizedTrackPoint] = Field(default_factory=list)
    recovered: list[FinalizedTrackPoint] = Field(default_factory=list)
    physics_reconstructed: list[FinalizedTrackPoint] = Field(default_factory=list)
    projected: list[FinalizedTrackPoint] = Field(default_factory=list)
    render_track: list[TrackingPoint] = Field(default_factory=list)
    source_consistency: TrackSourceConsistency
    termination: TrackTermination = Field(default_factory=TrackTermination)
    release_frame: int | None = None
    bounce_frame: int | None = None
    valid_interval_start_frame: int | None = None
    valid_interval_end_frame: int | None = None

def build_finalized_delivery_track(
    *,
    analysis_id: str,
    tracking_job_id: str,
    generated_at: datetime,
    primary_track: list[TrackingPoint],
    physics: DeliveryPhysicsResult,
    fps: float,
    width: int,
    height: int,
) -> FinalizedDeliveryTrack:
    """Merge tracker and physics outputs into one renderer-safe track."""
    track_id = tracking_job_id
    primary_by_frame = {point.frame_index: point for point in primary_track}
    physics_by_frame = {
        sample.frame_index: sample
        for sample in physics.trajectory_samples
        if _sample_pixels_valid(sample, width, height)
    }

    observed: list[FinalizedTrackPoint] = []
    recovered: list[FinalizedTrackPoint] = []
    for point in sorted(primary_track, key=lambda item: item.frame_index):
        finalized = _tracking_to_finalized(point, physics_by_frame.get(point.frame_index))
        if point.provenance == "OBSERVED":
            observed.append(finalized)
        else:
            recovered.append(finalized)

    physics_reconstructed: list[FinalizedTrackPoint] = []
    projected: list[FinalizedTrackPoint] = []
    start_frame = primary_track[0].frame_index if primary_track else None
    end_frame = primary_track[-1].frame_index if primary_track else None
    physics_end = (
        physics.delivery_interval.end_frame
        if physics.delivery_interval.end_frame is not None
        else end_frame
    )
    if start_frame is not None and physics_end is not None:
        for frame_index in range(start_frame, physics_end + 1):
            if frame_index in primary_by_frame:
                continue
            sample = physics_by_frame.get(frame_index)
            if sample is None:
                continue
            finalized = _sample_to_finalized(sample, width, height, fps)
            if sample.provenance == "PROJECTED":
                projected.append(finalized)
            else:
                physics_reconstructed.append(finalized)

    (
        physics_reconstructed,
        projected,
        termination,
    ) = gate_physics_extensions(
        observed=observed,
        recovered=recovered,
        physics_reconstructed=physics_reconstructed,
        projected=projected,
        width=width,
        height=height,
        fps=fps,
        physics_terminal_reason=physics.delivery_interval.terminal_reason,
    )

    render_track = finalized_render_track(
        observed=observed,
        recovered=recovered,
        physics_reconstructed=physics_reconstructed,
        projected=projected,
    )
    valid_start = render_track[0].frame_index if render_track else None
    valid_end = render_track[-1].frame_index if render_track else None
    if termination.valid_end_frame is None and valid_end is not None:
        termination.valid_end_frame = valid_end
    bounce_frame = physics.bounce.frame_index if physics.bounce.status != "INSUFFICIENT_EVIDENCE" else None
    consistency = validate_source_consistency(        track_id,
        speed_source_track_id=track_id,
        main_video_source_track_id=track_id,
        delivery_replay_source_track_id=track_id,
        replay_payload_source_track_id=track_id,
    )
    return FinalizedDeliveryTrack(
        track_id=track_id,
        tracking_job_id=tracking_job_id,
        analysis_id=analysis_id,
        generated_at=generated_at,
        observed=observed,
        recovered=recovered,
        physics_reconstructed=physics_reconstructed,
        projected=projected,
        render_track=render_track,
        source_consistency=consistency,
        termination=termination,
        release_frame=start_frame,
        bounce_frame=bounce_frame,
        valid_interval_start_frame=valid_start,
        valid_interval_end_frame=valid_end,
    )

def finalized_render_track(
    *,
    observed: list[FinalizedTrackPoint],
    recovered: list[FinalizedTrackPoint],
    physics_reconstructed: list[FinalizedTrackPoint],
    projected: list[FinalizedTrackPoint],
) -> list[TrackingPoint]:
    """Ordered one-point-per-frame render list with tracker precedence."""
    by_frame: dict[int, FinalizedTrackPoint] = {}
    precedence: dict[TrackingProvenance, int] = {
        "OBSERVED": 0,
        "TRACKER_RECOVERED": 1,
        "PHYSICS_RECONSTRUCTED": 2,
        "PROJECTED": 3,
    }
    for collection in (projected, physics_reconstructed, recovered, observed):
        for point in collection:
            existing = by_frame.get(point.frame_index)
            if existing is None or precedence[point.provenance] < precedence[existing.provenance]:
                by_frame[point.frame_index] = point
    ordered = sorted(by_frame.values(), key=lambda item: item.frame_index)
    return [_finalized_to_tracking(point, previous) for point, previous in _with_previous(ordered)]


def validate_source_consistency(
    track_id: str,
    *,
    speed_source_track_id: str,
    main_video_source_track_id: str,
    delivery_replay_source_track_id: str,
    replay_payload_source_track_id: str,
) -> TrackSourceConsistency:
    ids = {
        "speed": speed_source_track_id,
        "main_video": main_video_source_track_id,
        "delivery_replay": delivery_replay_source_track_id,
        "replay_payload": replay_payload_source_track_id,
    }
    errors: list[str] = []
    if len(set(ids.values())) != 1:
        errors.append(
            "Internal consistency error: renderers reference different source tracks "
            f"({ids})."
        )
    if track_id not in ids.values():
        errors.append(
            f"Internal consistency error: canonical track_id {track_id!r} "
            "is not referenced by all renderers."
        )
    return TrackSourceConsistency(
        speed_source_track_id=speed_source_track_id,
        main_video_source_track_id=main_video_source_track_id,
        delivery_replay_source_track_id=delivery_replay_source_track_id,
        replay_payload_source_track_id=replay_payload_source_track_id,
        consistent=not errors,
        errors=errors,
    )


def gate_physics_extensions(
    *,
    observed: list[FinalizedTrackPoint],
    recovered: list[FinalizedTrackPoint],
    physics_reconstructed: list[FinalizedTrackPoint],
    projected: list[FinalizedTrackPoint],
    width: int,
    height: int,
    fps: float,
    physics_terminal_reason: str | None = None,
) -> tuple[
    list[FinalizedTrackPoint],
    list[FinalizedTrackPoint],
    TrackTermination,
]:
    """Reject physics extensions that break pixel continuity with measured track."""
    measured = sorted(
        [*observed, *recovered],
        key=lambda item: item.frame_index,
    )
    extensions = sorted(
        [*physics_reconstructed, *projected],
        key=lambda item: item.frame_index,
    )
    termination = TrackTermination()
    if not measured or not extensions:
        if measured:
            termination.valid_end_frame = measured[-1].frame_index
            termination.termination_reason = "last_measured_point"
        return physics_reconstructed, projected, termination

    last_measured_frame = max(point.frame_index for point in measured)
    median_step = _median_pixel_step(measured, fps)
    jump_limit = max(MIN_JUMP_GATE_PX, median_step * JUMP_GATE_MEDIAN_MULTIPLIER)
    anchor = _last_point_before(measured, last_measured_frame)
    heading = _recent_heading(measured, fps)

    accepted_reconstructed: list[FinalizedTrackPoint] = []
    accepted_projected: list[FinalizedTrackPoint] = []
    rejected: list[RejectedExtensionPoint] = []
    previous: FinalizedTrackPoint | None = anchor
    projected_count = 0

    for point in extensions:
        if point.frame_index <= last_measured_frame:
            if point.provenance == "PROJECTED":
                accepted_projected.append(point)
            else:
                accepted_reconstructed.append(point)
            previous = point
            continue

        if projected_count >= MAX_PROJECTED_EXTENSION_FRAMES:
            rejected.append(
                RejectedExtensionPoint(
                    frame_index=point.frame_index,
                    reason="projection_horizon_reached",
                    provenance=point.provenance,
                )
            )
            if termination.first_invalid_frame is None:
                termination.first_invalid_frame = point.frame_index
                termination.first_invalid_reason = "projection_horizon_reached"
            continue

        if previous is None:
            rejected.append(
                RejectedExtensionPoint(
                    frame_index=point.frame_index,
                    reason="missing_anchor_point",
                    provenance=point.provenance,
                )
            )
            continue

        dt = point.timestamp_seconds - previous.timestamp_seconds
        if dt <= 0:
            rejected.append(
                RejectedExtensionPoint(
                    frame_index=point.frame_index,
                    reason="non_monotonic_timestamp",
                    provenance=point.provenance,
                )
            )
            continue

        jump = math.hypot(point.x - previous.x, point.y - previous.y)
        if jump > jump_limit:
            rejected.append(
                RejectedExtensionPoint(
                    frame_index=point.frame_index,
                    reason="pixel_jump_exceeds_gate",
                    provenance=point.provenance,
                    jump_px=round(jump, 2),
                )
            )
            if termination.first_invalid_frame is None:
                termination.first_invalid_frame = point.frame_index
                termination.first_invalid_reason = "pixel_jump_exceeds_gate"
                termination.termination_reason = "physics_reprojection_discontinuity"
                termination.valid_end_frame = previous.frame_index
            continue

        if heading is not None:
            step_x = (point.x - previous.x) / dt
            step_y = (point.y - previous.y) / dt
            heading_dot = _normalize_dot(heading, (step_x, step_y))
            if heading_dot < -0.35 and jump > jump_limit * 0.45:
                rejected.append(
                    RejectedExtensionPoint(
                        frame_index=point.frame_index,
                        reason="heading_reversal",
                        provenance=point.provenance,
                        jump_px=round(jump, 2),
                    )
                )
                if termination.first_invalid_frame is None:
                    termination.first_invalid_frame = point.frame_index
                    termination.first_invalid_reason = "heading_reversal"
                    termination.termination_reason = "physics_projection_heading_reversal"
                    termination.valid_end_frame = previous.frame_index
                continue

        if not (0 <= point.x < width and 0 <= point.y < height):
            rejected.append(
                RejectedExtensionPoint(
                    frame_index=point.frame_index,
                    reason="offscreen_projection",
                    provenance=point.provenance,
                    jump_px=round(jump, 2),
                )
            )
            if termination.first_invalid_frame is None:
                termination.first_invalid_frame = point.frame_index
                termination.first_invalid_reason = "offscreen_projection"
                termination.termination_reason = "projection_offscreen"
                termination.valid_end_frame = previous.frame_index
            continue

        if point.provenance == "PROJECTED":
            accepted_projected.append(point)
            projected_count += 1
        else:
            accepted_reconstructed.append(point)
        previous = point

    if termination.valid_end_frame is None:
        termination.valid_end_frame = (
            previous.frame_index if previous is not None else last_measured_frame
        )
        termination.termination_reason = (
            physics_terminal_reason or "last_valid_render_point"
        )
    termination.rejected_extension_points = rejected
    return accepted_reconstructed, accepted_projected, termination


def _median_pixel_step(points: list[FinalizedTrackPoint], fps: float) -> float:
    if len(points) < 2:
        return MIN_JUMP_GATE_PX * 0.25
    steps: list[float] = []
    for index in range(1, len(points)):
        previous = points[index - 1]
        current = points[index]
        dt = current.timestamp_seconds - previous.timestamp_seconds
        if dt <= 0:
            continue
        steps.append(math.hypot(current.x - previous.x, current.y - previous.y))
    if not steps:
        return MIN_JUMP_GATE_PX * 0.25
    return float(statistics.median(steps))


def _last_point_before(
    points: list[FinalizedTrackPoint],
    frame_index: int,
) -> FinalizedTrackPoint | None:
    eligible = [point for point in points if point.frame_index <= frame_index]
    if not eligible:
        return None
    return max(eligible, key=lambda item: item.frame_index)


def _recent_heading(
    points: list[FinalizedTrackPoint],
    fps: float,
) -> tuple[float, float] | None:
    if len(points) < 2:
        return None
    tail = points[-min(4, len(points)) :]
    start = tail[0]
    end = tail[-1]
    dt = end.timestamp_seconds - start.timestamp_seconds
    if dt <= 0:
        return None
    return ((end.x - start.x) / dt, (end.y - start.y) / dt)


def _normalize_dot(
    vector_a: tuple[float, float],
    vector_b: tuple[float, float],
) -> float:
    norm_a = math.hypot(*vector_a)
    norm_b = math.hypot(*vector_b)
    if norm_a <= 1e-8 or norm_b <= 1e-8:
        return 1.0
    return (vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]) / (norm_a * norm_b)


def cache_bust_url(url: str, *, tracking_job_id: str) -> str:
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}v={tracking_job_id}"


def _with_previous(
    points: list[FinalizedTrackPoint],
) -> list[tuple[FinalizedTrackPoint, FinalizedTrackPoint | None]]:
    previous = None
    pairs: list[tuple[FinalizedTrackPoint, FinalizedTrackPoint | None]] = []
    for point in points:
        pairs.append((point, previous))
        previous = point
    return pairs


def _sample_pixels_valid(sample: TrajectorySample, width: int, height: int) -> bool:
    return (
        math.isfinite(sample.pixel_x)
        and math.isfinite(sample.pixel_y)
        and 0 <= sample.pixel_x < width
        and 0 <= sample.pixel_y < height
    )


def _tracking_to_finalized(
    point: TrackingPoint,
    sample: TrajectorySample | None,
) -> FinalizedTrackPoint:
    return FinalizedTrackPoint(
        frame_index=point.frame_index,
        timestamp_seconds=point.timestamp_seconds,
        x=point.x,
        y=point.y,
        normalized_x=point.normalized_x,
        normalized_y=point.normalized_y,
        provenance=point.provenance,
        confidence=point.confidence,
        world_x_m=sample.world_x_m if sample is not None else None,
        world_y_m=sample.world_y_m if sample is not None else None,
        world_z_m=sample.world_z_m if sample is not None else None,
        candidate_id=point.candidate_id,
        source=_PROVENANCE_TO_SOURCE[point.provenance],
    )


def _sample_to_finalized(
    sample: TrajectorySample,
    width: int,
    height: int,
    fps: float,
) -> FinalizedTrackPoint:
    provenance: TrackingProvenance = (
        "PROJECTED"
        if sample.provenance == "PROJECTED"
        else "PHYSICS_RECONSTRUCTED"
    )
    timestamp = (
        sample.timestamp_seconds
        if math.isfinite(sample.timestamp_seconds)
        else sample.frame_index / fps
    )
    return FinalizedTrackPoint(
        frame_index=sample.frame_index,
        timestamp_seconds=timestamp,
        x=sample.pixel_x,
        y=sample.pixel_y,
        normalized_x=sample.pixel_x / width,
        normalized_y=sample.pixel_y / height,
        provenance=provenance,
        confidence=sample.confidence,
        world_x_m=sample.world_x_m,
        world_y_m=sample.world_y_m,
        world_z_m=sample.world_z_m,
        source=_PROVENANCE_TO_SOURCE[provenance],
    )


def _finalized_to_tracking(
    point: FinalizedTrackPoint,
    previous: FinalizedTrackPoint | None,
) -> TrackingPoint:
    vx = 0.0
    vy = 0.0
    if previous is not None:
        elapsed = point.timestamp_seconds - previous.timestamp_seconds
        if elapsed > 0:
            vx = (point.x - previous.x) / elapsed
            vy = (point.y - previous.y) / elapsed
    return TrackingPoint(
        frame_index=point.frame_index,
        timestamp_seconds=point.timestamp_seconds,
        source=_PROVENANCE_TO_SOURCE[point.provenance],
        provenance=point.provenance,
        candidate_id=point.candidate_id,
        x=point.x,
        y=point.y,
        normalized_x=point.normalized_x,
        normalized_y=point.normalized_y,
        confidence=point.confidence,
        uncertainty=max(0.0, 1.0 - point.confidence),
        vx=vx,
        vy=vy,
    )
