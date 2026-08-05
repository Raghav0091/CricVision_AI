"""Assemble Virtual Pitch Replay payloads from tracking and physics outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..schemas.camera_bridge import CameraBridgeResponse
from ..schemas.delivery_physics import (
    DeliveryPhysicsResult,
    GeometryValidationResult,
    TrajectorySample,
)
from ..schemas.replay_payload import (
    GeometryValidity,
    ImagePoint2D,
    MeasurementValidity,
    MetricAvailabilityStatus,
    ReplayBounce,
    ReplayCamera,
    ReplayDiagnostics,
    ReplayMetric,
    ReplayMetrics,
    ReplayPayloadV1,
    ReplayPlayback,
    ReplayTrajectorySample,
    TrajectoryProvenance,
    WorldPoint3D,
    build_stage0_replay_payload,
    unavailable_metric,
)
from ..schemas.video_analysis import (
    TrackingPoint,
    VideoBallTrackingDocument,
)
from .camera_bridge_service import load_analysis_camera_bridge
from .wicket_box_calibration_service import (
    load_active_accepted_wicket_box_calibration,
)
from .finalized_delivery_track import FinalizedDeliveryTrack
from .video_analysis_service import (
    ANALYSIS_ID_PATTERN,
    VIDEO_ANALYSIS_ROOT,
    VideoAnalysisServiceError,
    load_video_analysis,
)

REPLAY_PAYLOAD_FILENAME = "replay_payload.json"
TRACKING_RESULT_FILENAME = "tracking_result.json"
PHYSICS_RESULT_FILENAME = "physics_result.json"

_PHYSICS_PROVENANCE: dict[str, TrajectoryProvenance] = {
    "OBSERVED": "OBSERVED",
    "RECONSTRUCTED": "RECOVERED",
    "PROJECTED": "PHYSICS_FITTED",
}

_TRACKING_PROVENANCE: dict[str, TrajectoryProvenance] = {
    "OBSERVED": "OBSERVED",
    "TRACKER_RECOVERED": "RECOVERED",
    "PHYSICS_RECONSTRUCTED": "PHYSICS_FITTED",
    "PROJECTED": "PHYSICS_FITTED",
}


@dataclass(frozen=True)
class _ReplayArtifactFingerprint:
    analysis_id: str
    tracking_job_id: str | None = None
    source_track_id: str | None = None
    generated_at: str | None = None
    calibration_id: str | None = None


def _replay_payload_path(analysis_id: str) -> Path:
    return (
        VIDEO_ANALYSIS_ROOT / analysis_id / "tracking" / REPLAY_PAYLOAD_FILENAME
    )


def _tracking_result_path(analysis_id: str) -> Path:
    return (
        VIDEO_ANALYSIS_ROOT
        / analysis_id
        / "tracking"
        / TRACKING_RESULT_FILENAME
    )


def _physics_result_path(analysis_id: str) -> Path:
    return (
        VIDEO_ANALYSIS_ROOT
        / analysis_id
        / "tracking"
        / PHYSICS_RESULT_FILENAME
    )


def load_replay_payload(analysis_id: str) -> ReplayPayloadV1:
    if not analysis_id or not analysis_id.strip():
        raise VideoAnalysisServiceError(
            status_code=400,
            message="analysis_id is required.",
        )
    normalized = analysis_id.strip()
    if not ANALYSIS_ID_PATTERN.fullmatch(normalized):
        raise VideoAnalysisServiceError(
            "Invalid analysis ID.",
            status_code=404,
        )
    fingerprint = _load_replay_artifact_fingerprint(normalized)
    saved = _replay_payload_path(normalized)
    if saved.is_file():
        cached = _try_load_saved_replay_payload(saved)
        if cached is not None and _saved_replay_payload_is_current(
            cached,
            fingerprint,
        ):
            return cached
    try:
        load_video_analysis(normalized)
    except VideoAnalysisServiceError:
        return build_stage0_replay_payload(normalized)
    if not _tracking_result_path(normalized).is_file():
        return _insufficient_replay_payload(
            normalized,
            reason="Moving-ball tracking has not completed.",
        )
    payload = assemble_replay_payload(normalized)
    save_replay_payload(normalized, payload)
    return payload


def save_replay_payload(analysis_id: str, payload: ReplayPayloadV1) -> Path:
    output_path = _replay_payload_path(analysis_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return output_path


def assemble_replay_payload(analysis_id: str) -> ReplayPayloadV1:
    from .finalized_delivery_track import FINALIZED_TRACK_FILENAME
    from .video_ball_tracking_service import load_video_ball_tracking_result

    analysis = load_video_analysis(analysis_id)
    tracking = load_video_ball_tracking_result(analysis_id)
    physics = tracking.physics
    physics_path = _physics_result_path(analysis_id)
    if physics_path.is_file():
        try:
            physics = DeliveryPhysicsResult.model_validate(
                json.loads(physics_path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, ValueError):
            if physics is None:
                raise VideoAnalysisServiceError(
                    "Physics result is unavailable.",
                    status_code=404,
                ) from None
    elif physics is None:
        raise VideoAnalysisServiceError(
            "Physics result is unavailable.",
            status_code=404,
        )
    try:
        document = VideoBallTrackingDocument.model_validate(
            json.loads(
                _tracking_result_path(analysis_id).read_text(encoding="utf-8")
            )
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise VideoAnalysisServiceError(
            "Tracking document is unavailable.",
            status_code=404,
        ) from exc

    finalized_path = (
        VIDEO_ANALYSIS_ROOT
        / analysis_id
        / "tracking"
        / FINALIZED_TRACK_FILENAME
    )
    finalized_track: FinalizedDeliveryTrack | None = None
    if finalized_path.is_file():
        try:
            finalized_track = FinalizedDeliveryTrack.model_validate(
                json.loads(finalized_path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, ValueError):
            finalized_track = None
    render_track = tracking.render_track or document.primary_track

    tracking_job_id, source_track_id, generated_at = _resolve_replay_track_ids(
        analysis_id,
        finalized_track=finalized_track,
        summary=tracking.summary,
    )
    bridge_response = _safe_camera_bridge(analysis_id)
    camera, has_accepted_calibration = _resolve_replay_camera(
        analysis_id,
        physics,
        analysis.width,
        analysis.height,
    )
    measurement_validity = _resolve_measurement_validity(
        has_accepted_calibration=has_accepted_calibration,
        camera=camera,
        physics=physics,
        tracking_status=tracking.status,
        primary_track=render_track,
    )
    trajectory = _build_trajectory(
        physics=physics,
        primary_track=render_track,
        finalized_track=finalized_track,
        measurement_validity=measurement_validity,
    )
    playback = _build_playback(
        physics=physics,
        primary_track=render_track,
        fps=analysis.fps,
    )
    bounce = _build_bounce(physics, measurement_validity)
    metrics = _build_metrics(physics, measurement_validity, bounce)
    diagnostics = _build_diagnostics(
        measurement_validity=measurement_validity,
        physics=physics,
        tracking_status=tracking.status,
        trajectory=trajectory,
        bridge_response=bridge_response,
        camera=camera,
        source_track_id=source_track_id,
        tracking_job_id=tracking_job_id,
        generated_at=generated_at,
        calibration_id=_resolve_calibration_id(analysis_id),
        consistency_errors=tracking.track_source_consistency_errors,
    )
    return ReplayPayloadV1(
        analysis_id=analysis_id,
        measurement_validity=measurement_validity,
        camera=camera,
        playback=playback,
        trajectory=trajectory,
        bounce=bounce,
        metrics=metrics,
        diagnostics=diagnostics,
    )


def build_and_save_replay_payload(
    analysis_id: str,
    *,
    physics: DeliveryPhysicsResult,
    primary_track: list[TrackingPoint],
    finalized_track: FinalizedDeliveryTrack,
    tracking_status: str,
    fps: float,
    width: int,
    height: int,
) -> ReplayPayloadV1:
    """Build replay payload during tracking job and persist to disk."""
    track_id = finalized_track.track_id
    render_track = finalized_track.render_track
    bridge_response = _safe_camera_bridge(analysis_id)
    camera, has_accepted_calibration = _resolve_replay_camera(
        analysis_id,
        physics,
        width,
        height,
    )
    measurement_validity = _resolve_measurement_validity(
        has_accepted_calibration=has_accepted_calibration,
        camera=camera,
        physics=physics,
        tracking_status=tracking_status,
        primary_track=render_track or primary_track,
    )
    trajectory = _build_trajectory(
        physics=physics,
        primary_track=render_track or primary_track,
        finalized_track=finalized_track,
        measurement_validity=measurement_validity,
    )
    playback = _build_playback(
        physics=physics,
        primary_track=render_track or primary_track,
        fps=fps,
    )
    bounce = _build_bounce(physics, measurement_validity)
    metrics = _build_metrics(physics, measurement_validity, bounce)
    diagnostics = _build_diagnostics(
        measurement_validity=measurement_validity,
        physics=physics,
        tracking_status=tracking_status,
        trajectory=trajectory,
        bridge_response=bridge_response,
        camera=camera,
        source_track_id=track_id,
        tracking_job_id=finalized_track.tracking_job_id,
        generated_at=_diagnostics_timestamp(finalized_track.generated_at),
        calibration_id=_resolve_calibration_id(analysis_id),
        consistency_errors=finalized_track.source_consistency.errors,
    )
    payload = ReplayPayloadV1(
        analysis_id=analysis_id,
        measurement_validity=measurement_validity,
        camera=camera,
        playback=playback,
        trajectory=trajectory,
        bounce=bounce,
        metrics=metrics,
        diagnostics=diagnostics,
    )
    save_replay_payload(analysis_id, payload)
    return payload


def _safe_camera_bridge(analysis_id: str) -> CameraBridgeResponse | None:
    try:
        return load_analysis_camera_bridge(analysis_id)
    except VideoAnalysisServiceError:
        return None


def _replay_camera_from_wicket_box_snapshot(
    snapshot: dict[str, object],
    *,
    width: int,
    height: int,
) -> ReplayCamera:
    coefficients = snapshot.get("distortion_coefficients") or [0.0] * 5
    return ReplayCamera(
        source="CALIBRATED",
        calibration_source="ACCEPTED_WICKET_BOX_CALIBRATION",
        image_width=int(snapshot.get("source_image_width") or width),
        image_height=int(snapshot.get("source_image_height") or height),
        camera_matrix=snapshot["camera_matrix"],
        rotation_matrix=snapshot["rotation_matrix"],
        translation_vector=snapshot["translation_vector"],
        distortion_coefficients=[float(value) for value in coefficients],
        visualization_only=False,
    )


def _resolve_replay_camera(
    analysis_id: str,
    physics: DeliveryPhysicsResult,
    width: int,
    height: int,
) -> tuple[ReplayCamera, bool]:
    bridge_response = _safe_camera_bridge(analysis_id)
    if (
        bridge_response is not None
        and bridge_response.status == "AVAILABLE"
        and bridge_response.camera is not None
    ):
        return _build_replay_camera(bridge_response, physics, width, height)

    snapshot = load_active_accepted_wicket_box_calibration(analysis_id)
    if snapshot is not None:
        return (
            _replay_camera_from_wicket_box_snapshot(
                snapshot,
                width=width,
                height=height,
            ),
            True,
        )

    return _build_replay_camera(None, physics, width, height)


def _physics_has_world_samples(physics: DeliveryPhysicsResult) -> bool:
    return any(
        sample.world_x_m is not None and sample.world_y_m is not None
        for sample in physics.trajectory_samples
    )


def _build_replay_camera(
    bridge_response: CameraBridgeResponse | None,
    physics: DeliveryPhysicsResult,
    width: int,
    height: int,
) -> tuple[ReplayCamera, bool]:
    if (
        bridge_response is not None
        and bridge_response.status == "AVAILABLE"
        and bridge_response.camera is not None
    ):
        bridge = bridge_response.camera
        accepted = bridge.accepted and bridge.source in {
            "ACCEPTED_SCENE_CALIBRATION",
            "ACCEPTED_WICKET_BOX_CALIBRATION",
        }
        if accepted:
            return (
                ReplayCamera(
                    source="CALIBRATED",
                    calibration_source=bridge.source,
                    image_width=bridge.image_width,
                    image_height=bridge.image_height,
                    camera_matrix=bridge.camera_matrix,
                    rotation_matrix=bridge.rotation_matrix,
                    translation_vector=bridge.translation_vector,
                    distortion_coefficients=bridge.distortion.coefficients,
                    visualization_only=False,
                ),
                True,
            )
        preset_name = bridge.candidate_id or bridge.source
        return (
            ReplayCamera(
                source="PRESET_VISUALIZATION",
                preset_name=preset_name,
                image_width=bridge.image_width,
                image_height=bridge.image_height,
                camera_matrix=bridge.camera_matrix,
                rotation_matrix=bridge.rotation_matrix,
                translation_vector=bridge.translation_vector,
                distortion_coefficients=bridge.distortion.coefficients,
                visualization_only=True,
            ),
            False,
        )

    calibration = physics.calibration
    return (
        ReplayCamera(
            source="UNAVAILABLE",
            image_width=calibration.image_width or width,
            image_height=calibration.image_height or height,
            camera_matrix=calibration.camera_matrix,
            rotation_matrix=calibration.rotation_matrix,
            translation_vector=calibration.translation_vector,
            distortion_coefficients=calibration.distortion_coefficients,
            visualization_only=False,
        ),
        False,
    )


def _resolve_measurement_validity(
    *,
    has_accepted_calibration: bool,
    camera: ReplayCamera,
    physics: DeliveryPhysicsResult,
    tracking_status: str,
    primary_track: list[TrackingPoint],
) -> MeasurementValidity:
    if tracking_status != "ready" or not primary_track:
        return "INSUFFICIENT_EVIDENCE"
    if camera.visualization_only or camera.source == "PRESET_VISUALIZATION":
        return "VISUALIZATION_ONLY"
    if not has_accepted_calibration:
        if (
            physics.calibration.mode == "METRIC_3D"
            and _physics_has_world_samples(physics)
            and (
                physics.geometry_validation is None
                or physics.geometry_validation.validity == "VALID_METRIC_3D"
            )
        ):
            return "CALIBRATED"
        return "IMAGE_SPACE_ONLY"
    if physics.status in {"INSUFFICIENT_EVIDENCE", "FAILED"}:
        return "INSUFFICIENT_EVIDENCE"
    if physics.calibration.mode == "IMAGE_SPACE_ONLY":
        return "IMAGE_SPACE_ONLY"
    if physics.geometry_validation is not None:
        if physics.geometry_validation.validity != "VALID_METRIC_3D":
            return "IMAGE_SPACE_ONLY"
    elif not _physics_has_world_samples(physics):
        return "IMAGE_SPACE_ONLY"
    return "CALIBRATED"


def _delivery_window(
    physics: DeliveryPhysicsResult,
    primary_track: list[TrackingPoint],
) -> tuple[int | None, int | None]:
    interval = physics.delivery_interval
    start = interval.start_frame
    end = interval.end_frame
    if start is None and primary_track:
        start = primary_track[0].frame_index
    if end is None and primary_track:
        end = primary_track[-1].frame_index
    return start, end


def _build_playback(
    *,
    physics: DeliveryPhysicsResult,
    primary_track: list[TrackingPoint],
    fps: float,
) -> ReplayPlayback:
    start, end = _delivery_window(physics, primary_track)
    duration = None
    if (
        start is not None
        and end is not None
        and fps > 0
    ):
        duration = round((end - start + 1) / fps, 6)
    return ReplayPlayback(
        fps=round(fps, 6) if fps > 0 else None,
        duration_seconds=duration,
        start_frame_index=start,
        end_frame_index=end,
    )


def _build_trajectory(
    *,
    physics: DeliveryPhysicsResult,
    primary_track: list[TrackingPoint],
    finalized_track: FinalizedDeliveryTrack | None,
    measurement_validity: MeasurementValidity,
) -> list[ReplayTrajectorySample]:
    start, end = _delivery_window(physics, primary_track)
    physics_world_by_frame = _physics_world_by_frame(physics, measurement_validity)
    if finalized_track is not None and finalized_track.render_track:
        return _trajectory_from_finalized(
            finalized_track,
            measurement_validity,
            start,
            end,
            physics_world_by_frame=physics_world_by_frame,
        )
    samples = physics.trajectory_samples
    if samples:
        return _trajectory_from_physics(
            samples,
            measurement_validity,
            start,
            end,
        )
    if not primary_track:
        return []
    return _trajectory_from_tracking(
        primary_track,
        measurement_validity,
        start,
        end,
    )


def _physics_world_by_frame(
    physics: DeliveryPhysicsResult,
    measurement_validity: MeasurementValidity,
) -> dict[int, WorldPoint3D]:
    if measurement_validity != "CALIBRATED":
        return {}
    return {
        sample.frame_index: WorldPoint3D(
            x_m=sample.world_x_m,
            y_m=sample.world_y_m,
            z_m=sample.world_z_m if sample.world_z_m is not None else 0.0,
        )
        for sample in physics.trajectory_samples
        if sample.world_x_m is not None and sample.world_y_m is not None
    }


def _trajectory_from_finalized(
    finalized_track: FinalizedDeliveryTrack,
    measurement_validity: MeasurementValidity,
    start: int | None,
    end: int | None,
    *,
    physics_world_by_frame: dict[int, WorldPoint3D] | None = None,
) -> list[ReplayTrajectorySample]:
    clipped = [
        point
        for point in finalized_track.render_track
        if (start is None or point.frame_index >= start)
        and (end is None or point.frame_index <= end)
    ]
    world_by_frame = {
        point.frame_index: (
            point.world_x_m,
            point.world_y_m,
            point.world_z_m,
        )
        for collection in (
            finalized_track.observed,
            finalized_track.recovered,
            finalized_track.physics_reconstructed,
            finalized_track.projected,
        )
        for point in collection
        if point.world_x_m is not None and point.world_y_m is not None
    }
    clipped.sort(key=lambda item: (item.timestamp_seconds, item.frame_index))
    trajectory: list[ReplayTrajectorySample] = []
    for point in clipped:
        world = None
        if measurement_validity == "CALIBRATED":
            world = (physics_world_by_frame or {}).get(point.frame_index)
        if world is None:
            world_tuple = world_by_frame.get(point.frame_index)
            if (
                measurement_validity == "CALIBRATED"
                and world_tuple is not None
                and world_tuple[0] is not None
                and world_tuple[1] is not None
            ):
                world = WorldPoint3D(
                    x_m=world_tuple[0],
                    y_m=world_tuple[1],
                    z_m=world_tuple[2] if world_tuple[2] is not None else 0.0,
                )
        trajectory.append(
            ReplayTrajectorySample(
                frame_index=point.frame_index,
                timestamp_seconds=point.timestamp_seconds,
                world_position=world,
                image_position=ImagePoint2D(x=point.x, y=point.y),
                provenance=_TRACKING_PROVENANCE.get(
                    point.provenance,
                    "PHYSICS_FITTED",
                ),
                confidence=point.confidence,
            )
        )
    return trajectory


def _trajectory_from_physics(
    samples: list[TrajectorySample],
    measurement_validity: MeasurementValidity,
    start: int | None,
    end: int | None,
) -> list[ReplayTrajectorySample]:
    clipped = [
        sample
        for sample in samples
        if (start is None or sample.frame_index >= start)
        and (end is None or sample.frame_index <= end)
    ]
    clipped.sort(key=lambda item: (item.timestamp_seconds, item.frame_index))
    trajectory: list[ReplayTrajectorySample] = []
    for sample in clipped:
        world = None
        if (
            measurement_validity == "CALIBRATED"
            and sample.world_x_m is not None
            and sample.world_y_m is not None
        ):
            world = WorldPoint3D(
                x_m=sample.world_x_m,
                y_m=sample.world_y_m,
                z_m=sample.world_z_m if sample.world_z_m is not None else 0.0,
            )
        image = ImagePoint2D(x=sample.pixel_x, y=sample.pixel_y)
        trajectory.append(
            ReplayTrajectorySample(
                frame_index=sample.frame_index,
                timestamp_seconds=sample.timestamp_seconds,
                world_position=world,
                image_position=image,
                provenance=_PHYSICS_PROVENANCE.get(
                    sample.provenance,
                    "PHYSICS_FITTED",
                ),
                confidence=sample.confidence,
            )
        )
    return trajectory


def _trajectory_from_tracking(
    primary_track: list[TrackingPoint],
    measurement_validity: MeasurementValidity,
    start: int | None,
    end: int | None,
) -> list[ReplayTrajectorySample]:
    clipped = [
        point
        for point in primary_track
        if (start is None or point.frame_index >= start)
        and (end is None or point.frame_index <= end)
    ]
    clipped.sort(key=lambda item: (item.timestamp_seconds, item.frame_index))
    trajectory: list[ReplayTrajectorySample] = []
    for point in clipped:
        trajectory.append(
            ReplayTrajectorySample(
                frame_index=point.frame_index,
                timestamp_seconds=point.timestamp_seconds,
                world_position=None,
                image_position=ImagePoint2D(x=point.x, y=point.y),
                provenance=_TRACKING_PROVENANCE.get(
                    point.provenance,
                    "RECOVERED",
                ),
                confidence=point.confidence,
            )
        )
    _ = measurement_validity
    return trajectory


def _build_bounce(
    physics: DeliveryPhysicsResult,
    measurement_validity: MeasurementValidity,
) -> ReplayBounce:
    bounce = physics.bounce
    if bounce.status == "INSUFFICIENT_EVIDENCE":
        return ReplayBounce(
            detected=False,
            status="UNAVAILABLE",
            unavailable_reason="No reliable bounce point detected.",
        )

    image = None
    if bounce.pixel_x is not None and bounce.pixel_y is not None:
        image = ImagePoint2D(x=bounce.pixel_x, y=bounce.pixel_y)

    world = None
    if (
        measurement_validity == "CALIBRATED"
        and bounce.world_x_m is not None
        and bounce.world_y_m is not None
    ):
        world = WorldPoint3D(
            x_m=bounce.world_x_m,
            y_m=bounce.world_y_m,
            z_m=0.0,
        )

    metric_status: MetricAvailabilityStatus = (
        "AVAILABLE"
        if bounce.status in {"DETECTED", "ESTIMATED"}
        else "UNAVAILABLE"
    )
    return ReplayBounce(
        detected=True,
        frame_index=bounce.frame_index,
        timestamp_seconds=bounce.timestamp_seconds,
        world_position=world,
        image_position=image,
        confidence=bounce.confidence_score,
        status=metric_status,
        unavailable_reason=(
            None
            if metric_status == "AVAILABLE"
            else "Bounce hypothesis is not reliable enough for metrics."
        ),
    )


def _build_metrics(
    physics: DeliveryPhysicsResult,
    measurement_validity: MeasurementValidity,
    bounce: ReplayBounce,
) -> ReplayMetrics:
    if measurement_validity in {
        "IMAGE_SPACE_ONLY",
        "VISUALIZATION_ONLY",
        "INSUFFICIENT_EVIDENCE",
    }:
        reason = _metrics_unavailable_reason(measurement_validity)
        return ReplayMetrics(
            release_speed_kmh=unavailable_metric("km/h", reason),
            average_pre_bounce_speed_kmh=unavailable_metric("km/h", reason),
            speed_at_bounce_kmh=unavailable_metric("km/h", reason),
            overall_stump_to_stump_speed_kmh=unavailable_metric("km/h", reason),
            delivery_length_m=unavailable_metric("m", reason),
            estimated_lateral_deviation_m=unavailable_metric("m", reason),
        )

    speed = physics.speed
    overall = physics.overall_stump_to_stump
    lateral = physics.pre_bounce_lateral_movement
    metrics = ReplayMetrics(
        release_speed_kmh=_speed_metric(
            speed.earliest_measured_speed_kmh,
            method="fitted_initial_velocity",
            confidence=speed.confidence,
            unavailable_reason=speed.unavailable_reason,
        ),
        average_pre_bounce_speed_kmh=_speed_metric(
            speed.average_pre_bounce_speed_kmh,
            method="average_pre_bounce_fitted",
            confidence=speed.confidence,
            unavailable_reason=speed.unavailable_reason,
        ),
        speed_at_bounce_kmh=_speed_metric(
            speed.speed_at_bounce_kmh,
            method="fitted_at_bounce",
            confidence=speed.confidence,
            unavailable_reason=speed.unavailable_reason,
        ),
        overall_stump_to_stump_speed_kmh=_overall_stump_metric(overall),
        delivery_length_m=_length_metric(physics, bounce),
        estimated_lateral_deviation_m=_lateral_metric(lateral),
    )
    return metrics


def _overall_stump_metric(overall) -> ReplayMetric:
    if overall.status == "UNAVAILABLE" or overall.speed_kph is None:
        return unavailable_metric(
            "km/h",
            overall.unavailable_reason
            or "Overall stump-to-stump speed requires both wicket plane crossings.",
        )
    method = "path_average_stump_to_stump"
    if overall.status == "PARTIALLY_PROJECTED":
        method = (
            f"path_average_stump_to_stump_{round(overall.projected_fraction * 100)}pct_projected"
        )
    return ReplayMetric(
        value=overall.speed_kph,
        unit="km/h",
        confidence=_grade_to_float(overall.confidence),
        method=method,
        status="AVAILABLE",
    )


def _metrics_unavailable_reason(validity: MeasurementValidity) -> str:
    if validity == "IMAGE_SPACE_ONLY":
        return (
            "Metric measurements require accepted calibration; "
            "image-space tracking is preserved without world metrics."
        )
    if validity == "VISUALIZATION_ONLY":
        return "Preset visualization camera does not unlock measured metrics."
    return "Insufficient delivery evidence for metric measurements."


def _speed_metric(
    value: float | None,
    *,
    method: str,
    confidence: str,
    unavailable_reason: str | None,
) -> ReplayMetric:
    if value is None or confidence == "INSUFFICIENT_EVIDENCE":
        return unavailable_metric(
            "km/h",
            unavailable_reason or "Speed could not be measured reliably.",
        )
    confidence_score = _grade_to_float(confidence)
    return ReplayMetric(
        value=value,
        unit="km/h",
        confidence=confidence_score,
        method=method,
        status="AVAILABLE",
    )


def _length_metric(
    physics: DeliveryPhysicsResult,
    bounce: ReplayBounce,
) -> ReplayMetric:
    if (
        bounce.status != "AVAILABLE"
        or bounce.detected is not True
        or physics.bounce.distance_from_striker_wicket_m is None
    ):
        return unavailable_metric(
            "m",
            "Delivery length requires a reliable metric bounce point.",
        )
    return ReplayMetric(
        value=physics.bounce.distance_from_striker_wicket_m,
        unit="m",
        confidence=bounce.confidence,
        method="bounce_distance_from_striker_wicket",
        status="AVAILABLE",
    )


def _lateral_metric(lateral) -> ReplayMetric:
    if (
        lateral.movement_m is None
        or lateral.confidence == "INSUFFICIENT_EVIDENCE"
    ):
        return unavailable_metric(
            "m",
            lateral.unavailable_reason
            or "Estimated lateral deviation requires sufficient pre-bounce evidence.",
        )
    return ReplayMetric(
        value=abs(lateral.movement_m),
        unit="m",
        confidence=_grade_to_float(lateral.confidence),
        method="pre_bounce_lateral_movement",
        status="AVAILABLE",
    )


def _grade_to_float(grade: str) -> float | None:
    mapping = {
        "HIGH": 0.9,
        "MEDIUM": 0.65,
        "LOW": 0.35,
        "INSUFFICIENT_EVIDENCE": None,
    }
    return mapping.get(grade)


def _build_diagnostics(
    *,
    measurement_validity: MeasurementValidity,
    physics: DeliveryPhysicsResult,
    tracking_status: str,
    trajectory: list[ReplayTrajectorySample],
    bridge_response: CameraBridgeResponse | None,
    camera: ReplayCamera,
    source_track_id: str | None = None,
    tracking_job_id: str | None = None,
    generated_at: str | None = None,
    calibration_id: str | None = None,
    consistency_errors: list[str] | None = None,
) -> ReplayDiagnostics:
    warnings = list(physics.warnings)
    if bridge_response is not None:
        warnings.extend(bridge_response.warnings)
    if consistency_errors:
        warnings.extend(consistency_errors)
    geometry = physics.geometry_validation
    geometry_validity: GeometryValidity | None = (
        geometry.validity if geometry is not None else None
    )
    if (
        source_track_id
        and tracking_job_id
        and source_track_id != tracking_job_id
        and not source_track_id.startswith("tracking_job_")
    ):
        warnings.append(
            f"Track identifier mismatch: source_track_id={source_track_id} "
            f"tracking_job_id={tracking_job_id}."
        )
    if measurement_validity == "IMAGE_SPACE_ONLY":
        reason = (
            geometry.reason
            if geometry is not None and geometry.validity != "VALID_METRIC_3D"
            else physics.calibration.failure_reason
        ) or (
            "Tracking completed without accepted calibration; "
            "world metrics remain unavailable."
        )
        warnings.append(reason)
    if measurement_validity == "VISUALIZATION_ONLY":
        warnings.append(
            "Camera is for visualization only; measured metrics are suppressed."
        )

    if tracking_status != "ready" or not trajectory:
        return ReplayDiagnostics(
            status="INSUFFICIENT_EVIDENCE",
            measurement_validity=measurement_validity,
            geometry_validity=geometry_validity,
            source_track_id=source_track_id,
            tracking_job_id=tracking_job_id,
            generated_at=generated_at,
            calibration_id=calibration_id,
            calibration_source=camera.calibration_source,
            mean_reprojection_px=geometry.mean_reprojection_px if geometry else None,
            median_reprojection_px=geometry.median_reprojection_px if geometry else None,
            p95_reprojection_px=geometry.p95_reprojection_px if geometry else None,
            max_reprojection_px=geometry.max_reprojection_px if geometry else None,
            in_pitch_fraction=geometry.in_pitch_fraction if geometry else None,
            warnings=warnings,
            unavailable_reason=(
                "A reliable delivery track is unavailable for replay."
            ),
        )

    if measurement_validity == "INSUFFICIENT_EVIDENCE":
        return ReplayDiagnostics(
            status="INSUFFICIENT_EVIDENCE",
            measurement_validity=measurement_validity,
            geometry_validity=geometry_validity,
            source_track_id=source_track_id,
            tracking_job_id=tracking_job_id,
            generated_at=generated_at,
            calibration_id=calibration_id,
            calibration_source=camera.calibration_source,
            mean_reprojection_px=geometry.mean_reprojection_px if geometry else None,
            median_reprojection_px=geometry.median_reprojection_px if geometry else None,
            p95_reprojection_px=geometry.p95_reprojection_px if geometry else None,
            max_reprojection_px=geometry.max_reprojection_px if geometry else None,
            in_pitch_fraction=geometry.in_pitch_fraction if geometry else None,
            warnings=warnings,
            unavailable_reason=(
                physics.fit_diagnostics.optimizer_status
                if physics.status in {"INSUFFICIENT_EVIDENCE", "FAILED"}
                else "Delivery evidence is insufficient for calibrated replay."
            ),
        )

    degraded_reason = None
    if (
        geometry is not None
        and geometry.validity != "VALID_METRIC_3D"
        and measurement_validity == "IMAGE_SPACE_ONLY"
    ):
        degraded_reason = geometry.reason

    return ReplayDiagnostics(
        status="READY",
        measurement_validity=measurement_validity,
        geometry_validity=geometry_validity,
        source_track_id=source_track_id,
        tracking_job_id=tracking_job_id,
        generated_at=generated_at,
        calibration_id=calibration_id,
        calibration_source=camera.calibration_source,
        mean_reprojection_px=geometry.mean_reprojection_px if geometry else None,
        median_reprojection_px=geometry.median_reprojection_px if geometry else None,
        p95_reprojection_px=geometry.p95_reprojection_px if geometry else None,
        max_reprojection_px=geometry.max_reprojection_px if geometry else None,
        in_pitch_fraction=geometry.in_pitch_fraction if geometry else None,
        warnings=warnings,
        unavailable_reason=degraded_reason,
    )


def _resolve_replay_track_ids(
    analysis_id: str,
    *,
    finalized_track: FinalizedDeliveryTrack | None,
    summary,
) -> tuple[str | None, str | None, str | None]:
    if finalized_track is not None:
        return (
            finalized_track.tracking_job_id,
            finalized_track.track_id,
            _diagnostics_timestamp(finalized_track.generated_at),
        )
    summary_job = getattr(summary, "tracking_job_id", None)
    summary_source = getattr(summary, "source_track_id", None)
    if isinstance(summary_job, str) and summary_job:
        return summary_job, summary_source, None
    fingerprint = _load_replay_artifact_fingerprint(analysis_id)
    source_track_id = fingerprint.source_track_id or fingerprint.tracking_job_id
    return (
        fingerprint.tracking_job_id,
        source_track_id,
        fingerprint.generated_at,
    )


def _diagnostics_timestamp(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except ValueError:
        return text


def _resolve_calibration_id(analysis_id: str) -> str | None:
    try:
        snapshot = load_active_accepted_wicket_box_calibration(analysis_id)
    except VideoAnalysisServiceError:
        return None
    if snapshot is None:
        return None
    accepted_at = snapshot.get("accepted_at")
    if accepted_at is None:
        return None
    return _diagnostics_timestamp(accepted_at)


def _load_replay_artifact_fingerprint(analysis_id: str) -> _ReplayArtifactFingerprint:
    metadata_path = (
        VIDEO_ANALYSIS_ROOT / analysis_id / "reports" / "analysis_metadata.json"
    )
    metadata: dict[str, object] = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            metadata = {}

    finalized_path = (
        VIDEO_ANALYSIS_ROOT
        / analysis_id
        / "tracking"
        / "finalized_track.json"
    )
    finalized: dict[str, object] = {}
    if finalized_path.is_file():
        try:
            finalized = json.loads(finalized_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            finalized = {}

    tracking_job_id = finalized.get("tracking_job_id") or metadata.get(
        "tracking_job_id"
    )
    source_track_id = finalized.get("track_id")
    generated_at = _diagnostics_timestamp(finalized.get("generated_at"))
    if not isinstance(tracking_job_id, str):
        tracking_job_id = None
    if not isinstance(source_track_id, str):
        source_track_id = None

    return _ReplayArtifactFingerprint(
        analysis_id=analysis_id,
        tracking_job_id=tracking_job_id,
        source_track_id=source_track_id,
        generated_at=generated_at,
        calibration_id=_resolve_calibration_id(analysis_id),
    )


def _try_load_saved_replay_payload(path: Path) -> ReplayPayloadV1 | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    try:
        return ReplayPayloadV1.model_validate(payload)
    except ValueError:
        return None


def _saved_replay_payload_is_current(
    payload: ReplayPayloadV1,
    fingerprint: _ReplayArtifactFingerprint,
) -> bool:
    diagnostics = payload.diagnostics
    if payload.analysis_id != fingerprint.analysis_id:
        return False

    expected_job = fingerprint.tracking_job_id
    saved_job = diagnostics.tracking_job_id
    if expected_job is not None:
        if saved_job != expected_job:
            return False
    elif saved_job is not None:
        if not _tracking_result_path(fingerprint.analysis_id).is_file():
            return False

    expected_source = fingerprint.source_track_id
    saved_source = diagnostics.source_track_id
    if expected_source is not None:
        if saved_source != expected_source:
            return False
    elif saved_source is not None:
        return False

    expected_generated_at = fingerprint.generated_at
    saved_generated_at = diagnostics.generated_at
    if expected_generated_at is not None:
        if saved_generated_at is None or saved_generated_at != expected_generated_at:
            return False

    expected_calibration_id = fingerprint.calibration_id
    saved_calibration_id = diagnostics.calibration_id
    if expected_calibration_id is not None:
        if (
            saved_calibration_id is None
            or saved_calibration_id != expected_calibration_id
        ):
            return False

    return True


def _insufficient_replay_payload(
    analysis_id: str,
    *,
    reason: str,
) -> ReplayPayloadV1:
    return ReplayPayloadV1(
        analysis_id=analysis_id,
        measurement_validity="INSUFFICIENT_EVIDENCE",
        camera=ReplayCamera(source="UNAVAILABLE", visualization_only=False),
        playback=ReplayPlayback(),
        trajectory=[],
        bounce=ReplayBounce(status="UNAVAILABLE", unavailable_reason=reason),
        metrics=ReplayMetrics(
            release_speed_kmh=unavailable_metric("km/h", reason),
            average_pre_bounce_speed_kmh=unavailable_metric("km/h", reason),
            speed_at_bounce_kmh=unavailable_metric("km/h", reason),
            overall_stump_to_stump_speed_kmh=unavailable_metric("km/h", reason),
            delivery_length_m=unavailable_metric("m", reason),
            estimated_lateral_deviation_m=unavailable_metric("m", reason),
        ),
        diagnostics=ReplayDiagnostics(
            status="INSUFFICIENT_EVIDENCE",
            measurement_validity="INSUFFICIENT_EVIDENCE",
            unavailable_reason=reason,
            warnings=[reason],
        ),
    )
