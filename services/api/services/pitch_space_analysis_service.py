"""Orchestration and persistence for the pitch-space setup/fit stage."""

from __future__ import annotations

import json
from pathlib import Path
import time

import cv2
import numpy as np

from ..schemas.pitch_space_analysis import (
    CameraStabilityResult,
    ImageSpaceTrackPoint,
    PitchFitResult,
    PitchSpaceDeliveryAnalysisV1,
    RecentPitchSpaceAnalyses,
    RecentPitchSpaceAnalysis,
    StageTimings,
)
from .pitch_space_bounce_service import estimate_pitch_space_bounce
from .pitch_space_metrics_service import (
    calculate_line_and_length,
    estimate_lateral_movement,
    estimate_planar_speed,
)
from .pitch_space_track_service import convert_track_to_pitch_space
from .setup_frame_selection_service import select_setup_frame
from .two_wicket_pitch_fit_service import fit_two_wicket_pitch
from .video_analysis_service import (
    VIDEO_ANALYSIS_ROOT,
    VideoAnalysisServiceError,
    load_video_analysis,
)
from .video_ball_tracking_service import (
    VideoBallTrackingError,
    load_video_ball_tracking_result,
)
from .wicket_box_stabilization_service import (
    assess_camera_stability,
    stabilize_wicket_boxes,
)
from .wicket_observation_service import (
    load_wicket_observation,
    run_wicket_observation,
)


RESULT_FILENAME = "pitch_space_delivery_analysis_v1.json"
OVERLAY_FILENAME = "pitch_space_setup_overlay_v1.jpg"


def _write_result(result: PitchSpaceDeliveryAnalysisV1) -> float:
    started = time.perf_counter()
    reports = VIDEO_ANALYSIS_ROOT / result.analysis_id / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    destination = reports / RESULT_FILENAME
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return (time.perf_counter() - started) * 1000


def _draw_overlay(analysis_id: str, frame_index: int, primitives: list) -> str | None:
    analysis = load_video_analysis(analysis_id)
    raw_path = VIDEO_ANALYSIS_ROOT / analysis_id / "raw" / analysis.stored_filename
    capture = cv2.VideoCapture(str(raw_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        return None
    for primitive in primitives:
        points = [(int(round(point.x)), int(round(point.y))) for point in primitive.image_points]
        if len(points) < 2:
            continue
        colour = (80, 220, 80) if primitive.primitive_type != "WICKET_BASE" else (20, 180, 255)
        if primitive.primitive_type == "POLYGON":
            cv2.polylines(frame, [np.asarray(points)], True, colour, 2, cv2.LINE_AA)
        else:
            cv2.line(frame, points[0], points[1], colour, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"Pitch-space setup frame {frame_index}",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    output = VIDEO_ANALYSIS_ROOT / analysis_id / "calibration" / OVERLAY_FILENAME
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), frame):
        return None
    return f"/static/video-analysis/{analysis_id}/calibration/{OVERLAY_FILENAME}"


def run_pitch_space_analysis(
    analysis_id: str,
    *,
    upload_ms: float | None = None,
) -> PitchSpaceDeliveryAnalysisV1:
    total_started = time.perf_counter()
    analysis = load_video_analysis(analysis_id)
    observation_started = time.perf_counter()
    source = "persisted"
    try:
        observation = load_wicket_observation(analysis_id)
    except VideoAnalysisServiceError as exc:
        if exc.status_code != 404:
            raise
        observation = run_wicket_observation(analysis_id)
        source = "newly_generated"
    observation_ms = (time.perf_counter() - observation_started) * 1000

    setup_started = time.perf_counter()
    setup = select_setup_frame(
        observation.frame_candidates,
        observation.diagnostics.raw_detections,
    )
    setup_ms = (time.perf_counter() - setup_started) * 1000

    stabilise_started = time.perf_counter()
    near, far = stabilize_wicket_boxes(setup.evaluations, source=source)
    stabilise_ms = (time.perf_counter() - stabilise_started) * 1000
    camera_stability = assess_camera_stability(near, far)

    fit_started = time.perf_counter()
    fit = fit_two_wicket_pitch(
        near,
        far,
        image_width=analysis.width,
        image_height=analysis.height,
    )
    fit_ms = (time.perf_counter() - fit_started) * 1000

    overlay_started = time.perf_counter()
    overlay_url = (
        _draw_overlay(analysis_id, setup.selected_frame_index, fit.projected_pitch)
        if setup.selected_frame_index is not None and fit.status == "READY"
        else None
    )
    overlay_ms = (time.perf_counter() - overlay_started) * 1000
    if near is None or far is None or setup.selected_frame_index is None:
        status = "INSUFFICIENT_WICKETS"
    elif fit.status != "READY":
        status = "PITCH_FIT_FAILED"
    else:
        status = "PARTIAL"
    warnings = list(dict.fromkeys([*observation.warnings, *fit.warnings, *camera_stability.warnings]))
    unavailable = ["AIRBORNE_3D_HEIGHT"]
    downstream_statuses: list[str] = []
    image_track: list[ImageSpaceTrackPoint] = []
    pitch_track = []
    bounce = None
    line_length = None
    speed = None
    movement = None
    tracking_ms = conversion_ms = bounce_ms = speed_ms = movement_ms = None
    if fit.status == "READY" and fit.image_to_pitch_homography is not None:
        tracking_started = time.perf_counter()
        try:
            tracking = load_video_ball_tracking_result(analysis_id)
        except (VideoBallTrackingError, VideoAnalysisServiceError):
            tracking = None
        tracking_ms = (time.perf_counter() - tracking_started) * 1000
        if tracking is not None and tracking.status == "ready" and tracking.primary_track:
            image_track = [
                ImageSpaceTrackPoint(
                    frame_index=point.frame_index,
                    timestamp_seconds=point.timestamp_seconds,
                    image_x_px=point.x,
                    image_y_px=point.y,
                    detection_confidence=point.confidence,
                    provenance=point.provenance,
                    track_valid=True,
                )
                for point in tracking.primary_track
            ]
            conversion_started = time.perf_counter()
            pitch_track = convert_track_to_pitch_space(
                image_track,
                fit.image_to_pitch_homography,
                pitch_fit_confidence=fit.confidence,
                camera_stability=camera_stability.status,
                unstable_after_frame=camera_stability.reliable_until_frame,
            )
            conversion_ms = (time.perf_counter() - conversion_started) * 1000
            bounce_started = time.perf_counter()
            bounce = estimate_pitch_space_bounce(
                pitch_track,
                existing_bounce=tracking.bounce,
            )
            bounce_ms = (time.perf_counter() - bounce_started) * 1000
            if bounce.bounce_frame is not None:
                pitch_track = convert_track_to_pitch_space(
                    image_track,
                    fit.image_to_pitch_homography,
                    pitch_fit_confidence=fit.confidence,
                    bounce_frame=bounce.bounce_frame,
                    camera_stability=camera_stability.status,
                    unstable_after_frame=camera_stability.reliable_until_frame,
                )
            line_length = calculate_line_and_length(bounce)
            speed_started = time.perf_counter()
            speed = estimate_planar_speed(
                pitch_track,
                bounce_frame=bounce.bounce_frame,
            )
            speed_ms = (time.perf_counter() - speed_started) * 1000
            movement_started = time.perf_counter()
            movement = estimate_lateral_movement(
                pitch_track,
                bounce_frame=bounce.bounce_frame,
            )
            movement_ms = (time.perf_counter() - movement_started) * 1000
        else:
            unavailable.append("BALL_TRACK")
            downstream_statuses.append("BALL_TRACK_UNAVAILABLE")
    else:
        unavailable.append("BALL_TRACK")
        downstream_statuses.append("BALL_TRACK_UNAVAILABLE")

    metric_states = (
        ("BOUNCE", bounce),
        ("LINE", line_length),
        ("LENGTH", line_length),
        ("ESTIMATED_PLANAR_SPEED", speed),
        ("ESTIMATED_LATERAL_MOVEMENT", movement),
    )
    for metric_name, metric in metric_states:
        if metric is None or metric.status == "UNAVAILABLE":
            unavailable.append(metric_name)
            downstream_statuses.append(f"{metric_name.replace('ESTIMATED_PLANAR_', '').replace('ESTIMATED_LATERAL_', '')}_UNAVAILABLE")
    downstream_statuses = list(dict.fromkeys(downstream_statuses))
    if status == "PARTIAL" and image_track and not downstream_statuses:
        status = "COMPLETE"
    confidence = min(
        setup.quality_score,
        fit.confidence if fit.status == "READY" else 0,
        camera_stability.confidence if camera_stability.status != "UNAVAILABLE" else 0,
    )
    result = PitchSpaceDeliveryAnalysisV1(
        analysis_id=analysis_id,
        status=status,
        source_video_url=analysis.original_video_url,
        source_filename=analysis.original_filename,
        native_width=analysis.width,
        native_height=analysis.height,
        fps=analysis.fps,
        frame_count=analysis.frame_count,
        setup_frame_decision=setup,
        wicket_observation_source=source,
        stable_near_wicket=near,
        stable_far_wicket=far,
        pitch_fit=fit,
        camera_stability=camera_stability,
        image_space_track=image_track,
        pitch_space_track=pitch_track,
        bounce=bounce,
        line=line_length,
        length=line_length,
        estimated_planar_speed=speed,
        estimated_lateral_movement=movement,
        overlay_url=overlay_url,
        overall_confidence=round(confidence, 6),
        warnings=warnings,
        unavailable_metrics=list(dict.fromkeys(unavailable)),
        downstream_statuses=downstream_statuses,
        stage_timings=StageTimings(
            upload_ms=upload_ms,
            observation_load_or_run_ms=observation_ms,
            setup_selection_ms=setup_ms,
            box_stabilisation_ms=stabilise_ms,
            pitch_fit_ms=fit_ms,
            overlay_ms=overlay_ms,
            ball_tracking_load_ms=tracking_ms,
            pitch_space_conversion_ms=conversion_ms,
            bounce_ms=bounce_ms,
            speed_ms=speed_ms,
            movement_ms=movement_ms,
            replay_preparation_ms=0.0,
            total_ms=(time.perf_counter() - total_started) * 1000,
        ),
    )
    persistence_ms = _write_result(result)
    result.stage_timings.persistence_ms = persistence_ms
    result.stage_timings.total_ms = (time.perf_counter() - total_started) * 1000
    _write_result(result)
    return result


def load_pitch_space_analysis(analysis_id: str) -> PitchSpaceDeliveryAnalysisV1:
    load_video_analysis(analysis_id)
    path = VIDEO_ANALYSIS_ROOT / analysis_id / "reports" / RESULT_FILENAME
    if not path.is_file():
        raise VideoAnalysisServiceError("Pitch-space analysis has not been generated.", status_code=404)
    try:
        return PitchSpaceDeliveryAnalysisV1.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VideoAnalysisServiceError("Stored pitch-space analysis is unavailable.", status_code=500) from exc


def clear_pitch_space_analysis(analysis_id: str) -> None:
    load_video_analysis(analysis_id)
    (VIDEO_ANALYSIS_ROOT / analysis_id / "reports" / RESULT_FILENAME).unlink(missing_ok=True)
    (VIDEO_ANALYSIS_ROOT / analysis_id / "calibration" / OVERLAY_FILENAME).unlink(missing_ok=True)


def list_recent_pitch_space_analyses(limit: int = 20) -> RecentPitchSpaceAnalyses:
    items: list[tuple[float, RecentPitchSpaceAnalysis]] = []
    if not VIDEO_ANALYSIS_ROOT.exists():
        return RecentPitchSpaceAnalyses()
    for report in VIDEO_ANALYSIS_ROOT.glob(f"analysis_*/reports/{RESULT_FILENAME}"):
        try:
            result = PitchSpaceDeliveryAnalysisV1.model_validate_json(report.read_text(encoding="utf-8"))
            items.append(
                (
                    report.stat().st_mtime,
                    RecentPitchSpaceAnalysis(
                        analysis_id=result.analysis_id,
                        status=result.status,
                        source_filename=result.source_filename,
                        report_url=f"/pitch-space-analysis/{result.analysis_id}",
                    ),
                )
            )
        except (OSError, ValueError):
            continue
    return RecentPitchSpaceAnalyses(items=[item for _, item in sorted(items, key=lambda pair: (-pair[0], pair[1].analysis_id))[:limit]])


def source_video_path(analysis_id: str) -> Path:
    analysis = load_video_analysis(analysis_id)
    return VIDEO_ANALYSIS_ROOT / analysis_id / "raw" / analysis.stored_filename


def overlay_path(analysis_id: str) -> Path:
    load_video_analysis(analysis_id)
    path = VIDEO_ANALYSIS_ROOT / analysis_id / "calibration" / OVERLAY_FILENAME
    if not path.is_file():
        raise VideoAnalysisServiceError("Pitch-space overlay is unavailable.", status_code=404)
    return path
