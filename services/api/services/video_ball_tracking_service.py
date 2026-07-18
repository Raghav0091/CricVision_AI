"""Moving Ball Tracker v1 for persisted Video Analysis detections."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

import cv2

from ..schemas.video_analysis import (
    BallCandidate,
    TrackingCandidateDiagnostic,
    TrackingCandidateScoreComponents,
    TrackingPoint,
    VideoBallDetectionsDocument,
    VideoBallTrackingDocument,
    VideoBallTrackingResultLinks,
    VideoBallTrackingResultResponse,
    VideoBallTrackingSettings,
    VideoBallTrackingSummary,
)
from .ball_detection_clip import transcode_browser_mp4
from .video_analysis_service import (
    VIDEO_ANALYSIS_ROOT,
    VideoAnalysisServiceError,
    load_video_analysis,
)
from .video_ball_detection_job_store import utc_now
from .video_ball_tracking_job_store import video_ball_tracking_job_store


MAX_RECOVERABLE_GAP = 6
MINIMUM_OBSERVED_POINTS = 3
STATIC_RADIUS_NORMALIZED = 0.012
BASE_GATE_NORMALIZED = 0.025
MAXIMUM_GATE_NORMALIZED = 0.16
HISTORY_POINTS = 8
MINIMUM_LINK_SCORE = 0.15
TRACKING_RESULT_FILENAME = "tracking_result.json"
TRACKING_CSV_FILENAME = "tracking_points.csv"
TRACKING_SUMMARY_FILENAME = "tracking_summary.json"
TRACKING_VIDEO_FILENAME = "tracking_debug.mp4"


class VideoBallTrackingError(Exception):
    def __init__(self, message: str, *, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass
class RawTrackingCandidate:
    frame_index: int
    timestamp_seconds: float
    candidate_id: str
    confidence: float
    x: float
    y: float
    normalized_x: float
    normalized_y: float
    width_pixels: float
    height_pixels: float
    area_pixels: float
    inside_pitch_corridor: bool | None
    static_likelihood: float = 0.0


@dataclass
class Tracklet:
    observations: list[RawTrackingCandidate]
    link_score: float = 0.0
    score_components: dict[str, TrackingCandidateScoreComponents] = field(
        default_factory=dict
    )
    prediction_errors: dict[str, float] = field(default_factory=dict)


def validate_video_ball_tracking_input(analysis_id: str) -> None:
    analysis = load_video_analysis(analysis_id)
    if not math.isfinite(analysis.fps) or analysis.fps <= 0:
        raise VideoBallTrackingError(
            "Prepared analysis has an invalid FPS value.",
            status_code=400,
        )
    if not _detections_path(analysis_id).is_file():
        raise VideoBallTrackingError(
            "Every-frame detections are missing. Run Ball Detection first.",
            status_code=409,
        )


def mark_video_ball_tracking_queued(analysis_id: str, job_id: str) -> None:
    now = utc_now()
    _update_analysis_metadata(
        analysis_id,
        tracking_status="tracking_queued",
        tracking_job_id=job_id,
        tracking_started_at=_iso(now),
        tracking_completed_at=None,
        tracking_summary_url=None,
        tracking_video_url=None,
        updated_at=_iso(now),
    )


def run_video_ball_tracking_job(analysis_id: str, job_id: str) -> None:
    try:
        summary, primary_track = _process_video_ball_tracking(
            analysis_id,
            job_id,
        )
        links = VideoBallTrackingResultLinks(
            tracking_video_url=summary.tracking_video_url,
            tracking_json_url=summary.tracking_json_url,
            tracking_csv_url=summary.tracking_csv_url,
            tracking_summary_url=summary.tracking_summary_url,
        )
        is_ready = summary.status == "ready"
        video_ball_tracking_job_store.update(
            job_id,
            success=is_ready,
            status=summary.status,
            progress=100,
            error_message=None if is_ready else summary.message,
            result=links.model_dump(mode="json"),
            message=summary.message,
        )
        _update_analysis_metadata(
            analysis_id,
            tracking_status=(
                "tracking_complete"
                if is_ready
                else "tracking_no_reliable_track"
            ),
            tracking_completed_at=_iso(utc_now()),
            tracking_summary_url=summary.tracking_summary_url,
            tracking_video_url=summary.tracking_video_url,
            updated_at=_iso(utc_now()),
        )
        _ = primary_track
    except (VideoBallTrackingError, VideoAnalysisServiceError) as exc:
        message = getattr(exc, "message", str(exc))
        _mark_job_failed(analysis_id, job_id, message)
    except Exception as exc:
        _mark_job_failed(
            analysis_id,
            job_id,
            f"Moving-ball tracking failed: {type(exc).__name__}.",
        )


def load_video_ball_tracking_result(
    analysis_id: str,
) -> VideoBallTrackingResultResponse:
    load_video_analysis(analysis_id)
    output_dir = _tracking_output_dir(analysis_id)
    summary_path = output_dir / TRACKING_SUMMARY_FILENAME
    result_path = output_dir / TRACKING_RESULT_FILENAME
    csv_path = output_dir / TRACKING_CSV_FILENAME
    video_path = output_dir / TRACKING_VIDEO_FILENAME
    if not summary_path.is_file() or not result_path.is_file():
        raise VideoBallTrackingError(
            "Moving-ball tracking has not completed.",
            status_code=404,
        )
    try:
        summary = VideoBallTrackingSummary.model_validate(
            json.loads(summary_path.read_text(encoding="utf-8"))
        )
        document = VideoBallTrackingDocument.model_validate(
            json.loads(result_path.read_text(encoding="utf-8"))
        )
    except FileNotFoundError as exc:
        raise VideoBallTrackingError(
            "Saved tracking output files are missing.",
            status_code=404,
        ) from exc
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise VideoBallTrackingError(
            "Saved tracking results are unavailable.",
            status_code=500,
        ) from exc
    if (
        summary.analysis_id != analysis_id
        or document.analysis_id != analysis_id
        or summary.status != document.status
        or not csv_path.is_file()
        or not video_path.is_file()
    ):
        raise VideoBallTrackingError(
            "Saved tracking results are incomplete.",
            status_code=404,
        )
    return VideoBallTrackingResultResponse(
        success=summary.status == "ready",
        status=summary.status,
        analysis_id=analysis_id,
        summary=summary,
        primary_track=document.primary_track,
        message=summary.message,
    )


def _process_video_ball_tracking(
    analysis_id: str,
    job_id: str,
) -> tuple[VideoBallTrackingSummary, list[TrackingPoint]]:
    analysis = load_video_analysis(analysis_id)
    validate_video_ball_tracking_input(analysis_id)
    started_at = utc_now()
    started_clock = time.perf_counter()
    output_dir = _tracking_output_dir(analysis_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_previous_tracking_outputs(output_dir)
    _update_analysis_metadata(
        analysis_id,
        tracking_status="tracking_ball",
        updated_at=_iso(utc_now()),
    )

    _update_job(
        job_id,
        "loading_detections",
        5,
        "Loading every-frame detections...",
    )
    detection_document = _load_detection_document(
        analysis_id,
        analysis.frame_count,
    )
    candidates = _flatten_candidates(detection_document)

    _update_job(
        job_id,
        "analysing_candidates",
        15,
        "Analysing moving and stationary candidates...",
    )
    _assign_static_likelihoods(candidates)

    _update_job(
        job_id,
        "building_track",
        30,
        "Building the primary moving-ball track...",
    )
    primary_tracklet, best_components = _build_primary_track(candidates)

    _update_job(
        job_id,
        "recovering_gaps",
        50,
        "Recovering validated short gaps...",
    )
    reliable = (
        primary_tracklet is not None
        and _is_reliable_track(primary_tracklet)
    )
    primary_track = (
        _build_tracking_points(primary_tracklet, analysis.fps)
        if reliable and primary_tracklet is not None
        else []
    )
    candidate_diagnostics = _candidate_diagnostics(
        candidates,
        primary_tracklet if reliable else None,
        best_components,
    )
    track_confidence = (
        _track_confidence(primary_tracklet, primary_track)
        if reliable and primary_tracklet is not None
        else 0.0
    )

    _update_job(
        job_id,
        "rendering_video",
        60,
        "Generating the complete tracking debug video...",
    )
    tracking_video_path = _render_tracking_video(
        analysis_id=analysis_id,
        stored_filename=analysis.stored_filename,
        total_frames=analysis.frame_count,
        fps=analysis.fps,
        width=analysis.width,
        height=analysis.height,
        primary_track=primary_track,
        candidates=candidates,
        track_confidence=track_confidence,
        job_id=job_id,
    )
    output_frame_count, output_fps = _verify_output_video(tracking_video_path)
    if output_frame_count != analysis.frame_count:
        tracking_video_path.unlink(missing_ok=True)
        raise VideoBallTrackingError(
            "Tracking debug video frame count does not match the original."
        )
    if abs(output_fps - analysis.fps) > 0.01:
        tracking_video_path.unlink(missing_ok=True)
        raise VideoBallTrackingError(
            "Tracking debug video FPS does not match the original."
        )

    _update_job(
        job_id,
        "saving_results",
        92,
        "Saving tracking results...",
    )
    completed_at = utc_now()
    status = "ready" if reliable else "no_reliable_track"
    message = (
        "Moving Ball Tracker v1 completed."
        if reliable
        else (
            "A reliable moving-ball track could not be formed from the "
            "available detections."
        )
    )
    settings = VideoBallTrackingSettings(
        motion_model="constant_velocity_recent_median",
        max_recoverable_gap=MAX_RECOVERABLE_GAP,
        minimum_observed_points=MINIMUM_OBSERVED_POINTS,
        static_radius_normalized=STATIC_RADIUS_NORMALIZED,
        base_gate_normalized=BASE_GATE_NORMALIZED,
        maximum_gate_normalized=MAXIMUM_GATE_NORMALIZED,
        history_points=HISTORY_POINTS,
    )
    document = VideoBallTrackingDocument(
        analysis_id=analysis_id,
        status=status,
        created_at=started_at,
        completed_at=completed_at,
        settings=settings,
        primary_track=primary_track,
        candidate_diagnostics=candidate_diagnostics,
        message=message,
    )
    relative_base = f"/static/video-analysis/{analysis_id}/tracking"
    observed_points = [
        point for point in primary_track if point.source == "observed"
    ]
    recovered_points = [
        point for point in primary_track if point.source == "recovered"
    ]
    predicted_points = [
        point for point in primary_track if point.source == "predicted"
    ]
    observation_gaps = _observation_gaps(primary_tracklet)
    start_frame = primary_track[0].frame_index if primary_track else None
    end_frame = primary_track[-1].frame_index if primary_track else None
    duration_frames = (
        end_frame - start_frame + 1
        if start_frame is not None and end_frame is not None
        else 0
    )
    summary = VideoBallTrackingSummary(
        analysis_id=analysis_id,
        status=status,
        total_video_frames=analysis.frame_count,
        raw_candidate_count=len(candidates),
        candidate_frames=len({candidate.frame_index for candidate in candidates}),
        track_start_frame=start_frame,
        track_end_frame=end_frame,
        track_duration_frames=duration_frames,
        track_duration_seconds=round(
            (end_frame - start_frame) / analysis.fps
            if start_frame is not None and end_frame is not None
            else 0.0,
            6,
        ),
        observed_track_points=len(observed_points),
        predicted_points=len(predicted_points),
        recovered_points=len(recovered_points),
        rejected_candidates=max(0, len(candidates) - len(observed_points)),
        longest_gap_frames=max(observation_gaps, default=0),
        average_observed_confidence=round(
            statistics.fmean(point.confidence for point in observed_points)
            if observed_points
            else 0.0,
            6,
        ),
        track_confidence=track_confidence,
        track_quality=_track_quality(track_confidence),
        approximate_direction=_approximate_direction(primary_track),
        possible_bounce_transition_detected=_possible_bounce_transition(
            primary_tracklet if reliable else None
        ),
        tracking_video_url=f"{relative_base}/{TRACKING_VIDEO_FILENAME}",
        tracking_json_url=f"{relative_base}/{TRACKING_RESULT_FILENAME}",
        tracking_csv_url=f"{relative_base}/{TRACKING_CSV_FILENAME}",
        tracking_summary_url=f"{relative_base}/{TRACKING_SUMMARY_FILENAME}",
        processing_duration_seconds=round(
            time.perf_counter() - started_clock,
            3,
        ),
        message=message,
    )
    _write_json(
        output_dir / TRACKING_RESULT_FILENAME,
        document.model_dump(mode="json"),
    )
    _write_tracking_csv(output_dir / TRACKING_CSV_FILENAME, primary_track)
    _write_json(
        output_dir / TRACKING_SUMMARY_FILENAME,
        summary.model_dump(mode="json"),
    )
    return summary, primary_track


def _load_detection_document(
    analysis_id: str,
    expected_frame_count: int,
) -> VideoBallDetectionsDocument:
    try:
        document = VideoBallDetectionsDocument.model_validate(
            json.loads(_detections_path(analysis_id).read_text(encoding="utf-8"))
        )
    except FileNotFoundError as exc:
        raise VideoBallTrackingError(
            "Every-frame detections are missing. Run Ball Detection first.",
            status_code=409,
        ) from exc
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise VideoBallTrackingError(
            "Every-frame detections are malformed.",
            status_code=400,
        ) from exc
    frame_indexes = [frame.frame_index for frame in document.frames]
    if (
        document.analysis_id != analysis_id
        or len(document.frames) != expected_frame_count
        or frame_indexes != list(range(expected_frame_count))
        or not all(frame.processed for frame in document.frames)
    ):
        raise VideoBallTrackingError(
            "Every-frame detections do not match the prepared analysis.",
            status_code=400,
        )
    return document


def _flatten_candidates(
    document: VideoBallDetectionsDocument,
) -> list[RawTrackingCandidate]:
    candidates: list[RawTrackingCandidate] = []
    for frame in document.frames:
        for candidate in frame.detections:
            candidates.append(_raw_candidate(frame.frame_index, frame.timestamp_seconds, candidate))
    return candidates


def _raw_candidate(
    frame_index: int,
    timestamp_seconds: float,
    candidate: BallCandidate,
) -> RawTrackingCandidate:
    return RawTrackingCandidate(
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        candidate_id=candidate.candidate_id,
        confidence=candidate.confidence,
        x=candidate.center.x,
        y=candidate.center.y,
        normalized_x=candidate.center_normalized.x,
        normalized_y=candidate.center_normalized.y,
        width_pixels=candidate.width_pixels,
        height_pixels=candidate.height_pixels,
        area_pixels=candidate.area_pixels,
        inside_pitch_corridor=candidate.inside_pitch_corridor,
    )


def _assign_static_likelihoods(
    candidates: list[RawTrackingCandidate],
) -> None:
    for candidate in candidates:
        nearby = [
            other
            for other in candidates
            if other.candidate_id != candidate.candidate_id
            and abs(other.frame_index - candidate.frame_index) >= 2
            and _distance(candidate, other) <= STATIC_RADIUS_NORMALIZED
        ]
        if len(nearby) < 2:
            candidate.static_likelihood = 0.0
            continue
        all_frames = [candidate.frame_index, *(item.frame_index for item in nearby)]
        span = max(all_frames) - min(all_frames)
        persistence = min(1.0, len(nearby) / 6)
        span_score = min(1.0, span / 12)
        maximum_displacement = max(
            _distance(candidate, other) for other in nearby
        )
        displacement_score = max(
            0.0,
            1.0 - maximum_displacement / STATIC_RADIUS_NORMALIZED,
        )
        candidate.static_likelihood = round(
            _clamp(
                0.45 * persistence
                + 0.30 * span_score
                + 0.25 * displacement_score
            ),
            6,
        )


def _build_primary_track(
    candidates: list[RawTrackingCandidate],
) -> tuple[
    Tracklet | None,
    dict[str, TrackingCandidateScoreComponents],
]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (candidate.frame_index, candidate.candidate_id),
    )
    tracklets: list[Tracklet] = []
    best_components: dict[str, TrackingCandidateScoreComponents] = {}
    for candidate in ordered:
        best_tracklet = Tracklet(observations=[candidate])
        best_extension_value = -math.inf
        for previous_tracklet in tracklets:
            previous = previous_tracklet.observations[-1]
            frame_delta = candidate.frame_index - previous.frame_index
            if frame_delta <= 0 or frame_delta > MAX_RECOVERABLE_GAP + 1:
                continue
            association = _association_score(previous_tracklet, candidate)
            if association is None or association.total < MINIMUM_LINK_SCORE:
                continue
            extension_value = (
                previous_tracklet.link_score
                + association.total
                + _candidate_base_score(candidate)
            )
            if extension_value <= best_extension_value:
                continue
            best_extension_value = extension_value
            best_tracklet = Tracklet(
                observations=[*previous_tracklet.observations, candidate],
                link_score=previous_tracklet.link_score + association.total,
                score_components={
                    **previous_tracklet.score_components,
                    candidate.candidate_id: association,
                },
                prediction_errors={
                    **previous_tracklet.prediction_errors,
                    candidate.candidate_id: _prediction_error(
                        previous_tracklet,
                        candidate,
                    ),
                },
            )
            best_components[candidate.candidate_id] = association
        tracklets.append(best_tracklet)
    eligible = [
        tracklet
        for tracklet in tracklets
        if len(tracklet.observations) >= MINIMUM_OBSERVED_POINTS
    ]
    if not eligible:
        return None, best_components
    return max(eligible, key=_tracklet_score), best_components


def _association_score(
    tracklet: Tracklet,
    candidate: RawTrackingCandidate,
) -> TrackingCandidateScoreComponents | None:
    previous = tracklet.observations[-1]
    frame_delta = candidate.frame_index - previous.frame_index
    predicted_x, predicted_y, speed = _predict_normalized(
        tracklet.observations,
        candidate.timestamp_seconds,
    )
    prediction_error = math.hypot(
        candidate.normalized_x - predicted_x,
        candidate.normalized_y - predicted_y,
    )
    gate = min(
        MAXIMUM_GATE_NORMALIZED,
        BASE_GATE_NORMALIZED
        + speed
        * max(0.0, candidate.timestamp_seconds - previous.timestamp_seconds)
        * 0.85
        + max(0, frame_delta - 1) * 0.01,
    )
    displacement = _distance(previous, candidate)
    displacement_per_frame = displacement / frame_delta
    if prediction_error > gate or displacement_per_frame > 0.09:
        return None

    detector_confidence = candidate.confidence * 1.2
    prediction_proximity = max(0.0, 1.0 - prediction_error / gate) * 0.9
    if displacement_per_frame < 0.0025:
        motion = -0.35
    elif displacement_per_frame <= 0.06:
        motion = 0.28
    else:
        motion = -0.15

    direction = 0.0
    jump_penalty = 0.0
    previous_velocity = _recent_velocity(tracklet.observations)
    new_velocity = (
        (candidate.normalized_x - previous.normalized_x) / frame_delta,
        (candidate.normalized_y - previous.normalized_y) / frame_delta,
    )
    if previous_velocity is not None:
        cosine = _cosine(previous_velocity, new_velocity)
        direction = 0.35 * cosine
        if cosine < -0.75:
            jump_penalty += 0.45
        previous_speed = math.hypot(*previous_velocity)
        new_speed = math.hypot(*new_velocity)
        if previous_speed > 0.000001 and new_speed > 0.000001:
            speed_ratio = new_speed / previous_speed
            motion += 0.35 * math.exp(-abs(math.log(speed_ratio)))
            if speed_ratio > 6 or speed_ratio < 1 / 6:
                jump_penalty += 0.35

    area_ratio = max(candidate.area_pixels, 0.000001) / max(
        previous.area_pixels,
        0.000001,
    )
    size_consistency = 0.35 * math.exp(-abs(math.log(area_ratio)))
    if area_ratio > 8 or area_ratio < 1 / 8:
        jump_penalty += 0.35
    corridor = (
        0.08
        if candidate.inside_pitch_corridor is True
        else -0.04
        if candidate.inside_pitch_corridor is False
        else 0.0
    )
    static_penalty = candidate.static_likelihood * 1.6
    total = (
        detector_confidence
        + motion
        + prediction_proximity
        + direction
        + size_consistency
        + corridor
        - static_penalty
        - jump_penalty
    )
    return TrackingCandidateScoreComponents(
        detector_confidence=round(detector_confidence, 6),
        motion=round(motion, 6),
        prediction_proximity=round(prediction_proximity, 6),
        direction=round(direction, 6),
        size_consistency=round(size_consistency, 6),
        corridor=round(corridor, 6),
        static_penalty=round(static_penalty, 6),
        jump_penalty=round(jump_penalty, 6),
        total=round(total, 6),
    )


def _candidate_base_score(candidate: RawTrackingCandidate) -> float:
    corridor = 0.08 if candidate.inside_pitch_corridor is True else -0.04
    return candidate.confidence + corridor - candidate.static_likelihood


def _tracklet_score(tracklet: Tracklet) -> float:
    observations = tracklet.observations
    observed_count = len(observations)
    span = observations[-1].frame_index - observations[0].frame_index + 1
    average_confidence = statistics.fmean(
        candidate.confidence for candidate in observations
    )
    average_static = statistics.fmean(
        candidate.static_likelihood for candidate in observations
    )
    movement = _distance(observations[0], observations[-1])
    smoothness = _motion_consistency(observations)
    missing_frames = sum(_observation_gaps(tracklet))
    score = (
        observed_count
        + min(span, 30) * 0.06
        + average_confidence * 2
        + min(movement / 0.08, 2.0) * 1.5
        + smoothness * 2
        + tracklet.link_score * 0.15
        - average_static * 4
        - missing_frames * 0.12
    )
    if observed_count >= 3 and movement / max(1, span - 1) < 0.0025:
        score -= 5
    return score


def _is_reliable_track(tracklet: Tracklet) -> bool:
    observations = tracklet.observations
    if len(observations) < MINIMUM_OBSERVED_POINTS:
        return False
    span = observations[-1].frame_index - observations[0].frame_index + 1
    movement = _distance(observations[0], observations[-1])
    average_static = statistics.fmean(
        candidate.static_likelihood for candidate in observations
    )
    required_movement = max(0.012, 0.0015 * max(1, span - 1))
    return (
        span >= 3
        and movement >= required_movement
        and average_static < 0.8
        and _tracklet_score(tracklet) >= 5
    )


def _build_tracking_points(
    tracklet: Tracklet,
    fps: float,
) -> list[TrackingPoint]:
    observations = tracklet.observations
    points: list[TrackingPoint] = []
    for index, observation in enumerate(observations):
        points.append(
            TrackingPoint(
                frame_index=observation.frame_index,
                timestamp_seconds=observation.timestamp_seconds,
                source="observed",
                candidate_id=observation.candidate_id,
                x=observation.x,
                y=observation.y,
                normalized_x=observation.normalized_x,
                normalized_y=observation.normalized_y,
                confidence=observation.confidence,
                vx=0.0,
                vy=0.0,
                prediction_error=(
                    tracklet.prediction_errors.get(observation.candidate_id)
                    if index
                    else None
                ),
                inside_pitch_corridor=observation.inside_pitch_corridor,
            )
        )
        if index == len(observations) - 1:
            continue
        following = observations[index + 1]
        frame_delta = following.frame_index - observation.frame_index
        if frame_delta <= 1:
            continue
        for missing_offset in range(1, frame_delta):
            fraction = missing_offset / frame_delta
            confidence = min(
                observation.confidence,
                following.confidence,
            ) * 0.55 * (1 - 0.15 * fraction)
            points.append(
                TrackingPoint(
                    frame_index=observation.frame_index + missing_offset,
                    timestamp_seconds=(
                        observation.frame_index + missing_offset
                    ) / fps,
                    source="recovered",
                    candidate_id=None,
                    x=_lerp(observation.x, following.x, fraction),
                    y=_lerp(observation.y, following.y, fraction),
                    normalized_x=_lerp(
                        observation.normalized_x,
                        following.normalized_x,
                        fraction,
                    ),
                    normalized_y=_lerp(
                        observation.normalized_y,
                        following.normalized_y,
                        fraction,
                    ),
                    confidence=_clamp(confidence),
                    vx=0.0,
                    vy=0.0,
                    prediction_error=None,
                    inside_pitch_corridor=None,
                )
            )
    points.sort(key=lambda point: point.frame_index)
    _assign_point_velocities(points, fps)
    return points


def _assign_point_velocities(
    points: list[TrackingPoint],
    fps: float,
) -> None:
    for index, point in enumerate(points):
        if len(points) == 1:
            point.vx = 0.0
            point.vy = 0.0
            continue
        before = points[max(0, index - 1)]
        after = points[min(len(points) - 1, index + 1)]
        elapsed = (after.frame_index - before.frame_index) / fps
        if elapsed <= 0:
            point.vx = 0.0
            point.vy = 0.0
            continue
        point.vx = round((after.x - before.x) / elapsed, 6)
        point.vy = round((after.y - before.y) / elapsed, 6)


def _candidate_diagnostics(
    candidates: list[RawTrackingCandidate],
    primary_tracklet: Tracklet | None,
    best_components: dict[str, TrackingCandidateScoreComponents],
) -> list[TrackingCandidateDiagnostic]:
    selected_ids = {
        candidate.candidate_id
        for candidate in primary_tracklet.observations
    } if primary_tracklet is not None else set()
    primary_components = (
        primary_tracklet.score_components
        if primary_tracklet is not None
        else {}
    )
    diagnostics: list[TrackingCandidateDiagnostic] = []
    for candidate in candidates:
        selected = candidate.candidate_id in selected_ids
        if selected:
            reason = "Selected as a coherent primary-track observation."
        elif candidate.static_likelihood >= 0.6:
            reason = "Rejected due to high stationary-region likelihood."
        else:
            reason = "Rejected from the highest-scoring coherent track."
        diagnostics.append(
            TrackingCandidateDiagnostic(
                frame_index=candidate.frame_index,
                candidate_id=candidate.candidate_id,
                selected=selected,
                selection_reason=reason,
                static_likelihood=candidate.static_likelihood,
                score_components=(
                    primary_components.get(candidate.candidate_id)
                    or best_components.get(candidate.candidate_id)
                ),
            )
        )
    return diagnostics


def _track_confidence(
    tracklet: Tracklet,
    points: list[TrackingPoint],
) -> float:
    observations = tracklet.observations
    observed_count = len(observations)
    observed_ratio = observed_count / max(1, len(points))
    average_confidence = statistics.fmean(
        candidate.confidence for candidate in observations
    )
    average_static = statistics.fmean(
        candidate.static_likelihood for candidate in observations
    )
    span = observations[-1].frame_index - observations[0].frame_index + 1
    longest_gap = max(_observation_gaps(tracklet), default=0)
    confidence = (
        0.22 * min(1.0, observed_count / 8)
        + 0.18 * observed_ratio
        + 0.18 * average_confidence
        + 0.18 * _motion_consistency(observations)
        + 0.12 * min(1.0, span / 12)
        + 0.12 * (1.0 - average_static)
        - 0.03 * longest_gap
    )
    return round(_clamp(confidence), 2)


def _track_quality(confidence: float) -> str:
    if confidence >= 0.8:
        return "strong"
    if confidence >= 0.6:
        return "good"
    if confidence >= 0.4:
        return "medium"
    return "low"


def _approximate_direction(points: list[TrackingPoint]) -> str:
    if len(points) < 2:
        return "unavailable"
    delta_x = points[-1].normalized_x - points[0].normalized_x
    delta_y = points[-1].normalized_y - points[0].normalized_y
    horizontal = "right" if delta_x > 0.015 else "left" if delta_x < -0.015 else ""
    vertical = "down" if delta_y > 0.015 else "up" if delta_y < -0.015 else ""
    return "-".join(part for part in (horizontal, vertical) if part) or "minimal"


def _possible_bounce_transition(
    tracklet: Tracklet | None,
) -> bool | str:
    if tracklet is None or len(tracklet.observations) < 4:
        return "uncertain"
    velocities = _segment_velocities(tracklet.observations)
    if len(velocities) < 3:
        return "uncertain"
    angles = [
        math.degrees(math.acos(max(-1.0, min(1.0, _cosine(first, second)))))
        for first, second in zip(velocities, velocities[1:])
        if math.hypot(*first) > 0.000001 and math.hypot(*second) > 0.000001
    ]
    return any(55 <= angle <= 145 for angle in angles)


def _render_tracking_video(
    *,
    analysis_id: str,
    stored_filename: str,
    total_frames: int,
    fps: float,
    width: int,
    height: int,
    primary_track: list[TrackingPoint],
    candidates: list[RawTrackingCandidate],
    track_confidence: float,
    job_id: str,
) -> Path:
    source_path = (
        VIDEO_ANALYSIS_ROOT / analysis_id / "raw" / stored_filename
    )
    output_dir = _tracking_output_dir(analysis_id)
    intermediate_path = output_dir / "tracking_debug_intermediate.avi"
    encoded_path = output_dir / "tracking_debug_encoded.mp4"
    final_path = output_dir / TRACKING_VIDEO_FILENAME
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        capture.release()
        raise VideoBallTrackingError(
            "OpenCV could not open the original analysis video."
        )
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (
        not math.isfinite(source_fps)
        or source_fps <= 0
        or abs(source_fps - fps) > 0.01
        or source_frames != total_frames
        or source_width != width
        or source_height != height
    ):
        capture.release()
        raise VideoBallTrackingError(
            "Original video metadata no longer matches the prepared analysis."
        )
    writer = cv2.VideoWriter(
        str(intermediate_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise VideoBallTrackingError(
            "Could not create the tracking debug video."
        )
    point_by_frame = {
        point.frame_index: point for point in primary_track
    }
    candidate_by_frame: dict[int, list[RawTrackingCandidate]] = {}
    for candidate in candidates:
        candidate_by_frame.setdefault(candidate.frame_index, []).append(candidate)
    selected_ids = {
        point.candidate_id
        for point in primary_track
        if point.candidate_id is not None
    }
    start_frame = primary_track[0].frame_index if primary_track else None
    end_frame = primary_track[-1].frame_index if primary_track else None
    history: list[TrackingPoint] = []
    processed_frames = 0
    try:
        for frame_index in range(total_frames):
            ok, frame = capture.read()
            if not ok:
                raise VideoBallTrackingError(
                    f"Video decoding stopped at frame {frame_index} "
                    f"of {total_frames}."
                )
            active = (
                start_frame is not None
                and end_frame is not None
                and start_frame <= frame_index <= end_frame
            )
            if active:
                for candidate in candidate_by_frame.get(frame_index, []):
                    if candidate.candidate_id not in selected_ids:
                        cv2.circle(
                            frame,
                            (round(candidate.x), round(candidate.y)),
                            3,
                            (145, 145, 145),
                            1,
                            cv2.LINE_AA,
                        )
                point = point_by_frame.get(frame_index)
                if point is not None:
                    history.append(point)
                    history = history[-HISTORY_POINTS:]
                    _draw_short_history(frame, history)
                    _draw_tracking_point(frame, point)
            current_point = point_by_frame.get(frame_index)
            _draw_tracking_panel(
                frame,
                frame_index,
                total_frames,
                current_point,
                active,
                track_confidence,
            )
            writer.write(frame)
            processed_frames += 1
            progress = 60 + int(processed_frames / total_frames * 30)
            video_ball_tracking_job_store.update(
                job_id,
                status="rendering_video",
                progress=progress,
                message=(
                    f"Rendering tracking frame {processed_frames} "
                    f"of {total_frames}."
                ),
            )
    finally:
        capture.release()
        writer.release()
    if processed_frames != total_frames:
        raise VideoBallTrackingError(
            "Tracking debug video does not contain every original frame."
        )
    try:
        transcode_browser_mp4(
            intermediate_path,
            encoded_path,
            timeout_seconds=600,
        )
        encoded_path.replace(final_path)
    except Exception as exc:
        encoded_path.unlink(missing_ok=True)
        raise VideoBallTrackingError(
            "Could not encode a browser-compatible tracking debug video."
        ) from exc
    finally:
        intermediate_path.unlink(missing_ok=True)
    return final_path


def _draw_short_history(
    frame,
    history: list[TrackingPoint],
) -> None:
    for index, (first, second) in enumerate(zip(history, history[1:]), start=1):
        strength = index / max(1, len(history) - 1)
        color = (
            20,
            round(110 + 120 * strength),
            round(20 + 50 * strength),
        )
        cv2.line(
            frame,
            (round(first.x), round(first.y)),
            (round(second.x), round(second.y)),
            color,
            1,
            cv2.LINE_AA,
        )


def _draw_tracking_point(frame, point: TrackingPoint) -> None:
    colors = {
        "observed": (80, 230, 80),
        "predicted": (0, 230, 255),
        "recovered": (0, 150, 255),
    }
    labels = {
        "observed": "Observed",
        "predicted": "Predicted",
        "recovered": "Recovered",
    }
    color = colors[point.source]
    center = (round(point.x), round(point.y))
    cv2.circle(frame, center, 8, color, 2, cv2.LINE_AA)
    cv2.circle(frame, center, 2, color, -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"{labels[point.source]} {point.confidence:.2f}",
        (center[0] + 10, max(18, center[1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_tracking_panel(
    frame,
    frame_index: int,
    total_frames: int,
    point: TrackingPoint | None,
    active: bool,
    track_confidence: float,
) -> None:
    panel_width = min(frame.shape[1] - 10, 350)
    cv2.rectangle(frame, (10, 10), (panel_width, 82), (18, 18, 18), -1)
    cv2.putText(
        frame,
        "Moving Ball Tracker v1 - debug",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    state = point.source.title() if point is not None else "Searching" if active else "Inactive"
    cv2.putText(
        frame,
        f"Frame {frame_index + 1}/{total_frames} | {state}",
        (20, 53),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (80, 230, 80) if point is not None else (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Track confidence {track_confidence:.2f}",
        (20, 73),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _predict_normalized(
    observations: list[RawTrackingCandidate],
    target_timestamp: float,
) -> tuple[float, float, float]:
    last = observations[-1]
    velocity = _recent_velocity(observations)
    if velocity is None:
        return last.normalized_x, last.normalized_y, 0.0
    elapsed_seconds = max(
        0.0,
        target_timestamp - last.timestamp_seconds,
    )
    return (
        last.normalized_x + velocity[0] * elapsed_seconds,
        last.normalized_y + velocity[1] * elapsed_seconds,
        math.hypot(*velocity),
    )


def _prediction_error(
    tracklet: Tracklet,
    candidate: RawTrackingCandidate,
) -> float:
    predicted_x, predicted_y, _ = _predict_normalized(
        tracklet.observations,
        candidate.timestamp_seconds,
    )
    return round(
        math.hypot(
            candidate.normalized_x - predicted_x,
            candidate.normalized_y - predicted_y,
        ),
        6,
    )


def _recent_velocity(
    observations: list[RawTrackingCandidate],
) -> tuple[float, float] | None:
    velocities = _segment_velocities(observations[-4:])
    if not velocities:
        return None
    return (
        statistics.median(velocity[0] for velocity in velocities),
        statistics.median(velocity[1] for velocity in velocities),
    )


def _segment_velocities(
    observations: list[RawTrackingCandidate],
) -> list[tuple[float, float]]:
    velocities: list[tuple[float, float]] = []
    for first, second in zip(observations, observations[1:]):
        elapsed_seconds = (
            second.timestamp_seconds - first.timestamp_seconds
        )
        if elapsed_seconds <= 0:
            continue
        velocities.append(
            (
                (second.normalized_x - first.normalized_x) / elapsed_seconds,
                (second.normalized_y - first.normalized_y) / elapsed_seconds,
            )
        )
    return velocities


def _motion_consistency(
    observations: list[RawTrackingCandidate],
) -> float:
    velocities = _segment_velocities(observations)
    if len(velocities) < 2:
        return 0.5
    direction_scores = [
        (_cosine(first, second) + 1) / 2
        for first, second in zip(velocities, velocities[1:])
    ]
    speeds = [math.hypot(*velocity) for velocity in velocities]
    positive_speeds = [speed for speed in speeds if speed > 0.000001]
    if len(positive_speeds) < 2:
        speed_consistency = 0.0
    else:
        median_speed = statistics.median(positive_speeds)
        speed_consistency = statistics.fmean(
            math.exp(-abs(math.log(speed / median_speed)))
            for speed in positive_speeds
        )
    return _clamp(
        0.65 * statistics.fmean(direction_scores)
        + 0.35 * speed_consistency
    )


def _observation_gaps(tracklet: Tracklet | None) -> list[int]:
    if tracklet is None:
        return []
    return [
        max(0, second.frame_index - first.frame_index - 1)
        for first, second in zip(
            tracklet.observations,
            tracklet.observations[1:],
        )
    ]


def _distance(
    first: RawTrackingCandidate,
    second: RawTrackingCandidate,
) -> float:
    return math.hypot(
        second.normalized_x - first.normalized_x,
        second.normalized_y - first.normalized_y,
    )


def _cosine(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    denominator = math.hypot(*first) * math.hypot(*second)
    if denominator <= 0.0000001:
        return 0.0
    return (first[0] * second[0] + first[1] * second[1]) / denominator


def _verify_output_video(path: Path) -> tuple[int, float]:
    if not path.is_file() or path.stat().st_size == 0:
        raise VideoBallTrackingError(
            "Tracking debug video was not created."
        )
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoBallTrackingError(
            "Tracking debug video could not be verified."
        )
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise VideoBallTrackingError(
            "Tracking debug video metadata is invalid."
        )
    return frame_count, fps


def _write_tracking_csv(path: Path, points: list[TrackingPoint]) -> None:
    columns = [
        "analysis_id",
        "frame_index",
        "timestamp_seconds",
        "source",
        "candidate_id",
        "x",
        "y",
        "normalized_x",
        "normalized_y",
        "confidence",
        "vx",
        "vy",
        "prediction_error",
        "inside_pitch_corridor",
    ]
    analysis_id = path.parents[1].name
    rows = [
        {
            "analysis_id": analysis_id,
            "frame_index": point.frame_index,
            "timestamp_seconds": point.timestamp_seconds,
            "source": point.source,
            "candidate_id": point.candidate_id or "",
            "x": point.x,
            "y": point.y,
            "normalized_x": point.normalized_x,
            "normalized_y": point.normalized_y,
            "confidence": point.confidence,
            "vx": point.vx,
            "vy": point.vy,
            "prediction_error": (
                ""
                if point.prediction_error is None
                else point.prediction_error
            ),
            "inside_pitch_corridor": (
                ""
                if point.inside_pitch_corridor is None
                else point.inside_pitch_corridor
            ),
        }
        for point in points
    ]
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(path)
    except (OSError, csv.Error) as exc:
        temporary_path.unlink(missing_ok=True)
        raise VideoBallTrackingError(
            "tracking_points.csv could not be saved."
        ) from exc


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(path)
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        raise VideoBallTrackingError(
            f"{path.name} could not be saved."
        ) from exc


def _update_analysis_metadata(
    analysis_id: str,
    **updates: Any,
) -> None:
    metadata_path = (
        VIDEO_ANALYSIS_ROOT
        / analysis_id
        / "reports"
        / "analysis_metadata.json"
    )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(updates)
        temporary_path = metadata_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(metadata_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise VideoBallTrackingError(
            "Analysis metadata could not be updated."
        ) from exc


def _update_job(
    job_id: str,
    status: str,
    progress: int,
    message: str,
) -> None:
    video_ball_tracking_job_store.update(
        job_id,
        status=status,
        progress=progress,
        message=message,
    )


def _mark_job_failed(
    analysis_id: str,
    job_id: str,
    message: str,
) -> None:
    output_dir = _tracking_output_dir(analysis_id)
    for temporary_path in output_dir.glob("*.tmp"):
        temporary_path.unlink(missing_ok=True)
    for temporary_path in (
        output_dir / "tracking_debug_intermediate.avi",
        output_dir / "tracking_debug_encoded.mp4",
    ):
        temporary_path.unlink(missing_ok=True)
    video_ball_tracking_job_store.update(
        job_id,
        success=False,
        status="failed",
        error_message=message,
        message=message,
    )
    try:
        _update_analysis_metadata(
            analysis_id,
            tracking_status="tracking_failed",
            tracking_completed_at=_iso(utc_now()),
            updated_at=_iso(utc_now()),
        )
    except VideoBallTrackingError:
        pass


def _clear_previous_tracking_outputs(output_dir: Path) -> None:
    for filename in (
        TRACKING_RESULT_FILENAME,
        TRACKING_CSV_FILENAME,
        TRACKING_SUMMARY_FILENAME,
        TRACKING_VIDEO_FILENAME,
        "tracking_debug_intermediate.avi",
        "tracking_debug_encoded.mp4",
    ):
        (output_dir / filename).unlink(missing_ok=True)


def _detections_path(analysis_id: str) -> Path:
    return (
        VIDEO_ANALYSIS_ROOT
        / analysis_id
        / "detections"
        / "detections.json"
    )


def _tracking_output_dir(analysis_id: str) -> Path:
    return VIDEO_ANALYSIS_ROOT / analysis_id / "tracking"


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _lerp(first: float, second: float, amount: float) -> float:
    return first + (second - first) * amount


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
