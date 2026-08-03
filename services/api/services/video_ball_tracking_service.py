"""Complete Delivery Tracking v2 for persisted Video Analysis detections.

Extends Moving Ball Tracker v1: beam hypotheses, bidirectional refinement,
short-gap recovery with provenance, primary bounce, polished 2D replay.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Literal

import cv2
import numpy as np

from ..schemas.delivery_physics import DeliveryPhysicsResult
from ..schemas.video_analysis import (
    BallCandidate,
    PrimaryBounceResult,
    TrackingCandidateDiagnostic,
    TrackingCandidateScoreComponents,
    TrackingPoint,
    TrackingProvenance,
    VideoBallDetectionsDocument,
    VideoBallTrackingDocument,
    VideoBallTrackingResultLinks,
    VideoBallTrackingResultResponse,
    VideoBallTrackingSettings,
    VideoBallTrackingSummary,
)
from .ball_detection_clip import transcode_browser_mp4
from .delivery_physics_service import (
    analyse_delivery_physics,
    failed_physics_result,
)
from .video_analysis_service import (
    VIDEO_ANALYSIS_ROOT,
    VideoAnalysisServiceError,
    load_video_analysis,
)
from .video_ball_detection_job_store import utc_now
from .video_ball_tracking_job_store import video_ball_tracking_job_store


MAX_RECOVERABLE_GAP = 6
PREFERRED_GAP = 4
MINIMUM_OBSERVED_POINTS = 3
STATIC_RADIUS_NORMALIZED = 0.012
BASE_GATE_NORMALIZED = 0.025
MAXIMUM_GATE_NORMALIZED = 0.16
HISTORY_POINTS = 8
MINIMUM_LINK_SCORE = 0.15
BEAM_WIDTH = 4
TRAIL_DURATION_SECONDS = 0.18
TRACKING_RESULT_FILENAME = "tracking_result.json"
TRACKING_CSV_FILENAME = "tracking_points.csv"
TRACKING_SUMMARY_FILENAME = "tracking_summary.json"
TRACKING_VIDEO_FILENAME = "tracking_debug.mp4"
DELIVERY_REPLAY_FILENAME = "delivery_replay.mp4"
PHYSICS_RESULT_FILENAME = "physics_result.json"

_PROVENANCE_TO_SOURCE: dict[TrackingProvenance, Literal["observed", "predicted", "recovered"]] = {
    "OBSERVED": "observed",
    "TRACKER_RECOVERED": "recovered",
    "PHYSICS_RECONSTRUCTED": "recovered",
    "PROJECTED": "predicted",
}


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
    bounding_box: list[float]
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


def run_video_ball_tracking_job(
    analysis_id: str,
    job_id: str,
    *,
    include_delivery_analysis: bool = True,
) -> None:
    try:
        summary, primary_track = _process_video_ball_tracking(
            analysis_id,
            job_id,
            include_delivery_analysis=include_delivery_analysis,
        )
        links = VideoBallTrackingResultLinks(
            tracking_video_url=summary.tracking_video_url,
            tracking_json_url=summary.tracking_json_url,
            tracking_csv_url=summary.tracking_csv_url,
            tracking_summary_url=summary.tracking_summary_url,
            delivery_replay_url=summary.delivery_replay_url,
            physics_result_url=summary.physics_result_url,
        )
        is_ready = summary.status == "ready"
        video_ball_tracking_job_store.update(
            job_id,
            success=is_ready,
            status=summary.status,
            progress=100,
            error_message=None if is_ready else summary.message,
            failure_code=(
                None
                if is_ready
                else (
                    "NO_MOVING_BALL_CANDIDATES"
                    if summary.raw_candidate_count == 0
                    else (
                        "TRACK_TOO_SHORT"
                        if summary.candidate_frames < MINIMUM_OBSERVED_POINTS
                        else "TRACK_UNAVAILABLE"
                    )
                )
            ),
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
        raw_primary_track=document.raw_primary_track,
        candidate_diagnostics=document.candidate_diagnostics,
        bounce=document.bounce,
        physics=document.physics,
        message=summary.message,
    )


def _process_video_ball_tracking(
    analysis_id: str,
    job_id: str,
    *,
    include_delivery_analysis: bool = True,
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
        28,
        "Building primary delivery hypotheses...",
    )
    primary_tracklet, best_components, raw_tracklet = _build_primary_track(
        candidates
    )

    _update_job(
        job_id,
        "recovering_gaps",
        45,
        "Refining track, recovering gaps, rejecting outliers...",
    )
    if primary_tracklet is not None:
        primary_tracklet = _refine_primary_track(primary_tracklet, candidates)
        primary_tracklet = _reject_outliers(primary_tracklet)
        primary_tracklet = _trim_outgoing_shot(primary_tracklet)

    reliable = (
        primary_tracklet is not None
        and _is_reliable_track(primary_tracklet)
    )
    raw_primary_track = (
        _build_tracking_points(raw_tracklet, analysis.fps)
        if raw_tracklet is not None and reliable
        else []
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
    consistency = (
        _motion_consistency(primary_tracklet.observations)
        if reliable and primary_tracklet is not None
        else 0.0
    )

    bounce: PrimaryBounceResult | None = None
    physics: DeliveryPhysicsResult | None = None
    physics_relative_url: str | None = None
    physics_replay_track: list[TrackingPoint] = []
    physics_bounce: PrimaryBounceResult | None = None
    if include_delivery_analysis:
        bounce = _detect_primary_bounce(
            primary_track if reliable else [],
            fps=analysis.fps,
            width=analysis.width,
            height=analysis.height,
        )
        if (
            reliable
            and bounce.bounce_detected is True
            and bounce.bounce_frame is not None
        ):
            primary_track = _smooth_track_around_bounce(
                primary_track,
                bounce.bounce_frame,
            )

        _update_job(
            job_id,
            "fitting_physics",
            52,
            "Fitting calibrated delivery physics...",
        )
        try:
            physics = analyse_delivery_physics(
                analysis_id=analysis_id,
                primary_track=primary_track,
                detections=detection_document,
                tracker_bounce=bounce if reliable else None,
                fps=analysis.fps,
                width=analysis.width,
                height=analysis.height,
                total_frames=analysis.frame_count,
            )
        except Exception as exc:
            physics = failed_physics_result(
                analysis_id,
                analysis.width,
                analysis.height,
                (
                    "Physics Engine V1 failed without affecting the raw track: "
                    f"{type(exc).__name__}."
                ),
            )
        physics_relative_url = (
            f"/static/video-analysis/{analysis_id}/tracking/"
            f"{PHYSICS_RESULT_FILENAME}"
        )
        physics = physics.model_copy(
            update={"physics_result_url": physics_relative_url}
        )
        physics_replay_track = _physics_replay_points(
            physics,
            analysis.width,
            analysis.height,
        )
        physics_bounce = _physics_replay_bounce(
            physics,
            analysis.width,
            analysis.height,
        )

    _update_job(
        job_id,
        "rendering_video",
        58,
        "Generating tracking debug and delivery replay...",
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
        bounce=bounce if reliable else None,
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

    delivery_replay_url: str | None = None
    if include_delivery_analysis and reliable and primary_track:
        replay_path = _render_delivery_replay(
            analysis_id=analysis_id,
            stored_filename=analysis.stored_filename,
            total_frames=analysis.frame_count,
            fps=analysis.fps,
            width=analysis.width,
            height=analysis.height,
            primary_track=physics_replay_track or primary_track,
            bounce=physics_bounce or bounce,
            job_id=job_id,
        )
        replay_frames, replay_fps = _verify_output_video(replay_path)
        if (
            replay_frames != analysis.frame_count
            or abs(replay_fps - analysis.fps) > 0.01
        ):
            replay_path.unlink(missing_ok=True)
            raise VideoBallTrackingError(
                "Delivery replay metadata does not match the original video."
            )
        delivery_replay_url = (
            f"/static/video-analysis/{analysis_id}/tracking/"
            f"{DELIVERY_REPLAY_FILENAME}"
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
        "Complete Delivery Tracking v2 completed."
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
        beam_width=BEAM_WIDTH,
        tracker_version="delivery_track_v2",
    )
    document = VideoBallTrackingDocument(
        analysis_id=analysis_id,
        status=status,
        created_at=started_at,
        completed_at=completed_at,
        settings=settings,
        primary_track=primary_track,
        raw_primary_track=raw_primary_track,
        candidate_diagnostics=candidate_diagnostics,
        bounce=bounce if reliable else None,
        physics=physics,
        message=message,
    )
    relative_base = f"/static/video-analysis/{analysis_id}/tracking"
    observed_points = [
        point for point in primary_track if point.provenance == "OBSERVED"
    ]
    recovered_points = [
        point
        for point in primary_track
        if point.provenance == "TRACKER_RECOVERED"
    ]
    count_source = physics_replay_track if include_delivery_analysis else primary_track
    physics_points = [
        point
        for point in count_source
        if point.provenance == "PHYSICS_RECONSTRUCTED"
    ]
    projected_points = [
        point
        for point in count_source
        if point.provenance == "PROJECTED"
    ]
    observation_gaps = _observation_gaps(primary_tracklet)
    start_frame = primary_track[0].frame_index if primary_track else None
    end_frame = primary_track[-1].frame_index if primary_track else None
    duration_frames = (
        end_frame - start_frame + 1
        if start_frame is not None and end_frame is not None
        else 0
    )
    observation_ratio = (
        len(observed_points) / max(1, len(primary_track))
        if primary_track
        else 0.0
    )
    summary = VideoBallTrackingSummary(
        analysis_id=analysis_id,
        status=status,
        total_video_frames=analysis.frame_count,
        raw_candidate_count=len(candidates),
        candidate_frames=len({candidate.frame_index for candidate in candidates}),
        track_start_frame=start_frame,
        track_end_frame=end_frame,
        first_supported_delivery_point=start_frame,
        track_start_label="track_start" if start_frame is not None else "unavailable",
        track_duration_frames=duration_frames,
        track_duration_seconds=round(
            (end_frame - start_frame) / analysis.fps
            if start_frame is not None and end_frame is not None
            else 0.0,
            6,
        ),
        observed_track_points=len(observed_points),
        predicted_points=len(projected_points),
        recovered_points=len(recovered_points),
        physics_reconstructed_points=len(physics_points),
        projected_points=len(projected_points),
        rejected_candidates=max(0, len(candidates) - len(observed_points)),
        longest_gap_frames=max(observation_gaps, default=0),
        observation_ratio=round(observation_ratio, 6),
        average_observed_confidence=round(
            statistics.fmean(point.confidence for point in observed_points)
            if observed_points
            else 0.0,
            6,
        ),
        consistency_score=round(_clamp(consistency), 6),
        track_confidence=track_confidence,
        track_quality=_track_quality(track_confidence, reliable),
        approximate_direction=_approximate_direction(primary_track),
        possible_bounce_transition_detected=(
            bounce.bounce_detected if bounce is not None else "uncertain"
        ),
        bounce_detected=(
            bounce.bounce_detected if bounce is not None else "uncertain"
        ),
        bounce_frame=bounce.bounce_frame if bounce is not None else None,
        bounce_confidence=bounce.confidence if bounce is not None else 0.0,
        tracking_video_url=f"{relative_base}/{TRACKING_VIDEO_FILENAME}",
        delivery_replay_url=delivery_replay_url,
        physics_result_url=physics_relative_url,
        physics_engine_version="v1" if physics is not None else None,
        physics_status=physics.status if physics is not None else None,
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
    if physics is not None:
        _write_json(
            output_dir / PHYSICS_RESULT_FILENAME,
            physics.model_dump(mode="json"),
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
        bounding_box=list(candidate.bbox_xyxy),
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
    Tracklet | None,
]:
    """Forward beam search (K≈4) over association scores — not raw IoU."""
    by_frame: dict[int, list[RawTrackingCandidate]] = {}
    for candidate in candidates:
        by_frame.setdefault(candidate.frame_index, []).append(candidate)

    active: list[Tracklet] = []
    finished: list[Tracklet] = []
    best_components: dict[str, TrackingCandidateScoreComponents] = {}

    for frame_index in sorted(by_frame):
        frame_candidates = by_frame[frame_index]
        extensions: list[tuple[float, Tracklet]] = []
        surviving: list[Tracklet] = []

        for previous_tracklet in active:
            previous = previous_tracklet.observations[-1]
            gap = frame_index - previous.frame_index
            if gap <= 0:
                surviving.append(previous_tracklet)
                continue
            if gap > MAX_RECOVERABLE_GAP + 1:
                finished.append(previous_tracklet)
                continue
            surviving.append(previous_tracklet)
            for candidate in frame_candidates:
                association = _association_score(previous_tracklet, candidate)
                if association is None or association.total < MINIMUM_LINK_SCORE:
                    continue
                extension_value = (
                    previous_tracklet.link_score
                    + association.total
                    + _candidate_base_score(candidate)
                )
                extended = Tracklet(
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
                extensions.append((extension_value, extended))
                best_components[candidate.candidate_id] = association

        for candidate in frame_candidates:
            extensions.append(
                (
                    _candidate_base_score(candidate),
                    Tracklet(observations=[candidate]),
                )
            )

        extensions.sort(key=lambda item: item[0], reverse=True)
        kept_new = [item[1] for item in extensions[: max(BEAM_WIDTH, len(frame_candidates))]]
        active = _prune_tracklets([*surviving, *kept_new], BEAM_WIDTH * 3)

    finished.extend(active)
    eligible = [
        tracklet
        for tracklet in finished
        if len(tracklet.observations) >= MINIMUM_OBSERVED_POINTS
    ]
    if not eligible:
        return None, best_components, None
    primary = max(eligible, key=_tracklet_score)
    raw = Tracklet(
        observations=list(primary.observations),
        link_score=primary.link_score,
        score_components=dict(primary.score_components),
        prediction_errors=dict(primary.prediction_errors),
    )
    return primary, best_components, raw


def _prune_tracklets(tracklets: list[Tracklet], limit: int) -> list[Tracklet]:
    if len(tracklets) <= limit:
        return tracklets
    # Prefer longer coherent hypotheses; keep light diversity of endings.
    ranked = sorted(
        tracklets,
        key=lambda tracklet: (
            _tracklet_score(tracklet),
            len(tracklet.observations),
        ),
        reverse=True,
    )
    kept: list[Tracklet] = []
    seen_ends: set[str] = set()
    for tracklet in ranked:
        end_id = tracklet.observations[-1].candidate_id
        if end_id in seen_ends and len(kept) >= max(BEAM_WIDTH, limit // 2):
            continue
        kept.append(tracklet)
        seen_ends.add(end_id)
        if len(kept) >= limit:
            break
    return kept


def _refine_primary_track(
    tracklet: Tracklet,
    candidates: list[RawTrackingCandidate],
) -> Tracklet:
    """Backward-aware reassignment using past and future neighbours."""
    observations = list(tracklet.observations)
    if len(observations) < 2:
        return tracklet
    by_frame: dict[int, list[RawTrackingCandidate]] = {}
    for candidate in candidates:
        by_frame.setdefault(candidate.frame_index, []).append(candidate)

    # Forward then backward sweeps so future evidence can fix early mistakes.
    for _ in range(2):
        for index in range(len(observations)):
            observations[index] = _best_bidirectional_candidate(
                observations,
                index,
                by_frame.get(observations[index].frame_index, []),
            )
        for index in range(len(observations) - 1, -1, -1):
            observations[index] = _best_bidirectional_candidate(
                observations,
                index,
                by_frame.get(observations[index].frame_index, []),
            )

    # Deduplicate accidental same-frame picks after swaps.
    deduped: list[RawTrackingCandidate] = []
    for observation in observations:
        if deduped and deduped[-1].frame_index == observation.frame_index:
            if observation.confidence >= deduped[-1].confidence:
                deduped[-1] = observation
            continue
        deduped.append(observation)
    return Tracklet(
        observations=deduped,
        link_score=tracklet.link_score,
        score_components=tracklet.score_components,
        prediction_errors=tracklet.prediction_errors,
    )


def _best_bidirectional_candidate(
    observations: list[RawTrackingCandidate],
    index: int,
    alternatives: list[RawTrackingCandidate],
) -> RawTrackingCandidate:
    current = observations[index]
    if len(alternatives) <= 1:
        return current
    previous = observations[index - 1] if index > 0 else None
    following = (
        observations[index + 1] if index + 1 < len(observations) else None
    )
    best = current
    best_score = _bidirectional_score(previous, current, following)
    for alternative in alternatives:
        if alternative.static_likelihood >= 0.75 and alternative.candidate_id != current.candidate_id:
            continue
        score = _bidirectional_score(previous, alternative, following)
        if score > best_score:
            best = alternative
            best_score = score
    return best


def _bidirectional_score(
    previous: RawTrackingCandidate | None,
    candidate: RawTrackingCandidate,
    following: RawTrackingCandidate | None,
) -> float:
    score = candidate.confidence * 1.1 - candidate.static_likelihood * 1.4
    if candidate.inside_pitch_corridor is True:
        score += 0.06
    elif candidate.inside_pitch_corridor is False:
        score -= 0.03
    if previous is not None:
        gap = max(1, candidate.frame_index - previous.frame_index)
        step = _distance(previous, candidate) / gap
        score += 0.35 * math.exp(-step / 0.03)
        if step > 0.08:
            score -= 0.5
    if following is not None:
        gap = max(1, following.frame_index - candidate.frame_index)
        step = _distance(candidate, following) / gap
        score += 0.35 * math.exp(-step / 0.03)
        if step > 0.08:
            score -= 0.5
    if previous is not None and following is not None:
        v1 = (
            (candidate.normalized_x - previous.normalized_x)
            / max(1, candidate.frame_index - previous.frame_index),
            (candidate.normalized_y - previous.normalized_y)
            / max(1, candidate.frame_index - previous.frame_index),
        )
        v2 = (
            (following.normalized_x - candidate.normalized_x)
            / max(1, following.frame_index - candidate.frame_index),
            (following.normalized_y - candidate.normalized_y)
            / max(1, following.frame_index - candidate.frame_index),
        )
        score += 0.25 * _cosine(v1, v2)
    return score


def _reject_outliers(tracklet: Tracklet) -> Tracklet:
    observations = list(tracklet.observations)
    if len(observations) < 4:
        return tracklet
    steps = [
        _distance(observations[index], observations[index + 1])
        / max(1, observations[index + 1].frame_index - observations[index].frame_index)
        for index in range(len(observations) - 1)
    ]
    median_step = statistics.median(steps) if steps else 0.0
    # Perspective-aware: gate scales with local median motion, not a fixed px rule.
    jump_limit = max(0.035, median_step * 4.5, BASE_GATE_NORMALIZED * 2.5)
    keep = [True] * len(observations)
    for index in range(1, len(observations) - 1):
        before = observations[index - 1]
        current = observations[index]
        after = observations[index + 1]
        step_in = _distance(before, current) / max(
            1, current.frame_index - before.frame_index
        )
        step_out = _distance(current, after) / max(
            1, after.frame_index - current.frame_index
        )
        v_in = (
            (current.normalized_x - before.normalized_x),
            (current.normalized_y - before.normalized_y),
        )
        v_out = (
            (after.normalized_x - current.normalized_x),
            (after.normalized_y - current.normalized_y),
        )
        reverse = _cosine(v_in, v_out) < -0.7
        teleport = step_in > jump_limit and step_out > jump_limit
        static_lock = (
            current.static_likelihood >= 0.7
            and step_in < 0.002
            and step_out < 0.002
        )
        if teleport or static_lock or (reverse and step_in > jump_limit * 0.6):
            keep[index] = False
    filtered = [
        observation
        for observation, retained in zip(observations, keep)
        if retained
    ]
    if len(filtered) < MINIMUM_OBSERVED_POINTS:
        return tracklet
    return Tracklet(
        observations=filtered,
        link_score=tracklet.link_score,
        score_components=tracklet.score_components,
        prediction_errors=tracklet.prediction_errors,
    )


def _trim_outgoing_shot(tracklet: Tracklet) -> Tracklet:
    """End near striker / meaningful delivery; do not merge obvious outgoing shot."""
    observations = list(tracklet.observations)
    if len(observations) < 8:
        return tracklet
    velocities = _segment_velocities(observations)
    if len(velocities) < 5:
        return tracklet
    mid = max(2, len(velocities) // 2)
    early = velocities[:mid]
    mean_early = (
        statistics.fmean(item[0] for item in early),
        statistics.fmean(item[1] for item in early),
    )
    # Only inspect the final quarter — mid-track bounce must not trigger a cut.
    start_check = max(mid, int(len(velocities) * 0.75))
    cut_at = len(observations)
    for index in range(start_check, len(velocities)):
        if _cosine(mean_early, velocities[index]) >= -0.55:
            continue
        # Outgoing shot shrinks distance to track start; bounce usually still advances.
        obs_index = index + 1
        if obs_index >= len(observations):
            break
        dist_now = _distance(observations[0], observations[obs_index])
        dist_prev = _distance(observations[0], observations[obs_index - 1])
        if dist_now < dist_prev * 0.92:
            cut_at = obs_index
            break
    if cut_at < MINIMUM_OBSERVED_POINTS or cut_at >= len(observations):
        return tracklet
    return Tracklet(
        observations=observations[:cut_at],
        link_score=tracklet.link_score,
        score_components=tracklet.score_components,
        prediction_errors=tracklet.prediction_errors,
    )


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
    # Bootstrap: without a velocity estimate the constant-velocity gate is too tight.
    if len(tracklet.observations) < 3 or speed < 1e-8:
        gate = min(MAXIMUM_GATE_NORMALIZED, max(gate, BASE_GATE_NORMALIZED * 2.4))
    displacement = _distance(previous, candidate)
    displacement_per_frame = displacement / frame_delta
    previous_velocity = _recent_velocity(tracklet.observations)
    new_velocity = (
        (candidate.normalized_x - previous.normalized_x) / frame_delta,
        (candidate.normalized_y - previous.normalized_y) / frame_delta,
    )
    # Bounce-like reverse: allow a wider prediction gate so the track can continue.
    bounce_like = False
    if previous_velocity is not None:
        bounce_like = (
            _cosine(previous_velocity, new_velocity) < -0.35
            and 0.002 < displacement_per_frame < 0.075
        )
        if bounce_like:
            gate = min(MAXIMUM_GATE_NORMALIZED, gate * 2.0)
    if displacement_per_frame > 0.09:
        return None
    if prediction_error > gate and not bounce_like:
        return None
    if bounce_like and prediction_error > gate * 1.25:
        return None

    detector_confidence = candidate.confidence * 1.2
    prediction_proximity = max(0.0, 1.0 - prediction_error / max(gate, 1e-6)) * 0.9
    if displacement_per_frame < 0.0025:
        motion = -0.35
    elif displacement_per_frame <= 0.06:
        motion = 0.28
    else:
        motion = -0.15

    direction = 0.0
    jump_penalty = 0.0
    if previous_velocity is not None:
        cosine = _cosine(previous_velocity, new_velocity)
        direction = 0.35 * cosine
        if cosine < -0.75 and not bounce_like:
            jump_penalty += 0.45
        elif bounce_like:
            # Direction change is expected at bounce — do not treat as teleport.
            direction = 0.1
            motion += 0.12
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
            _make_tracking_point(
                frame_index=observation.frame_index,
                timestamp_seconds=observation.timestamp_seconds,
                provenance="OBSERVED",
                candidate_id=observation.candidate_id,
                bounding_box=observation.bounding_box,
                x=observation.x,
                y=observation.y,
                normalized_x=observation.normalized_x,
                normalized_y=observation.normalized_y,
                confidence=observation.confidence,
                uncertainty=0.0,
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
        # Short gaps only; longer fills need stronger bilateral support.
        allow_gap = frame_delta - 1 <= PREFERRED_GAP or (
            frame_delta - 1 <= MAX_RECOVERABLE_GAP
            and observation.confidence >= 0.35
            and following.confidence >= 0.35
        )
        if not allow_gap:
            continue
        for missing_offset in range(1, frame_delta):
            fraction = missing_offset / frame_delta
            gap_uncertainty = min(
                0.85,
                0.2 + 0.12 * missing_offset + 0.08 * (frame_delta - 1),
            )
            # Linear fill is TRACKER_RECOVERED; quadratic only for tiny gaps.
            if frame_delta - 1 <= 2:
                provenance: TrackingProvenance = "TRACKER_RECOVERED"
                x = _lerp(observation.x, following.x, fraction)
                y = _lerp(observation.y, following.y, fraction)
                nx = _lerp(observation.normalized_x, following.normalized_x, fraction)
                ny = _lerp(observation.normalized_y, following.normalized_y, fraction)
            elif frame_delta - 1 <= PREFERRED_GAP:
                provenance = "PHYSICS_RECONSTRUCTED"
                # Mild quadratic ease using endpoint velocities — not metric projectile.
                x, y, nx, ny = _quadratic_gap_point(
                    observation,
                    following,
                    fraction,
                )
            else:
                provenance = "PROJECTED"
                x = _lerp(observation.x, following.x, fraction)
                y = _lerp(observation.y, following.y, fraction)
                nx = _lerp(observation.normalized_x, following.normalized_x, fraction)
                ny = _lerp(observation.normalized_y, following.normalized_y, fraction)
                gap_uncertainty = min(0.95, gap_uncertainty + 0.15)
            confidence = min(
                observation.confidence,
                following.confidence,
            ) * 0.55 * (1 - 0.15 * fraction) * (1 - 0.25 * gap_uncertainty)
            points.append(
                _make_tracking_point(
                    frame_index=observation.frame_index + missing_offset,
                    timestamp_seconds=(
                        observation.frame_index + missing_offset
                    ) / fps,
                    provenance=provenance,
                    candidate_id=None,
                    bounding_box=None,
                    x=x,
                    y=y,
                    normalized_x=nx,
                    normalized_y=ny,
                    confidence=_clamp(confidence),
                    uncertainty=round(gap_uncertainty, 6),
                    prediction_error=None,
                    inside_pitch_corridor=None,
                )
            )
    points.sort(key=lambda point: point.frame_index)
    _assign_point_velocities(points, fps)
    return points


def _make_tracking_point(
    *,
    frame_index: int,
    timestamp_seconds: float,
    provenance: TrackingProvenance,
    candidate_id: str | None,
    bounding_box: list[float] | None,
    x: float,
    y: float,
    normalized_x: float,
    normalized_y: float,
    confidence: float,
    uncertainty: float,
    prediction_error: float | None,
    inside_pitch_corridor: bool | None,
) -> TrackingPoint:
    return TrackingPoint(
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        source=_PROVENANCE_TO_SOURCE[provenance],
        provenance=provenance,
        candidate_id=candidate_id,
        bounding_box=bounding_box,
        x=x,
        y=y,
        image_x_px=x,
        image_y_px=y,
        normalized_x=normalized_x,
        normalized_y=normalized_y,
        confidence=confidence,
        detector_confidence=(confidence if provenance == "OBSERVED" else None),
        tracking_confidence=confidence,
        valid=True,
        uncertainty=uncertainty,
        vx=0.0,
        vy=0.0,
        prediction_error=prediction_error,
        inside_pitch_corridor=inside_pitch_corridor,
    )


def _quadratic_gap_point(
    start: RawTrackingCandidate,
    end: RawTrackingCandidate,
    fraction: float,
) -> tuple[float, float, float, float]:
    # ponytail: ease-in/out between endpoints; not a true ballistic model.
    ease = fraction * fraction * (3 - 2 * fraction)
    return (
        _lerp(start.x, end.x, ease),
        _lerp(start.y, end.y, ease),
        _lerp(start.normalized_x, end.normalized_x, ease),
        _lerp(start.normalized_y, end.normalized_y, ease),
    )


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


def _track_quality(confidence: float, reliable: bool) -> str:
    if not reliable:
        return "failed"
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.5:
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


def _detect_primary_bounce(
    points: list[TrackingPoint],
    *,
    fps: float,
    width: int,
    height: int,
) -> PrimaryBounceResult:
    """Image-space primary bounce v1 — not merely lowest y."""
    empty = PrimaryBounceResult(
        bounce_detected="uncertain",
        confidence=0.0,
        evidence=[],
        warnings=["Insufficient track points for bounce estimation."],
    )
    if len(points) < 6:
        return empty

    # Pitch-region soft prior: middle band of the delivery span in image y.
    ys = [point.normalized_y for point in points]
    y_min, y_max = min(ys), max(ys)
    pitch_lo = y_min + 0.2 * (y_max - y_min)
    pitch_hi = y_min + 0.85 * (y_max - y_min)

    best_index = None
    best_score = -math.inf
    evidence_best: list[str] = []
    for index in range(2, len(points) - 2):
        point = points[index]
        before = points[index - 1]
        after = points[index + 1]
        farther_before = points[index - 2]
        farther_after = points[index + 2]
        vy_before = point.normalized_y - farther_before.normalized_y
        vy_after = farther_after.normalized_y - point.normalized_y
        # Vertical turn: descending then ascending in image coords (+y down).
        vertical_turn = vy_before > 0.002 and vy_after < -0.001
        v_before = (point.normalized_x - before.normalized_x, point.normalized_y - before.normalized_y)
        v_after = (after.normalized_x - point.normalized_x, after.normalized_y - point.normalized_y)
        direction_change = 1.0 - max(-1.0, min(1.0, _cosine(v_before, v_after)))
        speed_before = math.hypot(*v_before)
        speed_after = math.hypot(*v_after)
        speed_ratio = 0.0
        if speed_before > 1e-6 and speed_after > 1e-6:
            speed_ratio = abs(math.log(speed_after / speed_before))
        in_pitch = pitch_lo <= point.normalized_y <= pitch_hi
        # Soft pitch prior — never hard reject outside.
        pitch_score = 0.2 if in_pitch else -0.05
        # Continuity: prefer observed points.
        continuity = 0.15 if point.provenance == "OBSERVED" else -0.05
        score = (
            (0.55 if vertical_turn else -0.2)
            + 0.35 * direction_change
            + 0.15 * min(1.0, speed_ratio)
            + pitch_score
            + continuity
            + 0.1 * point.confidence
        )
        # Mild preference near local max image-y (groundward), not sole criterion.
        local_peak = (
            point.normalized_y >= before.normalized_y
            and point.normalized_y >= after.normalized_y
        )
        if local_peak:
            score += 0.15
        if score > best_score:
            best_score = score
            best_index = index
            evidence_best = []
            if vertical_turn:
                evidence_best.append("vertical_motion_reversal")
            if direction_change > 0.4:
                evidence_best.append("direction_change")
            if speed_ratio > 0.25:
                evidence_best.append("speed_change")
            if in_pitch:
                evidence_best.append("pitch_region_context")
            if local_peak:
                evidence_best.append("local_groundward_peak")
            if continuity > 0:
                evidence_best.append("observed_continuity")

    warnings: list[str] = []
    if best_index is None or best_score < 0.35:
        # Full toss / no clear bounce.
        if best_score < 0.1:
            return PrimaryBounceResult(
                bounce_detected=False,
                confidence=round(_clamp(0.35 + abs(min(0.0, best_score))), 3),
                evidence=["no_clear_discontinuity"],
                warnings=["Trajectory is consistent with a full toss or weak bounce signal."],
            )
        return PrimaryBounceResult(
            bounce_detected="uncertain",
            bounce_frame=points[best_index].frame_index if best_index is not None else None,
            bounce_timestamp_seconds=(
                points[best_index].timestamp_seconds if best_index is not None else None
            ),
            bounce_x=points[best_index].x if best_index is not None else None,
            bounce_y=points[best_index].y if best_index is not None else None,
            bounce_normalized_x=(
                points[best_index].normalized_x if best_index is not None else None
            ),
            bounce_normalized_y=(
                points[best_index].normalized_y if best_index is not None else None
            ),
            confidence=round(_clamp(best_score / 1.5), 3),
            evidence=evidence_best or ["weak_candidate"],
            warnings=["Bounce hypothesis is weak; treat as uncertain."],
        )

    point = points[best_index]
    confidence = _clamp(0.35 + best_score * 0.4)
    if point.provenance != "OBSERVED":
        confidence *= 0.85
        warnings.append("Bounce falls on a non-observed track point.")
    if point.x > width or point.y > height:
        warnings.append("Bounce point near or outside frame bounds.")
    return PrimaryBounceResult(
        bounce_detected=True,
        bounce_frame=point.frame_index,
        bounce_timestamp_seconds=point.timestamp_seconds,
        bounce_x=point.x,
        bounce_y=point.y,
        bounce_normalized_x=point.normalized_x,
        bounce_normalized_y=point.normalized_y,
        confidence=round(confidence, 3),
        evidence=evidence_best,
        warnings=warnings,
    )


def _smooth_track_around_bounce(
    points: list[TrackingPoint],
    bounce_frame: int,
) -> list[TrackingPoint]:
    """Light pre/post bounce smoothing — do not smooth away the bounce kink."""
    if len(points) < 5:
        return points
    bounce_index = next(
        (index for index, point in enumerate(points) if point.frame_index == bounce_frame),
        None,
    )
    if bounce_index is None:
        return points
    smoothed = list(points)
    for index, point in enumerate(points):
        if point.provenance == "OBSERVED":
            continue
        if abs(index - bounce_index) <= 1:
            continue
        # Separate windows: pre-bounce and post-bounce.
        if index < bounce_index:
            window = points[max(0, index - 1) : min(bounce_index, index + 2)]
        else:
            window = points[max(bounce_index + 1, index - 1) : index + 2]
        if len(window) < 2:
            continue
        point.x = statistics.fmean(item.x for item in window)
        point.y = statistics.fmean(item.y for item in window)
        point.normalized_x = statistics.fmean(item.normalized_x for item in window)
        point.normalized_y = statistics.fmean(item.normalized_y for item in window)
        smoothed[index] = point
    return smoothed


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
    bounce: PrimaryBounceResult | None,
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
                    _draw_short_history(frame, history, debug=True)
                    _draw_tracking_point(frame, point, debug=True)
                if (
                    bounce is not None
                    and bounce.bounce_detected is True
                    and bounce.bounce_frame == frame_index
                    and bounce.bounce_x is not None
                    and bounce.bounce_y is not None
                ):
                    _draw_bounce_marker(
                        frame,
                        bounce.bounce_x,
                        bounce.bounce_y,
                        debug=True,
                    )
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
            progress = 58 + int(processed_frames / total_frames * 18)
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


def _render_delivery_replay(
    *,
    analysis_id: str,
    stored_filename: str,
    total_frames: int,
    fps: float,
    width: int,
    height: int,
    primary_track: list[TrackingPoint],
    bounce: PrimaryBounceResult,
    job_id: str,
) -> Path:
    """Polished user replay on CLEAN original frames — CV never sees overlays."""
    source_path = VIDEO_ANALYSIS_ROOT / analysis_id / "raw" / stored_filename
    output_dir = _tracking_output_dir(analysis_id)
    intermediate_path = output_dir / "delivery_replay_intermediate.avi"
    encoded_path = output_dir / "delivery_replay_encoded.mp4"
    final_path = output_dir / DELIVERY_REPLAY_FILENAME
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        capture.release()
        raise VideoBallTrackingError(
            "OpenCV could not open the original analysis video for replay."
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
        raise VideoBallTrackingError("Could not create the delivery replay video.")

    overlay_rgba = _optional_scene_overlay_rgba(analysis_id, width, height)
    point_by_frame = {point.frame_index: point for point in primary_track}
    trail_frames = max(2, int(round(fps * TRAIL_DURATION_SECONDS)))
    history: list[TrackingPoint] = []
    processed_frames = 0
    try:
        for frame_index in range(total_frames):
            ok, frame = capture.read()
            if not ok:
                raise VideoBallTrackingError(
                    f"Replay decoding stopped at frame {frame_index}."
                )
            # Scene overlay is secondary display only; never fed back into CV.
            if overlay_rgba is not None:
                frame = _composite_bgr_with_rgba(frame, overlay_rgba)
            point = point_by_frame.get(frame_index)
            if point is not None:
                history.append(point)
                history = history[-trail_frames:]
                _draw_short_history(frame, history, debug=False)
                _draw_replay_ball(frame, point)
            if (
                bounce.bounce_detected is True
                and bounce.bounce_frame == frame_index
                and bounce.bounce_x is not None
                and bounce.bounce_y is not None
            ):
                _draw_bounce_marker(
                    frame,
                    bounce.bounce_x,
                    bounce.bounce_y,
                    debug=False,
                )
            writer.write(frame)
            processed_frames += 1
            progress = 76 + int(processed_frames / total_frames * 14)
            video_ball_tracking_job_store.update(
                job_id,
                status="rendering_video",
                progress=min(90, progress),
                message=(
                    f"Rendering delivery replay frame {processed_frames} "
                    f"of {total_frames}."
                ),
            )
    finally:
        capture.release()
        writer.release()
    if processed_frames != total_frames:
        raise VideoBallTrackingError(
            "Delivery replay does not contain every original frame."
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
            "Could not encode a browser-compatible delivery replay."
        ) from exc
    finally:
        intermediate_path.unlink(missing_ok=True)
    return final_path


def _physics_replay_points(
    physics: DeliveryPhysicsResult,
    width: int,
    height: int,
) -> list[TrackingPoint]:
    """Adapt backend physics samples to the established replay renderer."""
    provenance_map: dict[str, TrackingProvenance] = {
        "OBSERVED": "OBSERVED",
        "RECONSTRUCTED": "PHYSICS_RECONSTRUCTED",
        "PROJECTED": "PROJECTED",
    }
    points: list[TrackingPoint] = []
    previous = None
    for sample in physics.trajectory_samples:
        if not (
            math.isfinite(sample.pixel_x)
            and math.isfinite(sample.pixel_y)
            and 0 <= sample.pixel_x < width
            and 0 <= sample.pixel_y < height
        ):
            continue
        vx = 0.0
        vy = 0.0
        if previous is not None:
            elapsed = sample.timestamp_seconds - previous.timestamp_seconds
            if elapsed > 0:
                vx = (sample.pixel_x - previous.pixel_x) / elapsed
                vy = (sample.pixel_y - previous.pixel_y) / elapsed
        points.append(
            TrackingPoint(
                frame_index=sample.frame_index,
                timestamp_seconds=sample.timestamp_seconds,
                source=(
                    "observed"
                    if sample.provenance == "OBSERVED"
                    else "predicted"
                    if sample.provenance == "PROJECTED"
                    else "recovered"
                ),
                provenance=provenance_map[sample.provenance],
                x=sample.pixel_x,
                y=sample.pixel_y,
                normalized_x=sample.pixel_x / width,
                normalized_y=sample.pixel_y / height,
                confidence=sample.confidence,
                uncertainty=1.0 - sample.confidence,
                vx=vx,
                vy=vy,
            )
        )
        previous = sample
    return points


def _physics_replay_bounce(
    physics: DeliveryPhysicsResult,
    width: int,
    height: int,
) -> PrimaryBounceResult | None:
    bounce = physics.bounce
    if (
        bounce.status == "INSUFFICIENT_EVIDENCE"
        or bounce.frame_index is None
        or bounce.pixel_x is None
        or bounce.pixel_y is None
        or not 0 <= bounce.pixel_x < width
        or not 0 <= bounce.pixel_y < height
    ):
        return None
    return PrimaryBounceResult(
        bounce_detected=True,
        bounce_frame=bounce.frame_index,
        bounce_timestamp_seconds=bounce.timestamp_seconds,
        bounce_x=bounce.pixel_x,
        bounce_y=bounce.pixel_y,
        bounce_normalized_x=bounce.pixel_x / width,
        bounce_normalized_y=bounce.pixel_y / height,
        confidence=bounce.confidence_score,
        evidence=bounce.evidence,
        warnings=[],
    )


def _optional_scene_overlay_rgba(
    analysis_id: str,
    width: int,
    height: int,
):
    """Load accepted guided calibration overlay if present; soft/secondary only."""
    try:
        from .stump_detector_service import build_scene_overlay_rgba
        from .video_calibration_service import load_video_calibration

        calibration = load_video_calibration(analysis_id)
    except Exception:
        return None
    if getattr(calibration, "scene_overlay_status", None) not in {"ready", None}:
        # Still allow overlay from locked wickets even if video overlay failed.
        pass
    try:
        striker = calibration.striker_wicket.box
        non_striker = calibration.non_striker_wicket.box
        return build_scene_overlay_rgba(
            frame_width=width,
            frame_height=height,
            striker_bbox={
                "x": striker.x * width,
                "y": striker.y * height,
                "width": striker.width * width,
                "height": striker.height * height,
            },
            non_striker_bbox={
                "x": non_striker.x * width,
                "y": non_striker.y * height,
                "width": non_striker.width * width,
                "height": non_striker.height * height,
            },
        )
    except Exception:
        return None


def _composite_bgr_with_rgba(frame_bgr, overlay_rgba):
    """Alpha-composite a PIL RGBA overlay onto an OpenCV BGR frame (display only)."""
    from PIL import Image

    base = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    # Soften scene overlay so the ball remains dominant.
    overlay = overlay_rgba.copy()
    alpha = overlay.split()[-1].point(lambda value: int(value * 0.55))
    overlay.putalpha(alpha)
    composed = Image.alpha_composite(base, overlay).convert("RGB")
    return cv2.cvtColor(np.array(composed), cv2.COLOR_RGB2BGR)


def _draw_short_history(
    frame,
    history: list[TrackingPoint],
    *,
    debug: bool,
) -> None:
    for index, (first, second) in enumerate(zip(history, history[1:]), start=1):
        strength = index / max(1, len(history) - 1)
        if debug:
            color = _provenance_bgr(second.provenance, strength)
        else:
            # User replay: subtle fade, provenance nearly invisible.
            color = (
                round(40 + 40 * strength),
                round(180 + 50 * strength),
                round(40 + 30 * strength),
            )
        thickness = 2 if not debug else 1
        cv2.line(
            frame,
            (round(first.x), round(first.y)),
            (round(second.x), round(second.y)),
            color,
            thickness,
            cv2.LINE_AA,
        )


def _provenance_bgr(provenance: TrackingProvenance, strength: float = 1.0):
    palette = {
        "OBSERVED": (80, 230, 80),
        "TRACKER_RECOVERED": (0, 150, 255),
        "PHYSICS_RECONSTRUCTED": (0, 200, 255),
        "PROJECTED": (0, 230, 255),
    }
    base = palette.get(provenance, (180, 180, 180))
    return tuple(round(channel * (0.45 + 0.55 * strength)) for channel in base)


def _draw_tracking_point(frame, point: TrackingPoint, *, debug: bool = True) -> None:
    color = _provenance_bgr(point.provenance)
    center = (round(point.x), round(point.y))
    cv2.circle(frame, center, 8, color, 2, cv2.LINE_AA)
    cv2.circle(frame, center, 2, color, -1, cv2.LINE_AA)
    if debug:
        cv2.putText(
            frame,
            f"{point.provenance} {point.confidence:.2f}",
            (center[0] + 10, max(18, center[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )


def _draw_replay_ball(frame, point: TrackingPoint) -> None:
    center = (round(point.x), round(point.y))
    # Soft glow ring — ball dominant, scene secondary.
    cv2.circle(frame, center, 14, (60, 220, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, center, 8, (40, 255, 180), 2, cv2.LINE_AA)
    cv2.circle(frame, center, 3, (255, 255, 255), -1, cv2.LINE_AA)


def _draw_bounce_marker(frame, x: float, y: float, *, debug: bool) -> None:
    center = (round(x), round(y))
    color = (40, 180, 255) if not debug else (0, 140, 255)
    cv2.drawMarker(
        frame,
        center,
        color,
        markerType=cv2.MARKER_TRIANGLE_UP,
        markerSize=18 if not debug else 22,
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    if debug:
        cv2.putText(
            frame,
            "bounce",
            (center[0] + 12, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
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
    panel_width = min(frame.shape[1] - 10, 380)
    cv2.rectangle(frame, (10, 10), (panel_width, 82), (18, 18, 18), -1)
    cv2.putText(
        frame,
        "Delivery Tracking v2 - debug",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    state = (
        point.provenance
        if point is not None
        else "Searching" if active else "Inactive"
    )
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
        "provenance",
        "candidate_id",
        "x",
        "y",
        "normalized_x",
        "normalized_y",
        "confidence",
        "uncertainty",
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
            "provenance": point.provenance,
            "candidate_id": point.candidate_id or "",
            "x": point.x,
            "y": point.y,
            "normalized_x": point.normalized_x,
            "normalized_y": point.normalized_y,
            "confidence": point.confidence,
            "uncertainty": point.uncertainty,
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
    *,
    failure_code: str | None = None,
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
        failure_code=failure_code or _tracking_failure_code(message),
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


def _tracking_failure_code(message: str) -> str:
    lowered = message.lower()
    if "fps" in lowered:
        return "VIDEO_FPS_UNAVAILABLE"
    if "detection" in lowered:
        return "BALL_DETECTION_FAILED"
    if "saved tracking" in lowered:
        return "TRACK_RESULT_LOAD_FAILED"
    return "TRACK_UNAVAILABLE"


def _clear_previous_tracking_outputs(output_dir: Path) -> None:
    for filename in (
        TRACKING_RESULT_FILENAME,
        PHYSICS_RESULT_FILENAME,
        TRACKING_CSV_FILENAME,
        TRACKING_SUMMARY_FILENAME,
        TRACKING_VIDEO_FILENAME,
        DELIVERY_REPLAY_FILENAME,
        "tracking_debug_intermediate.avi",
        "tracking_debug_encoded.mp4",
        "delivery_replay_intermediate.avi",
        "delivery_replay_encoded.mp4",
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
