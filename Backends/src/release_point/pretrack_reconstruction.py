"""Release-specific pre-track reconstruction from persisted detector candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from .features import (
    BallObservation,
    BowlerPoseSequence,
    Keypoint,
    ReleasePointConfig,
    TrackObservation,
    body_scale_px,
    nearest_pose,
    selected_wrist,
    trajectory_direction_consistency,
)


PRETRACK_RECOVERED = "PRETRACK_RECOVERED"
RECONSTRUCTED = "RECONSTRUCTED"


@dataclass(frozen=True)
class BackwardProjection:
    frame_index: int
    expected_x: float
    expected_y: float
    gate_radius_px: float
    fit_quality: float


@dataclass(frozen=True)
class CandidateGate:
    candidate_id: str
    frame_index: int
    x: float
    y: float
    detector_confidence: float
    rank: int
    distance_px: float
    gate_radius_px: float
    score: float
    accepted: bool
    rejection_reason: str | None
    score_components: dict[str, float]


@dataclass(frozen=True)
class RecoveredPoint:
    frame_index: int
    observation: BallObservation | None
    track_point: TrackObservation
    projection: BackwardProjection
    provenance: str
    score: float
    reason: str


@dataclass(frozen=True)
class PretrackHypothesis:
    first_primary_track_frame: int | None
    search_window: dict[str, int | None]
    projections: list[BackwardProjection] = field(default_factory=list)
    candidate_gates: list[CandidateGate] = field(default_factory=list)
    recovered_points: list[RecoveredPoint] = field(default_factory=list)
    fit_quality: float = 0.0
    status: str = "not_run"
    reason: str | None = None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "first_primary_track_frame": self.first_primary_track_frame,
            "search_window": self.search_window,
            "fit_quality": round(self.fit_quality, 6),
            "predicted_backward_positions": [
                {
                    "frame_index": item.frame_index,
                    "expected_x": round(item.expected_x, 6),
                    "expected_y": round(item.expected_y, 6),
                    "gate_radius_px": round(item.gate_radius_px, 6),
                    "fit_quality": round(item.fit_quality, 6),
                }
                for item in self.projections
            ],
            "candidate_gates": [
                {
                    "candidate_id": item.candidate_id,
                    "frame_index": item.frame_index,
                    "x": round(item.x, 6),
                    "y": round(item.y, 6),
                    "detector_confidence": round(item.detector_confidence, 6),
                    "rank": item.rank,
                    "distance_px": round(item.distance_px, 6),
                    "gate_radius_px": round(item.gate_radius_px, 6),
                    "score": round(item.score, 6),
                    "accepted": item.accepted,
                    "rejection_reason": item.rejection_reason,
                    "score_components": {
                        key: round(value, 6)
                        for key, value in item.score_components.items()
                    },
                }
                for item in self.candidate_gates
            ],
            "recovered_pretrack_points": [
                {
                    "frame_index": item.frame_index,
                    "candidate_id": None
                    if item.observation is None
                    else item.observation.candidate_id,
                    "x": round(item.track_point.x, 6),
                    "y": round(item.track_point.y, 6),
                    "confidence": round(item.track_point.confidence, 6),
                    "provenance": item.provenance,
                    "score": round(item.score, 6),
                    "reason": item.reason,
                }
                for item in self.recovered_points
            ],
        }


def reconstruct_pretrack_hypothesis(
    *,
    detections_by_frame: dict[int, list[BallObservation]],
    primary_track: list[TrackObservation],
    pose_sequence: BowlerPoseSequence | None,
    wrist_keypoint_name: str | None,
    config: ReleasePointConfig,
) -> PretrackHypothesis:
    if len(primary_track) < config.minimum_track_points:
        return PretrackHypothesis(
            first_primary_track_frame=None,
            search_window={"start": None, "end": None},
            status="skipped",
            reason="insufficient_primary_track",
        )
    first_track = primary_track[0]
    fit_points = _first_reliable_points(primary_track, config)
    if len(fit_points) < 2:
        return PretrackHypothesis(
            first_primary_track_frame=first_track.frame_index,
            search_window={"start": None, "end": None},
            status="skipped",
            reason="insufficient_reliable_free_flight_points",
        )
    first, last = fit_points[0], fit_points[-1]
    frame_span = max(1, last.frame_index - first.frame_index)
    vx = (last.x - first.x) / frame_span
    vy = (last.y - first.y) / frame_span
    fit_quality = trajectory_direction_consistency(fit_points, config)
    start = max(0, first_track.frame_index - config.pretrack_search_back_frames)
    end = max(0, first_track.frame_index - 1)
    projections: list[BackwardProjection] = []
    gates: list[CandidateGate] = []
    recovered: list[RecoveredPoint] = []
    for frame_index in range(end, start - 1, -1):
        frames_back = first_track.frame_index - frame_index
        expected_x = first_track.x - vx * frames_back
        expected_y = first_track.y - vy * frames_back
        gate = (
            config.pretrack_projection_base_gate_px
            + config.pretrack_projection_gate_growth_px * frames_back
        )
        projection = BackwardProjection(
            frame_index=frame_index,
            expected_x=expected_x,
            expected_y=expected_y,
            gate_radius_px=gate,
            fit_quality=fit_quality,
        )
        projections.append(projection)
        candidates = detections_by_frame.get(frame_index, [])
        if not candidates:
            recovered.append(
                _reconstructed_point(
                    projection=projection,
                    first_track=first_track,
                    vx=vx,
                    vy=vy,
                    reason="no_persisted_detector_candidate",
                )
            )
            continue
        scored = [
            _score_candidate(
                candidate=candidate,
                projection=projection,
                pose_sequence=pose_sequence,
                wrist_keypoint_name=wrist_keypoint_name,
                fit_quality=fit_quality,
                config=config,
            )
            for candidate in candidates
        ]
        best = max(scored, key=lambda item: item.score)
        for gate_result in scored:
            accepted = gate_result.candidate_id == best.candidate_id and (
                gate_result.accepted
            )
            gates.append(
                CandidateGate(
                    **{
                        **gate_result.__dict__,
                        "accepted": accepted,
                        "rejection_reason": None
                        if accepted
                        else gate_result.rejection_reason
                        or "lower_scoring_candidate_same_frame",
                    }
                )
            )
        if best.accepted:
            candidate = next(
                item for item in candidates if item.candidate_id == best.candidate_id
            )
            recovered.append(
                RecoveredPoint(
                    frame_index=frame_index,
                    observation=candidate,
                    track_point=TrackObservation(
                        frame_index=frame_index,
                        timestamp_seconds=candidate.timestamp_seconds,
                        x=candidate.x,
                        y=candidate.y,
                        normalized_x=candidate.normalized_x,
                        normalized_y=candidate.normalized_y,
                        confidence=best.score,
                        provenance=PRETRACK_RECOVERED,
                        candidate_id=candidate.candidate_id,
                        uncertainty=projection.gate_radius_px,
                        vx=vx,
                        vy=vy,
                        inside_pitch_corridor=candidate.inside_pitch_corridor,
                    ),
                    projection=projection,
                    provenance=PRETRACK_RECOVERED,
                    score=best.score,
                    reason="persisted_candidate_matches_backward_projection",
                )
            )
        else:
            recovered.append(
                _reconstructed_point(
                    projection=projection,
                    first_track=first_track,
                    vx=vx,
                    vy=vy,
                    reason=best.rejection_reason or "no_candidate_passed_gate",
                )
            )
    return PretrackHypothesis(
        first_primary_track_frame=first_track.frame_index,
        search_window={"start": start, "end": end},
        projections=list(reversed(projections)),
        candidate_gates=sorted(gates, key=lambda item: (item.frame_index, item.rank)),
        recovered_points=sorted(recovered, key=lambda item: item.frame_index),
        fit_quality=fit_quality,
        status="ready",
    )


def _first_reliable_points(
    primary_track: list[TrackObservation],
    config: ReleasePointConfig,
) -> list[TrackObservation]:
    points: list[TrackObservation] = []
    previous_frame: int | None = None
    for point in primary_track:
        if previous_frame is not None:
            gap = point.frame_index - previous_frame
            if gap <= 0 or gap > config.max_track_gap_frames:
                break
        points.append(point)
        previous_frame = point.frame_index
        if len(points) >= config.track_fit_points:
            break
    return points


def _score_candidate(
    *,
    candidate: BallObservation,
    projection: BackwardProjection,
    pose_sequence: BowlerPoseSequence | None,
    wrist_keypoint_name: str | None,
    fit_quality: float,
    config: ReleasePointConfig,
) -> CandidateGate:
    distance = math.hypot(candidate.x - projection.expected_x, candidate.y - projection.expected_y)
    distance_score = max(0.0, 1.0 - distance / max(1.0, projection.gate_radius_px))
    confidence_score = math.sqrt(max(0.0, min(1.0, candidate.confidence)))
    rank_score = 1.0 if candidate.rank == 1 else max(0.2, 1.0 / candidate.rank)
    wrist_score = _wrist_proximity_score(
        frame_index=candidate.frame_index,
        candidate=candidate,
        pose_sequence=pose_sequence,
        wrist_keypoint_name=wrist_keypoint_name,
    )
    roi_score = (
        0.5
        if candidate.inside_pitch_corridor is None
        else 1.0
        if candidate.inside_pitch_corridor
        else 0.35
    )
    score_components = {
        "backward_projection_distance": distance_score,
        "detector_confidence": confidence_score,
        "candidate_rank": rank_score,
        "trajectory_fit_quality": fit_quality,
        "bowling_wrist_proximity": wrist_score,
        "scene_roi": roi_score,
    }
    score = (
        0.45 * distance_score
        + 0.18 * confidence_score
        + 0.12 * rank_score
        + 0.10 * fit_quality
        + 0.10 * wrist_score
        + 0.05 * roi_score
    )
    rejection = None
    if distance > projection.gate_radius_px:
        rejection = "outside_backward_projection_gate"
    elif score < config.pretrack_min_candidate_score:
        rejection = "compatibility_score_below_threshold"
    return CandidateGate(
        candidate_id=candidate.candidate_id,
        frame_index=candidate.frame_index,
        x=candidate.x,
        y=candidate.y,
        detector_confidence=candidate.confidence,
        rank=candidate.rank,
        distance_px=distance,
        gate_radius_px=projection.gate_radius_px,
        score=max(0.0, min(1.0, score)),
        accepted=rejection is None,
        rejection_reason=rejection,
        score_components=score_components,
    )


def _wrist_proximity_score(
    *,
    frame_index: int,
    candidate: BallObservation,
    pose_sequence: BowlerPoseSequence | None,
    wrist_keypoint_name: str | None,
) -> float:
    if pose_sequence is None or wrist_keypoint_name is None:
        return 0.5
    pose = nearest_pose(frame_index, pose_sequence)
    if pose is None:
        return 0.5
    _, wrist = selected_wrist(pose, wrist_keypoint_name)
    if wrist is None or wrist.confidence < 0.2:
        return 0.35
    scale = body_scale_px(pose) or 55.0
    distance = math.hypot(candidate.x - wrist.x, candidate.y - wrist.y)
    return max(0.0, min(1.0, 1.0 - distance / max(1.0, 2.4 * scale)))


def _reconstructed_point(
    *,
    projection: BackwardProjection,
    first_track: TrackObservation,
    vx: float,
    vy: float,
    reason: str,
) -> RecoveredPoint:
    normalized_x = first_track.normalized_x
    normalized_y = first_track.normalized_y
    if first_track.x:
        normalized_x = projection.expected_x * first_track.normalized_x / first_track.x
    if first_track.y:
        normalized_y = projection.expected_y * first_track.normalized_y / first_track.y
    point = TrackObservation(
        frame_index=projection.frame_index,
        timestamp_seconds=first_track.timestamp_seconds,
        x=projection.expected_x,
        y=projection.expected_y,
        normalized_x=normalized_x,
        normalized_y=normalized_y,
        confidence=0.0,
        provenance=RECONSTRUCTED,
        candidate_id=None,
        uncertainty=projection.gate_radius_px,
        vx=vx,
        vy=vy,
        inside_pitch_corridor=None,
    )
    return RecoveredPoint(
        frame_index=projection.frame_index,
        observation=None,
        track_point=point,
        projection=projection,
        provenance=RECONSTRUCTED,
        score=0.0,
        reason=reason,
    )
