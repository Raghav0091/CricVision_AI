"""Release-region ball observation recovery from persisted detector candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import statistics
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


OBSERVED_PRIMARY = "OBSERVED_PRIMARY"
OBSERVED_RELEASE_RECOVERED = "OBSERVED_RELEASE_RECOVERED"
RECONSTRUCTED = "RECONSTRUCTED"
NO_CREDIBLE_RELEASE_REGION_CHAIN = "NO_CREDIBLE_RELEASE_REGION_CHAIN"


@dataclass(frozen=True)
class RecoveryProjection:
    frame_index: int
    expected_x: float
    expected_y: float
    gate_radius_px: float

    def diagnostics(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "expected_x": round(self.expected_x, 6),
            "expected_y": round(self.expected_y, 6),
            "gate_radius_px": round(self.gate_radius_px, 6),
        }


@dataclass(frozen=True)
class RecoveryCandidate:
    frame_index: int
    timestamp_seconds: float
    candidate_id: str
    x: float
    y: float
    normalized_x: float
    normalized_y: float
    detector_confidence: float
    rank: int
    inside_pitch_corridor: bool | None
    selected_by_primary_tracker: bool
    backward_compatibility: float
    forward_compatibility: float
    pose_support: float | None
    static_risk: float
    node_score: float
    rejection_reason: str | None = None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "candidate_id": self.candidate_id,
            "x": round(self.x, 6),
            "y": round(self.y, 6),
            "normalized_x": round(self.normalized_x, 6),
            "normalized_y": round(self.normalized_y, 6),
            "detector_confidence": round(self.detector_confidence, 6),
            "rank": self.rank,
            "inside_pitch_corridor": self.inside_pitch_corridor,
            "candidate_source": "persisted_detector_top_k",
            "selected_by_primary_tracker": self.selected_by_primary_tracker,
            "backward_compatibility": round(self.backward_compatibility, 6),
            "forward_compatibility": round(self.forward_compatibility, 6),
            "pose_support": None
            if self.pose_support is None
            else round(self.pose_support, 6),
            "static_risk": round(self.static_risk, 6),
            "node_score": round(self.node_score, 6),
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class RecoveryEdge:
    from_candidate_id: str
    to_candidate_id: str
    frame_gap: int
    distance_px: float
    direction_compatibility: float
    velocity_compatibility: float
    static_penalty: float
    score: float

    def diagnostics(self) -> dict[str, Any]:
        return {
            "from_candidate_id": self.from_candidate_id,
            "to_candidate_id": self.to_candidate_id,
            "frame_gap": self.frame_gap,
            "distance_px": round(self.distance_px, 6),
            "direction_compatibility": round(self.direction_compatibility, 6),
            "velocity_compatibility": round(self.velocity_compatibility, 6),
            "static_penalty": round(self.static_penalty, 6),
            "score": round(self.score, 6),
        }


@dataclass(frozen=True)
class RecoveredObservation:
    frame_index: int
    observation: BallObservation
    track_point: TrackObservation
    association_score: float
    temporal_support_count: int
    backward_compatibility: float
    forward_compatibility: float
    pose_support: float | None
    recovery_reason: str

    def diagnostics(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "candidate_id": self.observation.candidate_id,
            "x": round(self.observation.x, 6),
            "y": round(self.observation.y, 6),
            "detector_confidence": round(self.observation.confidence, 6),
            "rank": self.observation.rank,
            "provenance": OBSERVED_RELEASE_RECOVERED,
            "association_score": round(self.association_score, 6),
            "temporal_support_count": self.temporal_support_count,
            "backward_compatibility": round(self.backward_compatibility, 6),
            "forward_compatibility": round(self.forward_compatibility, 6),
            "pose_support": None
            if self.pose_support is None
            else round(self.pose_support, 6),
            "recovery_reason": self.recovery_reason,
            "uncertainty": round(self.track_point.uncertainty, 6),
        }


@dataclass(frozen=True)
class ReconstructedPoint:
    frame_index: int
    x: float
    y: float
    uncertainty: float
    reason: str

    def diagnostics(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "x": round(self.x, 6),
            "y": round(self.y, 6),
            "provenance": RECONSTRUCTED,
            "uncertainty": round(self.uncertainty, 6),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ReleaseRegionRecoveryResult:
    status: str
    reason: str | None
    first_primary_track_frame: int | None
    search_window: dict[str, int | None]
    projections: list[RecoveryProjection] = field(default_factory=list)
    candidates_considered: list[RecoveryCandidate] = field(default_factory=list)
    associations: list[RecoveryEdge] = field(default_factory=list)
    selected_path_candidate_ids: list[str] = field(default_factory=list)
    recovered_observations: list[RecoveredObservation] = field(default_factory=list)
    reconstructed_points: list[ReconstructedPoint] = field(default_factory=list)
    path_score: float = 0.0
    uncertainty: float | None = None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "first_primary_track_frame": self.first_primary_track_frame,
            "search_window": self.search_window,
            "path_score": round(self.path_score, 6),
            "uncertainty": None if self.uncertainty is None else round(self.uncertainty, 6),
            "predicted_backward_positions": [
                item.diagnostics() for item in self.projections
            ],
            "candidates_considered": [
                item.diagnostics() for item in self.candidates_considered
            ],
            "candidate_graph_associations": [
                item.diagnostics() for item in self.associations
            ],
            "selected_path_candidate_ids": self.selected_path_candidate_ids,
            "recovered_observed_points": [
                item.diagnostics() for item in self.recovered_observations
            ],
            "reconstructed_only_points": [
                item.diagnostics() for item in self.reconstructed_points
            ],
        }


def recover_release_region_observations(
    *,
    detections_by_frame: dict[int, list[BallObservation]],
    primary_track: list[TrackObservation],
    pose_sequence: BowlerPoseSequence | None,
    config: ReleasePointConfig,
) -> ReleaseRegionRecoveryResult:
    if len(primary_track) < config.minimum_track_points:
        return ReleaseRegionRecoveryResult(
            status="skipped",
            reason="insufficient_primary_track",
            first_primary_track_frame=None,
            search_window={"start": None, "end": None},
        )
    first_track = primary_track[0]
    fit_points = _first_reliable_points(primary_track, config)
    if len(fit_points) < 2:
        return ReleaseRegionRecoveryResult(
            status="skipped",
            reason="insufficient_reliable_free_flight_points",
            first_primary_track_frame=first_track.frame_index,
            search_window={"start": None, "end": None},
        )

    first_fit = fit_points[0]
    last_fit = fit_points[-1]
    span = max(1, last_fit.frame_index - first_fit.frame_index)
    vx = (last_fit.x - first_fit.x) / span
    vy = (last_fit.y - first_fit.y) / span
    speed = max(1.0, math.hypot(vx, vy))
    direction_quality = trajectory_direction_consistency(fit_points, config)
    start = max(0, first_track.frame_index - config.pretrack_search_back_frames)
    end = max(0, first_track.frame_index - 1)
    search_window = {"start": start, "end": end}
    selected_primary_ids = {
        str(point.candidate_id)
        for point in primary_track
        if point.candidate_id is not None
    }
    wrist_name = _wrist_name(pose_sequence)
    projections = [
        _projection_for_frame(
            frame_index=frame,
            first_track=first_track,
            vx=vx,
            vy=vy,
            config=config,
        )
        for frame in range(start, end + 1)
    ]
    projection_by_frame = {item.frame_index: item for item in projections}
    candidates = []
    for frame in range(start, end + 1):
        for observation in detections_by_frame.get(frame, []):
            candidates.append(
                _score_observation(
                    observation=observation,
                    projection=projection_by_frame[frame],
                    first_track=first_track,
                    vx=vx,
                    vy=vy,
                    speed=speed,
                    selected_primary_ids=selected_primary_ids,
                    pose_sequence=pose_sequence,
                    wrist_name=wrist_name,
                    config=config,
                )
            )
    if not candidates:
        return ReleaseRegionRecoveryResult(
            status=NO_CREDIBLE_RELEASE_REGION_CHAIN,
            reason="no_persisted_detector_candidates_in_release_region",
            first_primary_track_frame=first_track.frame_index,
            search_window=search_window,
            projections=projections,
            reconstructed_points=_reconstructed_gap(
                projections=projections,
                reason="no_persisted_detector_candidates_in_release_region",
            ),
        )

    candidates_by_frame: dict[int, list[RecoveryCandidate]] = {}
    for candidate in candidates:
        candidates_by_frame.setdefault(candidate.frame_index, []).append(candidate)
    ordered_candidates = sorted(
        candidates,
        key=lambda item: (item.frame_index, item.rank, -item.detector_confidence),
    )
    edges: dict[tuple[str, str], RecoveryEdge] = {}
    best_score: dict[str, float] = {}
    best_path: dict[str, list[RecoveryCandidate]] = {}
    for candidate in ordered_candidates:
        best_score[candidate.candidate_id] = candidate.node_score
        best_path[candidate.candidate_id] = [candidate]
        for previous in ordered_candidates:
            if previous.frame_index >= candidate.frame_index:
                break
            gap = candidate.frame_index - previous.frame_index
            if gap > max(1, config.max_track_gap_frames):
                continue
            edge = _association_edge(
                previous=previous,
                current=candidate,
                vx=vx,
                vy=vy,
                speed=speed,
            )
            edges[(previous.candidate_id, candidate.candidate_id)] = edge
            if edge.score <= 0.15:
                continue
            score = best_score[previous.candidate_id] + candidate.node_score + edge.score
            if score > best_score[candidate.candidate_id]:
                best_score[candidate.candidate_id] = score
                best_path[candidate.candidate_id] = [
                    *best_path[previous.candidate_id],
                    candidate,
                ]

    terminal_scores = []
    for candidate in ordered_candidates:
        terminal = _terminal_compatibility(candidate, first_track, vx=vx, vy=vy, speed=speed)
        path = best_path[candidate.candidate_id]
        temporal_support = len(path)
        average_score = (
            best_score[candidate.candidate_id] + terminal
        ) / max(1, temporal_support * 2)
        path_motion = _path_motion_score(path)
        terminal_scores.append(
            (
                average_score
                + 0.18 * terminal
                + 0.18 * path_motion
                + min(0.12, 0.04 * temporal_support),
                terminal,
                path_motion,
                path,
            )
        )
    best_total, terminal, path_motion, path = max(terminal_scores, key=lambda item: item[0])
    temporal_support = len(path)
    strong_pose_single = (
        temporal_support == 1
        and path[0].pose_support is not None
        and path[0].pose_support >= 0.75
        and path[0].backward_compatibility >= 0.75
    )
    credible = (
        direction_quality >= 0.55
        and terminal >= 0.45
        and path_motion >= 0.35
        and best_total >= 0.50
        and (temporal_support >= 2 or strong_pose_single)
    )
    all_edges = sorted(edges.values(), key=lambda item: (item.from_candidate_id, item.to_candidate_id))
    if not credible:
        return ReleaseRegionRecoveryResult(
            status=NO_CREDIBLE_RELEASE_REGION_CHAIN,
            reason=_no_chain_reason(
                temporal_support=temporal_support,
                terminal=terminal,
                path_motion=path_motion,
                best_total=best_total,
                direction_quality=direction_quality,
            ),
            first_primary_track_frame=first_track.frame_index,
            search_window=search_window,
            projections=projections,
            candidates_considered=_mark_rejections(candidates, selected_path=[]),
            associations=all_edges,
            selected_path_candidate_ids=[item.candidate_id for item in path],
            reconstructed_points=_reconstructed_gap(
                projections=projections,
                reason="no_credible_release_region_chain",
            ),
            path_score=max(0.0, min(1.0, best_total)),
            uncertainty=_path_uncertainty(path, first_track, config),
        )

    recovered = _recovered_observations(
        path=path,
        first_track=first_track,
        vx=vx,
        vy=vy,
        path_score=max(0.0, min(1.0, best_total)),
        terminal=terminal,
        config=config,
    )
    return ReleaseRegionRecoveryResult(
        status="ready",
        reason="credible_release_region_candidate_chain",
        first_primary_track_frame=first_track.frame_index,
        search_window=search_window,
        projections=projections,
        candidates_considered=_mark_rejections(candidates, selected_path=path),
        associations=all_edges,
        selected_path_candidate_ids=[item.candidate_id for item in path],
        recovered_observations=recovered,
        reconstructed_points=_reconstructed_gap(
            projections=[
                projection
                for projection in projections
                if projection.frame_index < path[0].frame_index
            ],
            reason="no_detector_observation_before_recovered_chain",
        ),
        path_score=max(0.0, min(1.0, best_total)),
        uncertainty=_path_uncertainty(path, first_track, config),
    )


def augment_tracking_with_recovery(
    tracking_document: dict[str, Any],
    recovery: ReleaseRegionRecoveryResult,
) -> dict[str, Any]:
    if not recovery.recovered_observations:
        augmented = dict(tracking_document)
        augmented["release_region_observation_recovery"] = recovery.diagnostics()
        return augmented
    original_track = list(tracking_document.get("primary_track") or [])
    existing_frames = {int(point.get("frame_index", -1)) for point in original_track}
    recovered_points = [
        _track_point_to_document(item.track_point)
        for item in recovery.recovered_observations
        if item.frame_index not in existing_frames
    ]
    augmented = dict(tracking_document)
    augmented["primary_track"] = sorted(
        [*recovered_points, *original_track],
        key=lambda item: int(item.get("frame_index", 0)),
    )
    augmented["release_region_observation_recovery"] = recovery.diagnostics()
    settings = dict(augmented.get("settings") or {})
    settings["release_region_observation_recovery"] = "v1"
    augmented["settings"] = settings
    return augmented


def _first_reliable_points(
    primary_track: list[TrackObservation],
    config: ReleasePointConfig,
) -> list[TrackObservation]:
    return primary_track[: max(config.minimum_free_flight_points, config.track_fit_points)]


def _projection_for_frame(
    *,
    frame_index: int,
    first_track: TrackObservation,
    vx: float,
    vy: float,
    config: ReleasePointConfig,
) -> RecoveryProjection:
    frames_back = first_track.frame_index - frame_index
    return RecoveryProjection(
        frame_index=frame_index,
        expected_x=first_track.x - vx * frames_back,
        expected_y=first_track.y - vy * frames_back,
        gate_radius_px=(
            config.pretrack_projection_base_gate_px
            + config.pretrack_projection_gate_growth_px * frames_back
        ),
    )


def _score_observation(
    *,
    observation: BallObservation,
    projection: RecoveryProjection,
    first_track: TrackObservation,
    vx: float,
    vy: float,
    speed: float,
    selected_primary_ids: set[str],
    pose_sequence: BowlerPoseSequence | None,
    wrist_name: str | None,
    config: ReleasePointConfig,
) -> RecoveryCandidate:
    distance = math.hypot(observation.x - projection.expected_x, observation.y - projection.expected_y)
    backward = max(0.0, 1.0 - distance / max(1.0, projection.gate_radius_px))
    terminal = _point_to_track_compatibility(
        x=observation.x,
        y=observation.y,
        from_frame=observation.frame_index,
        to_track=first_track,
        vx=vx,
        vy=vy,
        speed=speed,
    )
    confidence = max(0.0, min(1.0, observation.confidence))
    rank = 1.0 if observation.rank <= 1 else max(0.15, 1.0 / observation.rank)
    pose_support = _pose_support(
        observation=observation,
        pose_sequence=pose_sequence,
        wrist_name=wrist_name,
        config=config,
    )
    static_risk = _static_risk(observation, pose_sequence)
    pose_value = 0.45 if pose_support is None else pose_support
    node_score = (
        0.34 * backward
        + 0.24 * terminal
        + 0.14 * confidence
        + 0.10 * rank
        + 0.14 * pose_value
        - 0.10 * static_risk
    )
    return RecoveryCandidate(
        frame_index=observation.frame_index,
        timestamp_seconds=observation.timestamp_seconds,
        candidate_id=observation.candidate_id,
        x=observation.x,
        y=observation.y,
        normalized_x=observation.normalized_x,
        normalized_y=observation.normalized_y,
        detector_confidence=observation.confidence,
        rank=observation.rank,
        inside_pitch_corridor=observation.inside_pitch_corridor,
        selected_by_primary_tracker=observation.candidate_id in selected_primary_ids,
        backward_compatibility=backward,
        forward_compatibility=terminal,
        pose_support=pose_support,
        static_risk=static_risk,
        node_score=max(0.0, min(1.0, node_score)),
    )


def _association_edge(
    *,
    previous: RecoveryCandidate,
    current: RecoveryCandidate,
    vx: float,
    vy: float,
    speed: float,
) -> RecoveryEdge:
    gap = current.frame_index - previous.frame_index
    dx = current.x - previous.x
    dy = current.y - previous.y
    distance = math.hypot(dx, dy)
    direction = _direction_compatibility(dx=dx, dy=dy, vx=vx, vy=vy)
    expected_distance = speed * max(1, gap)
    velocity = max(0.0, 1.0 - abs(distance - expected_distance) / max(8.0, expected_distance))
    static_penalty = 1.0 if distance < max(2.0, 0.18 * expected_distance) else 0.0
    score = 0.55 * direction + 0.35 * velocity - 0.30 * static_penalty
    return RecoveryEdge(
        from_candidate_id=previous.candidate_id,
        to_candidate_id=current.candidate_id,
        frame_gap=gap,
        distance_px=distance,
        direction_compatibility=direction,
        velocity_compatibility=velocity,
        static_penalty=static_penalty,
        score=max(0.0, min(1.0, score)),
    )


def _terminal_compatibility(
    candidate: RecoveryCandidate,
    first_track: TrackObservation,
    *,
    vx: float,
    vy: float,
    speed: float,
) -> float:
    return _point_to_track_compatibility(
        x=candidate.x,
        y=candidate.y,
        from_frame=candidate.frame_index,
        to_track=first_track,
        vx=vx,
        vy=vy,
        speed=speed,
    )


def _point_to_track_compatibility(
    *,
    x: float,
    y: float,
    from_frame: int,
    to_track: TrackObservation,
    vx: float,
    vy: float,
    speed: float,
) -> float:
    gap = max(1, to_track.frame_index - from_frame)
    expected_x = to_track.x - vx * gap
    expected_y = to_track.y - vy * gap
    distance = math.hypot(x - expected_x, y - expected_y)
    gate = 26.0 + 9.0 * gap
    line_score = max(0.0, 1.0 - distance / max(1.0, gate))
    dx = to_track.x - x
    dy = to_track.y - y
    direction = _direction_compatibility(dx=dx, dy=dy, vx=vx, vy=vy)
    observed_speed = math.hypot(dx, dy) / gap
    velocity = max(0.0, 1.0 - abs(observed_speed - speed) / max(10.0, speed))
    return max(0.0, min(1.0, 0.44 * line_score + 0.34 * direction + 0.22 * velocity))


def _direction_compatibility(*, dx: float, dy: float, vx: float, vy: float) -> float:
    first = math.hypot(dx, dy)
    second = math.hypot(vx, vy)
    if first == 0 or second == 0:
        return 0.0
    cosine = (dx * vx + dy * vy) / (first * second)
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def _path_motion_score(path: list[RecoveryCandidate]) -> float:
    if len(path) == 1:
        return 0.5
    distances = [
        math.hypot(second.x - first.x, second.y - first.y)
        / max(1, second.frame_index - first.frame_index)
        for first, second in zip(path, path[1:])
    ]
    if not distances:
        return 0.0
    median = statistics.median(distances)
    if median < 2.0:
        return 0.0
    return max(0.0, min(1.0, median / 22.0))


def _pose_support(
    *,
    observation: BallObservation,
    pose_sequence: BowlerPoseSequence | None,
    wrist_name: str | None,
    config: ReleasePointConfig,
) -> float | None:
    if pose_sequence is None or pose_sequence.arm_ambiguous or wrist_name is None:
        return None
    pose = nearest_pose(observation.frame_index, pose_sequence)
    if pose is None:
        return None
    _, wrist = selected_wrist(pose, wrist_name)
    if wrist is None or wrist.confidence < config.wrist_confidence_threshold:
        return None
    scale = body_scale_px(pose) or config.close_ball_wrist_px_fallback
    distance = math.hypot(observation.x - wrist.x, observation.y - wrist.y)
    proximity = max(0.0, 1.0 - distance / max(1.0, 2.2 * scale))
    confidence = max(0.0, min(1.0, statistics.fmean([pose.confidence, wrist.confidence])))
    return max(0.0, min(1.0, 0.65 * proximity + 0.35 * confidence))


def _static_risk(
    observation: BallObservation,
    pose_sequence: BowlerPoseSequence | None,
) -> float:
    del observation, pose_sequence
    return 0.0


def _wrist_name(pose_sequence: BowlerPoseSequence | None) -> str | None:
    if pose_sequence is None or pose_sequence.arm_ambiguous:
        return None
    if pose_sequence.bowling_arm in {"left", "right"}:
        return f"{pose_sequence.bowling_arm}_wrist"
    return None


def _mark_rejections(
    candidates: list[RecoveryCandidate],
    *,
    selected_path: list[RecoveryCandidate],
) -> list[RecoveryCandidate]:
    selected_ids = {item.candidate_id for item in selected_path}
    output = []
    for candidate in candidates:
        if candidate.candidate_id in selected_ids:
            output.append(candidate)
            continue
        reasons = []
        if candidate.backward_compatibility < 0.25:
            reasons.append("weak_backward_compatibility")
        if candidate.forward_compatibility < 0.25:
            reasons.append("weak_forward_compatibility")
        if candidate.static_risk > 0.6:
            reasons.append("static_object_risk")
        if not reasons:
            reasons.append("lower_scoring_temporal_path")
        output.append(
            RecoveryCandidate(
                **{
                    **candidate.__dict__,
                    "rejection_reason": "+".join(reasons),
                }
            )
        )
    return sorted(output, key=lambda item: (item.frame_index, item.rank))


def _recovered_observations(
    *,
    path: list[RecoveryCandidate],
    first_track: TrackObservation,
    vx: float,
    vy: float,
    path_score: float,
    terminal: float,
    config: ReleasePointConfig,
) -> list[RecoveredObservation]:
    recovered = []
    temporal_support = len(path)
    candidate_by_id = {item.candidate_id: item for item in path}
    for candidate in path:
        gap = first_track.frame_index - candidate.frame_index
        uncertainty = config.pretrack_projection_base_gate_px + gap * config.pretrack_projection_gate_growth_px
        observation = BallObservation(
            frame_index=candidate.frame_index,
            timestamp_seconds=candidate.timestamp_seconds,
            candidate_id=candidate.candidate_id,
            x=candidate.x,
            y=candidate.y,
            normalized_x=candidate.normalized_x,
            normalized_y=candidate.normalized_y,
            confidence=candidate.detector_confidence,
            rank=candidate.rank,
            inside_pitch_corridor=candidate.inside_pitch_corridor,
        )
        track_point = TrackObservation(
            frame_index=candidate.frame_index,
            timestamp_seconds=candidate.timestamp_seconds,
            x=candidate.x,
            y=candidate.y,
            normalized_x=candidate.normalized_x,
            normalized_y=candidate.normalized_y,
            confidence=max(0.0, min(1.0, path_score)),
            provenance=OBSERVED_RELEASE_RECOVERED,
            candidate_id=candidate.candidate_id,
            uncertainty=uncertainty,
            vx=vx,
            vy=vy,
            inside_pitch_corridor=candidate.inside_pitch_corridor,
        )
        recovered.append(
            RecoveredObservation(
                frame_index=candidate.frame_index,
                observation=observation,
                track_point=track_point,
                association_score=max(0.0, min(1.0, path_score)),
                temporal_support_count=temporal_support,
                backward_compatibility=candidate.backward_compatibility,
                forward_compatibility=max(candidate.forward_compatibility, terminal),
                pose_support=candidate.pose_support,
                recovery_reason="release_region_temporal_path_connected_to_primary_free_flight",
            )
        )
    return sorted(recovered, key=lambda item: item.frame_index)


def _reconstructed_gap(
    *,
    projections: list[RecoveryProjection],
    reason: str,
) -> list[ReconstructedPoint]:
    return [
        ReconstructedPoint(
            frame_index=item.frame_index,
            x=item.expected_x,
            y=item.expected_y,
            uncertainty=item.gate_radius_px,
            reason=reason,
        )
        for item in projections
    ]


def _path_uncertainty(
    path: list[RecoveryCandidate],
    first_track: TrackObservation,
    config: ReleasePointConfig,
) -> float | None:
    if not path:
        return None
    gap = first_track.frame_index - path[0].frame_index
    return config.pretrack_projection_base_gate_px + gap * config.pretrack_projection_gate_growth_px


def _no_chain_reason(
    *,
    temporal_support: int,
    terminal: float,
    path_motion: float,
    best_total: float,
    direction_quality: float,
) -> str:
    if temporal_support < 2:
        return "insufficient_multi_frame_temporal_support"
    if terminal < 0.45:
        return "does_not_connect_to_primary_free_flight"
    if path_motion < 0.35:
        return "static_or_low_motion_candidate_path"
    if direction_quality < 0.55:
        return "primary_free_flight_direction_unstable"
    if best_total < 0.50:
        return "association_score_below_recovery_threshold"
    return "no_credible_release_region_chain"


def _track_point_to_document(point: TrackObservation) -> dict[str, Any]:
    return {
        "frame_index": point.frame_index,
        "timestamp_seconds": point.timestamp_seconds,
        "source": "release_region_observation_recovery",
        "provenance": point.provenance,
        "candidate_id": point.candidate_id,
        "x": point.x,
        "y": point.y,
        "normalized_x": point.normalized_x,
        "normalized_y": point.normalized_y,
        "confidence": point.confidence,
        "uncertainty": point.uncertainty,
        "vx": point.vx,
        "vy": point.vy,
        "prediction_error": None,
        "inside_pitch_corridor": point.inside_pitch_corridor,
    }
