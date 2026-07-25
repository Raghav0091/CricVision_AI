"""Inspectable Release Point V1 feature extraction.

This module intentionally keeps thresholds in ``ReleasePointConfig``. The first
version is heuristic, but every number is visible and can be replaced by learned
calibration later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import statistics
from typing import Any


@dataclass(frozen=True)
class ReleasePointConfig:
    schema_version: str = "1.3"
    search_back_frames: int = 8
    search_forward_frames: int = 6
    pretrack_search_back_frames: int = 14
    pretrack_projection_base_gate_px: float = 28.0
    pretrack_projection_gate_growth_px: float = 8.0
    pretrack_min_candidate_score: float = 0.46
    bowling_arm_confidence_threshold: float = 0.25
    track_fit_points: int = 5
    minimum_track_points: int = 3
    minimum_free_flight_points: int = 3
    max_track_gap_frames: int = 3
    pose_confidence_threshold: float = 0.35
    wrist_confidence_threshold: float = 0.35
    close_ball_wrist_normalized: float = 0.55
    close_ball_wrist_px_fallback: float = 55.0
    separation_growth_normalized: float = 0.18
    persistent_separation_frames: int = 3
    forward_direction_cosine_min: float = 0.78
    backward_fit_good_px: float = 18.0
    backward_fit_weak_px: float = 45.0
    min_detector_confidence: float = 0.15
    low_confidence_threshold: float = 0.38
    unresolved_confidence_threshold: float = 0.22


@dataclass(frozen=True)
class BallObservation:
    frame_index: int
    timestamp_seconds: float
    candidate_id: str
    x: float
    y: float
    normalized_x: float
    normalized_y: float
    confidence: float
    rank: int
    inside_pitch_corridor: bool | None = None


@dataclass(frozen=True)
class TrackObservation:
    frame_index: int
    timestamp_seconds: float
    x: float
    y: float
    normalized_x: float
    normalized_y: float
    confidence: float
    provenance: str
    candidate_id: str | None = None
    uncertainty: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    inside_pitch_corridor: bool | None = None


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class PoseFrame:
    frame_index: int
    timestamp_seconds: float
    person_id: str | None
    confidence: float
    keypoints: dict[str, Keypoint]


@dataclass(frozen=True)
class BowlerPoseSequence:
    bowler_id: str | None
    selection_confidence: float
    poses_by_frame: dict[int, PoseFrame]
    quality_flags: list[str] = field(default_factory=list)
    provider: dict[str, str | None] = field(default_factory=dict)
    bowling_arm: str | None = None
    arm_confidence: float | None = None
    arm_ambiguous: bool = False


@dataclass(frozen=True)
class ReleaseCandidate:
    frame_index: int
    source: str
    ball: BallObservation | None = None
    track_point: TrackObservation | None = None


@dataclass(frozen=True)
class ReleaseFeatures:
    frame_index: int
    ball_candidate_id: str | None
    ball_x: float | None
    ball_y: float | None
    detector_confidence: float
    candidate_rank: int | None
    track_confidence: float
    tracker_provenance: str | None
    pose_confidence: float | None
    wrist_keypoint_name: str | None
    wrist_x: float | None
    wrist_y: float | None
    wrist_confidence: float | None
    ball_wrist_distance_px: float | None
    normalized_ball_wrist_distance: float | None
    separation_velocity: float | None
    separation_persistence_frames: int
    wrist_velocity: float | None
    wrist_acceleration_proxy: float | None
    arm_angle_degrees: float | None
    bowling_arm_extension_proxy: float | None
    early_trajectory_direction_consistency: float
    backward_trajectory_fit_error_px: float | None
    forward_free_flight_points: int
    forward_free_flight_confirmation: float
    scene_roi_consistency: float | None
    pose_keypoint_confidence: float | None


def parse_detection_observations(document: dict[str, Any]) -> dict[int, list[BallObservation]]:
    observations: dict[int, list[BallObservation]] = {}
    for frame in document.get("frames", []):
        frame_index = int(frame.get("frame_index", 0))
        timestamp = float(frame.get("timestamp_seconds", 0.0))
        ranked = sorted(
            frame.get("detections", []),
            key=lambda item: float(item.get("confidence", 0.0)),
            reverse=True,
        )
        frame_observations: list[BallObservation] = []
        for rank, candidate in enumerate(ranked, start=1):
            center = candidate.get("center", {}) or {}
            normalized = candidate.get("center_normalized", {}) or {}
            frame_observations.append(
                BallObservation(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp,
                    candidate_id=str(candidate.get("candidate_id", "")),
                    x=float(center.get("x", 0.0)),
                    y=float(center.get("y", 0.0)),
                    normalized_x=float(normalized.get("x", 0.0)),
                    normalized_y=float(normalized.get("y", 0.0)),
                    confidence=float(candidate.get("confidence", 0.0)),
                    rank=rank,
                    inside_pitch_corridor=candidate.get("inside_pitch_corridor"),
                )
            )
        observations[frame_index] = frame_observations
    return observations


def parse_track_observations(document: dict[str, Any]) -> list[TrackObservation]:
    points = document.get("primary_track", []) or []
    return [
        TrackObservation(
            frame_index=int(point.get("frame_index", 0)),
            timestamp_seconds=float(point.get("timestamp_seconds", 0.0)),
            x=float(point.get("x", 0.0)),
            y=float(point.get("y", 0.0)),
            normalized_x=float(point.get("normalized_x", 0.0)),
            normalized_y=float(point.get("normalized_y", 0.0)),
            confidence=float(point.get("confidence", 0.0)),
            provenance=str(point.get("provenance", "")),
            candidate_id=point.get("candidate_id"),
            uncertainty=float(point.get("uncertainty", 0.0)),
            vx=float(point.get("vx", 0.0)),
            vy=float(point.get("vy", 0.0)),
            inside_pitch_corridor=point.get("inside_pitch_corridor"),
        )
        for point in points
    ]


def parse_bowler_pose_sequence(data: dict[str, Any] | None) -> BowlerPoseSequence | None:
    if not data:
        return None
    raw_poses = data.get("poses_by_frame", {}) or {}
    poses_by_frame: dict[int, PoseFrame] = {}
    for key, pose in raw_poses.items():
        frame_index = int(pose.get("frame_index", key))
        raw_keypoints = pose.get("keypoints", {}) or {}
        keypoints = {
            name: Keypoint(
                x=float(value.get("x", 0.0)),
                y=float(value.get("y", 0.0)),
                confidence=float(value.get("confidence", 0.0)),
            )
            for name, value in raw_keypoints.items()
        }
        poses_by_frame[frame_index] = PoseFrame(
            frame_index=frame_index,
            timestamp_seconds=float(pose.get("timestamp_seconds", 0.0)),
            person_id=pose.get("person_id"),
            confidence=float(pose.get("confidence", 0.0)),
            keypoints=keypoints,
        )
    arm = data.get("bowling_arm") or {}
    if not isinstance(arm, dict):
        arm = {}
    arm_name = arm.get("bowling_arm")
    arm_confidence = arm.get("confidence")
    try:
        parsed_arm_confidence = (
            None if arm_confidence is None else float(arm_confidence)
        )
    except (TypeError, ValueError):
        parsed_arm_confidence = None
    return BowlerPoseSequence(
        bowler_id=data.get("bowler_id"),
        selection_confidence=float(data.get("selection_confidence", 0.0)),
        poses_by_frame=poses_by_frame,
        quality_flags=list(data.get("quality_flags", []) or []),
        provider=dict(data.get("provider", {}) or {}),
        bowling_arm=str(arm_name) if arm_name in {"left", "right"} else None,
        arm_confidence=parsed_arm_confidence,
        arm_ambiguous=(
            "bowling_arm_ambiguous" in set(data.get("quality_flags", []) or [])
            or any(
                flag == "bowling_arm_ambiguous"
                for flag in arm.get("quality_flags", []) or []
            )
        ),
    )


def extract_release_features(
    candidate: ReleaseCandidate,
    *,
    detections_by_frame: dict[int, list[BallObservation]],
    primary_track: list[TrackObservation],
    pose_sequence: BowlerPoseSequence | None,
    config: ReleasePointConfig,
    wrist_keypoint_name: str | None = None,
) -> ReleaseFeatures:
    track_point = candidate.track_point or nearest_track_point(candidate.frame_index, primary_track)
    ball = candidate.ball or best_ball_for_candidate(
        candidate.frame_index,
        detections_by_frame,
        track_point,
    )
    pose = nearest_pose(candidate.frame_index, pose_sequence)
    wrist_name, wrist = (
        selected_wrist(pose, wrist_keypoint_name)
        if pose
        else (None, None)
    )
    scale = body_scale_px(pose)
    distance_px = (
        math.hypot(ball.x - wrist.x, ball.y - wrist.y)
        if ball is not None and wrist is not None
        else None
    )
    normalized_distance = (
        distance_px / max(1.0, scale)
        if distance_px is not None and scale is not None
        else None
    )
    persistence, separation_velocity = separation_evidence(
        candidate.frame_index,
        ball,
        wrist,
        detections_by_frame,
        pose_sequence,
        config,
        wrist_keypoint_name=wrist_name,
    )
    wrist_velocity = keypoint_velocity(
        pose_sequence,
        candidate.frame_index,
        wrist_name,
        step=1,
    )
    previous_velocity = keypoint_velocity(
        pose_sequence,
        candidate.frame_index - 1,
        wrist_name,
        step=1,
    )
    acceleration = (
        abs(wrist_velocity - previous_velocity)
        if wrist_velocity is not None and previous_velocity is not None
        else None
    )
    arm_angle, extension = arm_geometry(pose, wrist_name)
    direction_consistency = trajectory_direction_consistency(primary_track, config)
    fit_error = backward_fit_error(candidate.frame_index, ball, primary_track, config)
    forward_points, forward_confirmation = forward_free_flight(
        candidate.frame_index,
        primary_track,
        config,
    )
    scene_roi = scene_roi_consistency(ball, track_point)
    pose_keypoint_confidence = keypoint_confidence(pose, wrist_name)
    return ReleaseFeatures(
        frame_index=candidate.frame_index,
        ball_candidate_id=None if ball is None else ball.candidate_id,
        ball_x=None if ball is None else ball.x,
        ball_y=None if ball is None else ball.y,
        detector_confidence=0.0 if ball is None else ball.confidence,
        candidate_rank=None if ball is None else ball.rank,
        track_confidence=0.0 if track_point is None else track_point.confidence,
        tracker_provenance=None if track_point is None else track_point.provenance,
        pose_confidence=None if pose is None else pose.confidence,
        wrist_keypoint_name=wrist_name,
        wrist_x=None if wrist is None else wrist.x,
        wrist_y=None if wrist is None else wrist.y,
        wrist_confidence=None if wrist is None else wrist.confidence,
        ball_wrist_distance_px=distance_px,
        normalized_ball_wrist_distance=normalized_distance,
        separation_velocity=separation_velocity,
        separation_persistence_frames=persistence,
        wrist_velocity=wrist_velocity,
        wrist_acceleration_proxy=acceleration,
        arm_angle_degrees=arm_angle,
        bowling_arm_extension_proxy=extension,
        early_trajectory_direction_consistency=direction_consistency,
        backward_trajectory_fit_error_px=fit_error,
        forward_free_flight_points=forward_points,
        forward_free_flight_confirmation=forward_confirmation,
        scene_roi_consistency=scene_roi,
        pose_keypoint_confidence=pose_keypoint_confidence,
    )


def nearest_track_point(
    frame_index: int,
    primary_track: list[TrackObservation],
) -> TrackObservation | None:
    if not primary_track:
        return None
    exact = [point for point in primary_track if point.frame_index == frame_index]
    if exact:
        return exact[0]
    return min(primary_track, key=lambda point: abs(point.frame_index - frame_index))


def best_ball_for_candidate(
    frame_index: int,
    detections_by_frame: dict[int, list[BallObservation]],
    track_point: TrackObservation | None,
) -> BallObservation | None:
    candidates = detections_by_frame.get(frame_index, [])
    if not candidates:
        return None
    if track_point is None:
        return candidates[0]
    return min(
        candidates,
        key=lambda item: (
            math.hypot(item.x - track_point.x, item.y - track_point.y),
            item.rank,
        ),
    )


def nearest_pose(
    frame_index: int,
    pose_sequence: BowlerPoseSequence | None,
) -> PoseFrame | None:
    if pose_sequence is None or not pose_sequence.poses_by_frame:
        return None
    if frame_index in pose_sequence.poses_by_frame:
        return pose_sequence.poses_by_frame[frame_index]
    nearest_frame = min(
        pose_sequence.poses_by_frame,
        key=lambda pose_frame: abs(pose_frame - frame_index),
    )
    return pose_sequence.poses_by_frame[nearest_frame]


def best_wrist(pose: PoseFrame) -> tuple[str | None, Keypoint | None]:
    wrists = [
        (name, keypoint)
        for name, keypoint in pose.keypoints.items()
        if name.endswith("_wrist")
    ]
    if not wrists:
        return None, None
    return max(wrists, key=lambda item: item[1].confidence)


def selected_wrist(
    pose: PoseFrame,
    wrist_keypoint_name: str | None,
) -> tuple[str | None, Keypoint | None]:
    if wrist_keypoint_name:
        wrist = pose.keypoints.get(wrist_keypoint_name)
        if wrist is not None:
            return wrist_keypoint_name, wrist
    return best_wrist(pose)


def body_scale_px(pose: PoseFrame | None) -> float | None:
    if pose is None:
        return None
    left_shoulder = pose.keypoints.get("left_shoulder")
    right_shoulder = pose.keypoints.get("right_shoulder")
    left_hip = pose.keypoints.get("left_hip")
    right_hip = pose.keypoints.get("right_hip")
    distances = []
    if left_shoulder and right_shoulder:
        distances.append(_point_distance(left_shoulder, right_shoulder))
    if left_shoulder and left_hip:
        distances.append(_point_distance(left_shoulder, left_hip))
    if right_shoulder and right_hip:
        distances.append(_point_distance(right_shoulder, right_hip))
    usable = [distance for distance in distances if distance > 1.0]
    return statistics.median(usable) if usable else None


def separation_evidence(
    frame_index: int,
    ball: BallObservation | None,
    wrist: Keypoint | None,
    detections_by_frame: dict[int, list[BallObservation]],
    pose_sequence: BowlerPoseSequence | None,
    config: ReleasePointConfig,
    wrist_keypoint_name: str | None = None,
) -> tuple[int, float | None]:
    if ball is None or wrist is None:
        return 0, None
    first_distance = math.hypot(ball.x - wrist.x, ball.y - wrist.y)
    persistent = 0
    velocities: list[float] = []
    for offset in range(1, config.search_forward_frames + 1):
        next_frame = frame_index + offset
        next_ball = (detections_by_frame.get(next_frame) or [None])[0]
        next_pose = nearest_pose(next_frame, pose_sequence)
        _, next_wrist = (
            selected_wrist(next_pose, wrist_keypoint_name)
            if next_pose
            else (None, None)
        )
        if next_ball is None or next_wrist is None:
            continue
        distance = math.hypot(next_ball.x - next_wrist.x, next_ball.y - next_wrist.y)
        velocities.append(distance - first_distance)
        if distance > first_distance + config.separation_growth_normalized * max(
            1.0,
            body_scale_px(next_pose) or config.close_ball_wrist_px_fallback,
        ):
            persistent += 1
    velocity = statistics.fmean(velocities) if velocities else None
    return persistent, velocity


def keypoint_velocity(
    pose_sequence: BowlerPoseSequence | None,
    frame_index: int,
    keypoint_name: str | None,
    *,
    step: int,
) -> float | None:
    if pose_sequence is None or keypoint_name is None:
        return None
    current = pose_sequence.poses_by_frame.get(frame_index)
    previous = pose_sequence.poses_by_frame.get(frame_index - step)
    if current is None or previous is None:
        return None
    first = previous.keypoints.get(keypoint_name)
    second = current.keypoints.get(keypoint_name)
    if first is None or second is None:
        return None
    return _point_distance(first, second) / max(1, step)


def arm_geometry(
    pose: PoseFrame | None,
    wrist_name: str | None,
) -> tuple[float | None, float | None]:
    if pose is None or wrist_name is None:
        return None, None
    side = wrist_name.removesuffix("_wrist")
    shoulder = pose.keypoints.get(f"{side}_shoulder")
    elbow = pose.keypoints.get(f"{side}_elbow")
    wrist = pose.keypoints.get(wrist_name)
    if shoulder is None or elbow is None or wrist is None:
        return None, None
    upper = _vector(elbow, shoulder)
    lower = _vector(elbow, wrist)
    angle = _angle_degrees(upper, lower)
    shoulder_wrist = _point_distance(shoulder, wrist)
    segment_sum = _point_distance(shoulder, elbow) + _point_distance(elbow, wrist)
    extension = shoulder_wrist / segment_sum if segment_sum > 1.0 else None
    return angle, extension


def trajectory_direction_consistency(
    primary_track: list[TrackObservation],
    config: ReleasePointConfig,
) -> float:
    points = primary_track[: max(0, config.track_fit_points)]
    if len(points) < 3:
        return 0.0
    vectors = [
        (second.x - first.x, second.y - first.y)
        for first, second in zip(points, points[1:])
        if second.frame_index > first.frame_index
    ]
    if len(vectors) < 2:
        return 0.0
    scores = [
        (_cosine(first, second) + 1.0) / 2.0
        for first, second in zip(vectors, vectors[1:])
    ]
    return _clamp(statistics.fmean(scores))


def backward_fit_error(
    frame_index: int,
    ball: BallObservation | None,
    primary_track: list[TrackObservation],
    config: ReleasePointConfig,
) -> float | None:
    if ball is None:
        return None
    points = primary_track[: max(0, config.track_fit_points)]
    if len(points) < 2:
        return None
    first, second = points[0], points[1]
    delta_frame = second.frame_index - first.frame_index
    if delta_frame == 0:
        return None
    vx = (second.x - first.x) / delta_frame
    vy = (second.y - first.y) / delta_frame
    predicted_x = first.x - vx * (first.frame_index - frame_index)
    predicted_y = first.y - vy * (first.frame_index - frame_index)
    return math.hypot(ball.x - predicted_x, ball.y - predicted_y)


def forward_free_flight(
    frame_index: int,
    primary_track: list[TrackObservation],
    config: ReleasePointConfig,
) -> tuple[int, float]:
    after = [
        point
        for point in primary_track
        if frame_index < point.frame_index <= frame_index + config.search_forward_frames
    ]
    if len(after) < config.minimum_free_flight_points:
        return len(after), 0.0
    observed_or_recovered = [
        point
        for point in after
        if point.provenance
        in {
            "OBSERVED",
            "TRACKER_RECOVERED",
            "PHYSICS_RECONSTRUCTED",
            "PRETRACK_RECOVERED",
            "OBSERVED_RELEASE_RECOVERED",
        }
    ]
    direction = trajectory_direction_consistency(after, config)
    density = min(1.0, len(observed_or_recovered) / config.minimum_free_flight_points)
    return len(after), _clamp(0.55 * direction + 0.45 * density)


def scene_roi_consistency(
    ball: BallObservation | None,
    track_point: TrackObservation | None,
) -> float | None:
    values = [
        value
        for value in (
            None if ball is None else ball.inside_pitch_corridor,
            None if track_point is None else track_point.inside_pitch_corridor,
        )
        if value is not None
    ]
    if not values:
        return None
    return sum(1.0 for value in values if value) / len(values)


def keypoint_confidence(
    pose: PoseFrame | None,
    wrist_name: str | None,
) -> float | None:
    if pose is None:
        return None
    names = [name for name in pose.keypoints if name.endswith(("_shoulder", "_elbow", "_wrist"))]
    if wrist_name and wrist_name not in names:
        names.append(wrist_name)
    values = [pose.keypoints[name].confidence for name in names if name in pose.keypoints]
    return statistics.fmean(values) if values else pose.confidence


def feature_dict(features: ReleaseFeatures) -> dict[str, Any]:
    return {
        "ball_candidate_id": features.ball_candidate_id,
        "ball_wrist_distance_px": _round_optional(features.ball_wrist_distance_px),
        "normalized_ball_wrist_distance": _round_optional(
            features.normalized_ball_wrist_distance
        ),
        "separation_velocity": _round_optional(features.separation_velocity),
        "separation_persistence_frames": features.separation_persistence_frames,
        "wrist_velocity": _round_optional(features.wrist_velocity),
        "wrist_acceleration_proxy": _round_optional(features.wrist_acceleration_proxy),
        "arm_angle_degrees": _round_optional(features.arm_angle_degrees),
        "bowling_arm_extension_proxy": _round_optional(
            features.bowling_arm_extension_proxy
        ),
        "early_trajectory_direction_consistency": round(
            features.early_trajectory_direction_consistency,
            6,
        ),
        "backward_trajectory_fit_error_px": _round_optional(
            features.backward_trajectory_fit_error_px
        ),
        "forward_free_flight_points": features.forward_free_flight_points,
        "forward_free_flight_confirmation": round(
            features.forward_free_flight_confirmation,
            6,
        ),
        "detector_confidence": round(features.detector_confidence, 6),
        "candidate_rank": features.candidate_rank,
        "track_confidence": round(features.track_confidence, 6),
        "tracker_provenance": features.tracker_provenance,
        "scene_roi_consistency": _round_optional(features.scene_roi_consistency),
        "pose_confidence": _round_optional(features.pose_confidence),
        "pose_keypoint_confidence": _round_optional(features.pose_keypoint_confidence),
        "wrist_confidence": _round_optional(features.wrist_confidence),
    }


def _point_distance(first: Keypoint, second: Keypoint) -> float:
    return math.hypot(second.x - first.x, second.y - first.y)


def _vector(origin: Keypoint, target: Keypoint) -> tuple[float, float]:
    return target.x - origin.x, target.y - origin.y


def _angle_degrees(first: tuple[float, float], second: tuple[float, float]) -> float:
    cosine = _cosine(first, second)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _cosine(first: tuple[float, float], second: tuple[float, float]) -> float:
    denominator = math.hypot(*first) * math.hypot(*second)
    if denominator <= 0.000001:
        return 0.0
    return (first[0] * second[0] + first[1] * second[1]) / denominator


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _round_optional(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)
