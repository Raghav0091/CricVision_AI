"""Release Point V1 candidate generation and evidence fusion."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any

from .features import (
    BowlerPoseSequence,
    ReleaseCandidate,
    ReleaseFeatures,
    ReleasePointConfig,
    TrackObservation,
    extract_release_features,
    feature_dict,
    parse_bowler_pose_sequence,
    parse_detection_observations,
    parse_track_observations,
)


METHOD = "release_point_v1_heuristic_fusion"


@dataclass(frozen=True)
class CandidateScore:
    frame_index: int
    score: float
    method: str
    release_type: str
    source: str
    observed: bool
    features: ReleaseFeatures
    score_components: dict[str, float]
    quality_flags: list[str]


@dataclass(frozen=True)
class ReleaseEstimate:
    status: str
    result: dict[str, Any]
    candidate_scores: list[CandidateScore]
    quality_summary: dict[str, Any]
    message: str


class ReleaseEstimator:
    def __init__(self, config: ReleasePointConfig | None = None) -> None:
        self.config = config or ReleasePointConfig()

    def estimate(
        self,
        *,
        analysis_id: str,
        fps: float,
        detections_document: dict[str, Any],
        tracking_document: dict[str, Any],
        bowler_pose_sequence: dict[str, Any] | BowlerPoseSequence | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> ReleaseEstimate:
        detections_by_frame = parse_detection_observations(detections_document)
        primary_track = parse_track_observations(tracking_document)
        pose_sequence = (
            bowler_pose_sequence
            if isinstance(bowler_pose_sequence, BowlerPoseSequence)
            else parse_bowler_pose_sequence(bowler_pose_sequence)
        )
        provenance = provenance or {}
        pose_evidence_real = _pose_evidence_is_real(provenance, pose_sequence)
        if not pose_evidence_real:
            pose_sequence = None

        if len(primary_track) < self.config.minimum_track_points:
            return self._unresolved(
                analysis_id,
                fps,
                "insufficient_primary_track",
                "Release Point V1 could not estimate release without a reliable primary track.",
                provenance,
            )

        candidates = self._generate_candidates(
            detections_by_frame=detections_by_frame,
            primary_track=primary_track,
            pose_sequence=pose_sequence,
        )
        scores = [
            self._score_candidate(
                candidate,
                detections_by_frame=detections_by_frame,
                primary_track=primary_track,
                pose_sequence=pose_sequence,
            )
            for candidate in candidates
        ]
        scores = sorted(scores, key=lambda item: item.score, reverse=True)
        if not scores or scores[0].score < self.config.unresolved_confidence_threshold:
            return self._unresolved(
                analysis_id,
                fps,
                "insufficient_release_evidence",
                "Release Point V1 could not find enough consistent release evidence.",
                provenance,
                scores,
            )

        best = scores[0]
        result = self._result_from_score(
            analysis_id=analysis_id,
            fps=fps,
            score=best,
            all_scores=scores,
            provenance=provenance,
        )
        quality_summary = self._quality_summary(scores, pose_sequence)
        return ReleaseEstimate(
            status=result["status"],
            result=result,
            candidate_scores=scores,
            quality_summary=quality_summary,
            message=(
                "Release Point V1 completed."
                if result["status"] == "ready"
                else "Release Point V1 completed with weak evidence."
            ),
        )

    def _generate_candidates(
        self,
        *,
        detections_by_frame: dict[int, list[Any]],
        primary_track: list[TrackObservation],
        pose_sequence: BowlerPoseSequence | None,
    ) -> list[ReleaseCandidate]:
        start_frame = primary_track[0].frame_index
        candidate_sources: dict[int, set[str]] = {}

        def add(frame_index: int, source: str) -> None:
            if frame_index < 0:
                return
            candidate_sources.setdefault(frame_index, set()).add(source)

        for frame in range(
            start_frame - self.config.search_back_frames,
            start_frame + self.config.search_forward_frames + 1,
        ):
            add(frame, "track_start_window")

        earliest = self._earliest_reliable_segment(primary_track)
        if earliest is not None:
            for frame in range(
                earliest - self.config.search_back_frames,
                earliest + 2,
            ):
                add(frame, "earliest_reliable_moving_segment")

        if pose_sequence is not None:
            for frame in self._pose_timing_frames(pose_sequence):
                if start_frame - self.config.search_back_frames <= frame <= start_frame + self.config.search_forward_frames:
                    add(frame, "wrist_arm_motion")
            for frame, detections in detections_by_frame.items():
                if not (start_frame - self.config.search_back_frames <= frame <= start_frame + self.config.search_forward_frames):
                    continue
                pose = pose_sequence.poses_by_frame.get(frame)
                if pose is None:
                    continue
                wrists = [kp for name, kp in pose.keypoints.items() if name.endswith("_wrist")]
                if not wrists:
                    continue
                for ball in detections:
                    if any(math.hypot(ball.x - wrist.x, ball.y - wrist.y) <= self.config.close_ball_wrist_px_fallback for wrist in wrists):
                        add(frame, "ball_wrist_proximity")
                        break

        if len(primary_track) >= 2:
            add(max(0, start_frame - 1), "backward_trajectory_intersection")
            add(max(0, start_frame - 2), "backward_trajectory_intersection")

        return [
            ReleaseCandidate(frame_index=frame, source="+".join(sorted(sources)))
            for frame, sources in sorted(candidate_sources.items())
        ]

    def _earliest_reliable_segment(
        self,
        primary_track: list[TrackObservation],
    ) -> int | None:
        for index in range(0, len(primary_track) - self.config.minimum_free_flight_points + 1):
            window = primary_track[index : index + self.config.minimum_free_flight_points]
            gaps = [
                second.frame_index - first.frame_index
                for first, second in zip(window, window[1:])
            ]
            if all(0 < gap <= self.config.max_track_gap_frames for gap in gaps):
                return window[0].frame_index
        return primary_track[0].frame_index if primary_track else None

    def _pose_timing_frames(self, pose_sequence: BowlerPoseSequence) -> list[int]:
        velocities: list[tuple[int, float]] = []
        frames = sorted(pose_sequence.poses_by_frame)
        for first_frame, second_frame in zip(frames, frames[1:]):
            first = pose_sequence.poses_by_frame[first_frame]
            second = pose_sequence.poses_by_frame[second_frame]
            values = []
            for name, point in second.keypoints.items():
                if not name.endswith("_wrist") or name not in first.keypoints:
                    continue
                previous = first.keypoints[name]
                values.append(math.hypot(point.x - previous.x, point.y - previous.y))
            if values:
                velocities.append((second_frame, max(values)))
        if not velocities:
            return []
        threshold = statistics.quantiles([item[1] for item in velocities], n=4)[2] if len(velocities) >= 4 else max(item[1] for item in velocities)
        return [frame for frame, velocity in velocities if velocity >= threshold]

    def _score_candidate(
        self,
        candidate: ReleaseCandidate,
        *,
        detections_by_frame: dict[int, list[Any]],
        primary_track: list[TrackObservation],
        pose_sequence: BowlerPoseSequence | None,
    ) -> CandidateScore:
        features = extract_release_features(
            candidate,
            detections_by_frame=detections_by_frame,
            primary_track=primary_track,
            pose_sequence=pose_sequence,
            config=self.config,
        )
        pose_usable = self._pose_usable(features, pose_sequence)
        close = self._close_hand_score(features)
        separation = min(
            1.0,
            features.separation_persistence_frames / max(1, self.config.persistent_separation_frames),
        )
        trajectory = self._trajectory_score(features)
        forward = features.forward_free_flight_confirmation
        detector = max(0.0, min(1.0, features.detector_confidence))
        rank = 1.0 if features.candidate_rank in {None, 1} else max(0.2, 1.0 / features.candidate_rank)
        track = max(0.0, min(1.0, features.track_confidence))
        roi = 0.5 if features.scene_roi_consistency is None else features.scene_roi_consistency
        wrist_motion = 0.0 if features.wrist_velocity is None else min(1.0, features.wrist_velocity / 35.0)
        arm_extension = 0.5 if features.bowling_arm_extension_proxy is None else features.bowling_arm_extension_proxy

        if pose_usable and close > 0.65 and separation >= 0.66:
            release_type = "OBSERVED_RELEASE"
            mode = "observed_pose_ball_separation"
            components = {
                "ball_wrist_proximity": close,
                "separation_persistence": separation,
                "forward_free_flight": forward,
                "detector_confidence": detector,
                "track_confidence": track,
                "pose_confidence": self._pose_score(features),
                "candidate_rank": rank,
                "scene_roi": roi,
            }
            score = (
                0.24 * close
                + 0.20 * separation
                + 0.18 * forward
                + 0.11 * detector
                + 0.11 * track
                + 0.10 * self._pose_score(features)
                + 0.04 * rank
                + 0.02 * roi
            )
        elif pose_usable:
            release_type = "INFERRED_RELEASE"
            mode = "trajectory_pose_inferred"
            components = {
                "trajectory_consistency": trajectory,
                "forward_free_flight": forward,
                "wrist_motion": wrist_motion,
                "arm_extension": arm_extension,
                "detector_confidence": detector,
                "track_confidence": track,
                "pose_confidence": self._pose_score(features),
                "candidate_rank": rank,
                "scene_roi": roi,
            }
            score = (
                0.25 * trajectory
                + 0.19 * forward
                + 0.14 * wrist_motion
                + 0.10 * arm_extension
                + 0.10 * detector
                + 0.10 * track
                + 0.08 * self._pose_score(features)
                + 0.03 * rank
                + 0.01 * roi
            )
        else:
            release_type = "INFERRED_RELEASE"
            mode = "fallback_trajectory_only"
            components = {
                "trajectory_consistency": trajectory,
                "forward_free_flight": forward,
                "detector_confidence": detector,
                "track_confidence": track,
                "candidate_rank": rank,
                "scene_roi": roi,
            }
            score = (
                0.34 * trajectory
                + 0.26 * forward
                + 0.14 * detector
                + 0.16 * track
                + 0.06 * rank
                + 0.04 * roi
            )
            score = min(score, 0.58)

        flags = self._quality_flags(features, pose_usable, mode, score)
        return CandidateScore(
            frame_index=candidate.frame_index,
            score=round(max(0.0, min(1.0, score)), 6),
            method=mode,
            release_type=release_type,
            source=candidate.source,
            observed=mode == "observed_pose_ball_separation",
            features=features,
            score_components={key: round(value, 6) for key, value in components.items()},
            quality_flags=flags,
        )

    def _close_hand_score(self, features: ReleaseFeatures) -> float:
        if features.normalized_ball_wrist_distance is not None:
            return max(
                0.0,
                1.0
                - features.normalized_ball_wrist_distance
                / self.config.close_ball_wrist_normalized,
            )
        if features.ball_wrist_distance_px is not None:
            return max(
                0.0,
                1.0 - features.ball_wrist_distance_px / self.config.close_ball_wrist_px_fallback,
            )
        return 0.0

    def _trajectory_score(self, features: ReleaseFeatures) -> float:
        direction = features.early_trajectory_direction_consistency
        if features.backward_trajectory_fit_error_px is None:
            fit = 0.45
        elif features.backward_trajectory_fit_error_px <= self.config.backward_fit_good_px:
            fit = 1.0
        elif features.backward_trajectory_fit_error_px >= self.config.backward_fit_weak_px:
            fit = 0.0
        else:
            span = self.config.backward_fit_weak_px - self.config.backward_fit_good_px
            fit = 1.0 - ((features.backward_trajectory_fit_error_px - self.config.backward_fit_good_px) / span)
        return max(0.0, min(1.0, 0.55 * direction + 0.45 * fit))

    def _pose_usable(
        self,
        features: ReleaseFeatures,
        pose_sequence: BowlerPoseSequence | None,
    ) -> bool:
        if pose_sequence is None:
            return False
        pose_confidence = features.pose_confidence or 0.0
        wrist_confidence = features.wrist_confidence or 0.0
        selection = pose_sequence.selection_confidence
        return (
            pose_confidence >= self.config.pose_confidence_threshold
            and wrist_confidence >= self.config.wrist_confidence_threshold
            and selection >= self.config.pose_confidence_threshold
        )

    def _pose_score(self, features: ReleaseFeatures) -> float:
        values = [
            value
            for value in (
                features.pose_confidence,
                features.wrist_confidence,
                features.pose_keypoint_confidence,
            )
            if value is not None
        ]
        return max(0.0, min(1.0, statistics.fmean(values))) if values else 0.0

    def _quality_flags(
        self,
        features: ReleaseFeatures,
        pose_usable: bool,
        mode: str,
        score: float,
    ) -> list[str]:
        flags: list[str] = []
        if not pose_usable:
            flags.append("pose_unavailable_or_unreliable")
        if mode == "fallback_trajectory_only":
            flags.append("trajectory_only_estimate")
        if features.detector_confidence < self.config.min_detector_confidence:
            flags.append("low_detector_confidence")
        if features.forward_free_flight_confirmation < 0.45:
            flags.append("weak_forward_free_flight_confirmation")
        if (
            features.backward_trajectory_fit_error_px is not None
            and features.backward_trajectory_fit_error_px > self.config.backward_fit_weak_px
        ):
            flags.append("bad_backward_trajectory_fit")
        if features.wrist_confidence is not None and features.wrist_confidence < self.config.wrist_confidence_threshold:
            flags.append("low_confidence_wrist")
        if features.scene_roi_consistency == 0.0:
            flags.append("outside_scene_roi")
        if score < self.config.low_confidence_threshold:
            flags.append("low_confidence_release")
        return flags

    def _result_from_score(
        self,
        *,
        analysis_id: str,
        fps: float,
        score: CandidateScore,
        all_scores: list[CandidateScore],
        provenance: dict[str, Any],
    ) -> dict[str, Any]:
        uncertainty = self._uncertainty(score, all_scores)
        status = "ready" if score.score >= self.config.low_confidence_threshold else "unresolved"
        features = score.features
        evidence = feature_dict(features)
        evidence.update(
            {
                "score_components": score.score_components,
                "candidate_source": score.source,
                "observed": score.observed,
                "bowler_person_id": provenance.get("bowler_id"),
            }
        )
        quality_flags = _unique_flags(
            score.quality_flags + list(provenance.get("quality_flags") or [])
        )
        return {
            "schema_version": self.config.schema_version,
            "analysis_id": analysis_id,
            "status": status,
            "release_frame": score.frame_index,
            "release_time_seconds": round(score.frame_index / fps, 6),
            "release_point_px": (
                None
                if features.ball_x is None or features.ball_y is None
                else {"x": round(features.ball_x, 6), "y": round(features.ball_y, 6)}
            ),
            "confidence": score.score,
            "frame_uncertainty": uncertainty,
            "method": METHOD,
            "evidence_mode": score.method,
            "release_type": score.release_type,
            "evidence": evidence,
            "quality_flags": quality_flags,
            "provenance": provenance,
        }

    def _uncertainty(
        self,
        score: CandidateScore,
        all_scores: list[CandidateScore],
    ) -> dict[str, int]:
        base = 1 if score.method == "observed_pose_ball_separation" else 2
        if score.method == "fallback_trajectory_only":
            base = 3
        if score.score < 0.5:
            base += 1
        near_scores = [
            item.frame_index
            for item in all_scores
            if item.score >= max(0.0, score.score - 0.08)
        ]
        start = min([score.frame_index - base, *near_scores])
        end = max([score.frame_index + base, *near_scores])
        return {"start": max(0, start), "end": max(0, end)}

    def _quality_summary(
        self,
        scores: list[CandidateScore],
        pose_sequence: BowlerPoseSequence | None,
    ) -> dict[str, Any]:
        best = scores[0] if scores else None
        flags = sorted({flag for score in scores[:5] for flag in score.quality_flags})
        return {
            "candidate_count": len(scores),
            "best_score": None if best is None else best.score,
            "best_frame": None if best is None else best.frame_index,
            "best_evidence_mode": None if best is None else best.method,
            "pose_available": pose_sequence is not None,
            "quality_flags": flags,
            "config": self.config.__dict__,
        }

    def _unresolved(
        self,
        analysis_id: str,
        fps: float,
        reason: str,
        message: str,
        provenance: dict[str, Any],
        scores: list[CandidateScore] | None = None,
    ) -> ReleaseEstimate:
        result = {
            "schema_version": self.config.schema_version,
            "analysis_id": analysis_id,
            "status": "unresolved",
            "release_frame": None,
            "release_time_seconds": None,
            "release_point_px": None,
            "confidence": 0.0,
            "frame_uncertainty": None,
            "method": METHOD,
            "evidence_mode": "unresolved",
            "release_type": "UNRESOLVED",
            "evidence": {"reason": reason},
            "quality_flags": _unique_flags(
                [reason] + list(provenance.get("quality_flags") or [])
            ),
            "provenance": provenance,
        }
        quality_flags = _unique_flags(
            [reason] + list(provenance.get("quality_flags") or [])
        )
        return ReleaseEstimate(
            status="unresolved",
            result=result,
            candidate_scores=scores or [],
            quality_summary={
                "candidate_count": 0 if scores is None else len(scores),
                "best_score": None if not scores else scores[0].score,
                "best_frame": None if not scores else scores[0].frame_index,
                "best_evidence_mode": None if not scores else scores[0].method,
                "pose_available": False,
                "quality_flags": quality_flags,
                "config": self.config.__dict__,
            },
            message=message,
        )


def _unique_flags(flags: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw_flag in flags:
        flag = str(raw_flag)
        if flag and flag not in seen:
            result.append(flag)
            seen.add(flag)
    return result


def _pose_evidence_is_real(
    provenance: dict[str, Any],
    pose_sequence: BowlerPoseSequence | None,
) -> bool:
    if pose_sequence is None:
        return False
    if provenance.get("pose_evidence_real") is not True:
        return False
    provider_name = str(provenance.get("pose_provider") or "").strip().lower()
    if provider_name in {"", "fake_pose", "fakeposeprovider"}:
        return False
    return True


def candidate_score_to_dict(score: CandidateScore) -> dict[str, Any]:
    return {
        "frame_index": score.frame_index,
        "score": score.score,
        "method": score.method,
        "release_type": score.release_type,
        "observed": score.observed,
        "source": score.source,
        "features": feature_dict(score.features),
        "score_components": score.score_components,
        "quality_flags": score.quality_flags,
    }
