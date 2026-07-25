"""Release Point V1 candidate generation and evidence fusion."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
from .pretrack_reconstruction import (
    PRETRACK_RECOVERED,
    PretrackHypothesis,
    reconstruct_pretrack_hypothesis,
)


METHOD = "release_point_v1_3_late_free_flight_bias_guard"


@dataclass(frozen=True)
class ReleaseHypothesis:
    frame_index: int
    candidate_type: str
    exact_observation: bool
    trajectory_compatibility: float
    hand_association: str
    hand_association_score: float | None
    separation_transition: str
    separation_score: float
    pose_quality: float | None
    bowling_arm_confidence: float | None
    forward_flight: float
    backward_fit_quality: float
    detector_evidence: float
    free_flight_onset_frame: int | None
    free_flight_age_frames: int | None
    established_free_flight: bool
    future_confirmation_strength: float
    event_localization_strength: float
    late_flight_penalty: float
    release_temporal_eligibility: str
    ambiguity_flags: list[str]
    eligible: bool
    arbitration_score: float
    reason_codes: list[str]
    source_score: CandidateScore

    def diagnostics(self) -> dict[str, Any]:
        return {
            "candidate_frame": self.frame_index,
            "candidate_type": self.candidate_type,
            "observed_recovered_reconstructed_status": self.candidate_type,
            "exact_observation": self.exact_observation,
            "trajectory_compatibility": round(self.trajectory_compatibility, 6),
            "hand_association_evidence": self.hand_association,
            "hand_association_score": None
            if self.hand_association_score is None
            else round(self.hand_association_score, 6),
            "separation_transition_evidence": self.separation_transition,
            "separation_score": round(self.separation_score, 6),
            "pose_quality": None if self.pose_quality is None else round(self.pose_quality, 6),
            "bowling_arm_confidence": self.bowling_arm_confidence,
            "forward_flight_evidence": round(self.forward_flight, 6),
            "backward_fit_quality": round(self.backward_fit_quality, 6),
            "detector_evidence": round(self.detector_evidence, 6),
            "free_flight_onset_frame": self.free_flight_onset_frame,
            "free_flight_age_frames": self.free_flight_age_frames,
            "established_free_flight": self.established_free_flight,
            "future_confirmation_strength": round(
                self.future_confirmation_strength,
                6,
            ),
            "event_localization_strength": round(
                self.event_localization_strength,
                6,
            ),
            "late_flight_penalty": round(self.late_flight_penalty, 6),
            "release_temporal_eligibility": self.release_temporal_eligibility,
            "ambiguity_flags": self.ambiguity_flags,
            "final_arbitration_eligibility": self.eligible,
            "arbitration_score": round(self.arbitration_score, 6),
            "arbitration_reason_codes": self.reason_codes,
            "release_feature_score": self.source_score.score,
            "release_feature_mode": self.source_score.method,
            "source": self.source_score.source,
            "features": feature_dict(self.source_score.features),
        }


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
    diagnostics: dict[str, Any] | None = None


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
        wrist_context = self._wrist_context(pose_sequence)

        if len(primary_track) < self.config.minimum_track_points:
            return self._unresolved(
                analysis_id,
                fps,
                "insufficient_primary_track",
                "Release Point V1 could not estimate release without a reliable primary track.",
                provenance,
            )

        pretrack_hypothesis = reconstruct_pretrack_hypothesis(
            detections_by_frame=detections_by_frame,
            primary_track=primary_track,
            pose_sequence=pose_sequence,
            wrist_keypoint_name=wrist_context["wrist_used"],
            config=self.config,
        )
        release_track = self._release_hypothesis_track(primary_track, pretrack_hypothesis)

        candidates = self._generate_candidates(
            detections_by_frame=detections_by_frame,
            primary_track=primary_track,
            pose_sequence=pose_sequence,
            pretrack_hypothesis=pretrack_hypothesis,
        )
        scores = [
            self._score_candidate(
                candidate,
                detections_by_frame=detections_by_frame,
                primary_track=release_track,
                pose_sequence=pose_sequence,
                wrist_context=wrist_context,
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

        best, arbitration = self._arbitrate_scores(
            scores,
            primary_track=primary_track,
            pretrack_hypothesis=pretrack_hypothesis,
            wrist_context=wrist_context,
        )
        if best is None:
            return self._unresolved(
                analysis_id,
                fps,
                "no_arbitration_eligible_release_hypothesis",
                "Release Point V1.3 could not find an arbitration-eligible release hypothesis.",
                provenance,
                scores,
            )
        result = self._result_from_score(
            analysis_id=analysis_id,
            fps=fps,
            score=best,
            all_scores=scores,
            provenance=provenance,
            pretrack_hypothesis=pretrack_hypothesis,
            wrist_context=wrist_context,
            arbitration=arbitration,
        )
        quality_summary = self._quality_summary(
            scores,
            pose_sequence,
            pretrack_hypothesis=pretrack_hypothesis,
            wrist_context=wrist_context,
            arbitration=arbitration,
        )
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
        pretrack_hypothesis: PretrackHypothesis | None = None,
    ) -> list[ReleaseCandidate]:
        start_frame = primary_track[0].frame_index
        candidate_sources: dict[int, set[str]] = {}
        candidate_by_frame: dict[int, ReleaseCandidate] = {}

        def add(
            frame_index: int,
            source: str,
            candidate: ReleaseCandidate | None = None,
        ) -> None:
            if frame_index < 0:
                return
            candidate_sources.setdefault(frame_index, set()).add(source)
            if candidate is not None:
                candidate_by_frame[frame_index] = candidate

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

        if pretrack_hypothesis is not None:
            for point in pretrack_hypothesis.recovered_points:
                if point.provenance != PRETRACK_RECOVERED or point.observation is None:
                    continue
                add(
                    point.frame_index,
                    "pretrack_recovered_candidate",
                    ReleaseCandidate(
                        frame_index=point.frame_index,
                        source="pretrack_recovered_candidate",
                        ball=point.observation,
                        track_point=point.track_point,
                    ),
                )

        return [
            (
                ReleaseCandidate(
                    frame_index=frame,
                    source="+".join(sorted(sources)),
                    ball=candidate_by_frame[frame].ball,
                    track_point=candidate_by_frame[frame].track_point,
                )
                if frame in candidate_by_frame
                else ReleaseCandidate(frame_index=frame, source="+".join(sorted(sources)))
            )
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
        wrist_context: dict[str, Any],
    ) -> CandidateScore:
        features = extract_release_features(
            candidate,
            detections_by_frame=detections_by_frame,
            primary_track=primary_track,
            pose_sequence=pose_sequence,
            config=self.config,
            wrist_keypoint_name=wrist_context["wrist_used"],
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

        score, confidence_flags = self._confidence_safety_adjustment(
            candidate,
            features,
            score,
            close=close,
            separation=separation,
            wrist_context=wrist_context,
        )
        flags = self._quality_flags(features, pose_usable, mode, score)
        flags.extend(confidence_flags)
        return CandidateScore(
            frame_index=candidate.frame_index,
            score=round(max(0.0, min(1.0, score)), 6),
            method=mode,
            release_type=release_type,
            source=candidate.source,
            observed=mode == "observed_pose_ball_separation",
            features=features,
            score_components={key: round(value, 6) for key, value in components.items()},
            quality_flags=_unique_flags(flags),
            diagnostics={
                "wrist_context": wrist_context,
                "candidate_provenance": features.tracker_provenance,
                "confidence_safety_flags": confidence_flags,
            },
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
        if pose_sequence.arm_ambiguous:
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

    def _confidence_safety_adjustment(
        self,
        candidate: ReleaseCandidate,
        features: ReleaseFeatures,
        score: float,
        *,
        close: float,
        separation: float,
        wrist_context: dict[str, Any],
    ) -> tuple[float, list[str]]:
        flags: list[str] = []
        adjusted = score
        if wrist_context.get("arm_ambiguous"):
            adjusted *= 0.88
            flags.append("ambiguous_bowling_arm_confidence_penalty")
        if (
            features.tracker_provenance == PRETRACK_RECOVERED
            and candidate.track_point is not None
            and candidate.track_point.uncertainty > 85.0
        ):
            adjusted *= 0.9
            flags.append("large_backward_extrapolation")
        if (
            features.ball_candidate_id is None
            or features.tracker_provenance is None
            or features.tracker_provenance == "RECONSTRUCTED"
        ):
            adjusted *= 0.82
            flags.append("no_observed_or_recovered_ball_near_release")
        if close < 0.15 and separation <= 0.0 and features.forward_free_flight_confirmation >= 0.8:
            adjusted *= 0.82
            flags.append("weak_hand_ball_transition_despite_free_flight")
        if features.separation_persistence_frames == 0 and features.separation_velocity is not None and features.separation_velocity <= 0:
            adjusted *= 0.92
            flags.append("no_increasing_hand_ball_separation")
        return max(0.0, min(1.0, adjusted)), flags

    def _arbitrate_scores(
        self,
        scores: list[CandidateScore],
        *,
        primary_track: list[TrackObservation],
        pretrack_hypothesis: PretrackHypothesis,
        wrist_context: dict[str, Any],
    ) -> tuple[CandidateScore | None, dict[str, Any]]:
        observed_frames = {point.frame_index for point in primary_track}
        recovered_frames = {
            point.frame_index
            for point in pretrack_hypothesis.recovered_points
            if point.provenance == PRETRACK_RECOVERED
        }
        free_flight_onset = self._free_flight_onset_frame(
            primary_track,
            pretrack_hypothesis,
        )
        hypotheses = [
            self._build_release_hypothesis(
                score,
                observed_frames=observed_frames,
                recovered_frames=recovered_frames,
                recovered_count=len(recovered_frames),
                wrist_context=wrist_context,
                free_flight_onset=free_flight_onset,
                first_primary_track_frame=primary_track[0].frame_index,
            )
            for score in scores
        ]
        hypotheses = self._suppress_weak_single_recovered_overrides(hypotheses)
        eligible = [item for item in hypotheses if item.eligible]
        if not eligible:
            return None, {
                "status": "unresolved",
                "reason": "no_eligible_hypothesis",
                "release_hypotheses": [item.diagnostics() for item in hypotheses],
            }
        best_hypothesis = max(
            eligible,
            key=lambda item: (item.arbitration_score, item.source_score.score, -item.frame_index),
        )
        disagreement_penalty = self._hypothesis_disagreement_penalty(
            best_hypothesis,
            eligible,
        )
        if disagreement_penalty:
            adjusted = max(0.0, best_hypothesis.source_score.score - disagreement_penalty)
            best_score = CandidateScore(
                frame_index=best_hypothesis.source_score.frame_index,
                score=round(adjusted, 6),
                method=best_hypothesis.source_score.method,
                release_type=best_hypothesis.source_score.release_type,
                source=best_hypothesis.source_score.source,
                observed=best_hypothesis.source_score.observed,
                features=best_hypothesis.source_score.features,
                score_components=best_hypothesis.source_score.score_components,
                quality_flags=_unique_flags(
                    [
                        *best_hypothesis.source_score.quality_flags,
                        "competing_hypotheses_disagree",
                        *self._selected_temporal_quality_flags(best_hypothesis),
                    ]
                ),
                diagnostics=best_hypothesis.source_score.diagnostics,
            )
        else:
            temporal_flags = self._selected_temporal_quality_flags(best_hypothesis)
            if temporal_flags:
                best_score = replace(
                    best_hypothesis.source_score,
                    quality_flags=_unique_flags(
                        [*best_hypothesis.source_score.quality_flags, *temporal_flags]
                    ),
                )
            else:
                best_score = best_hypothesis.source_score
        return best_score, {
            "status": "ready",
            "selected_frame": best_score.frame_index,
            "selected_candidate_type": best_hypothesis.candidate_type,
            "selected_arbitration_score": round(best_hypothesis.arbitration_score, 6),
            "confidence_disagreement_penalty": round(disagreement_penalty, 6),
            "selected_reason_codes": best_hypothesis.reason_codes,
            "free_flight_onset_frame": free_flight_onset,
            "selected_free_flight_age_frames": best_hypothesis.free_flight_age_frames,
            "selected_event_localization_strength": round(
                best_hypothesis.event_localization_strength,
                6,
            ),
            "selected_future_confirmation_strength": round(
                best_hypothesis.future_confirmation_strength,
                6,
            ),
            "selected_late_flight_penalty": round(
                best_hypothesis.late_flight_penalty,
                6,
            ),
            "selected_release_temporal_eligibility": (
                best_hypothesis.release_temporal_eligibility
            ),
            "release_hypotheses": [item.diagnostics() for item in hypotheses],
        }

    def _free_flight_onset_frame(
        self,
        primary_track: list[TrackObservation],
        pretrack_hypothesis: PretrackHypothesis,
    ) -> int | None:
        del pretrack_hypothesis
        ordered = sorted(primary_track, key=lambda point: point.frame_index)
        if not ordered:
            return None
        for index, point in enumerate(ordered):
            after = [
                later
                for later in ordered[index + 1 :]
                if later.frame_index - point.frame_index <= self.config.search_forward_frames
            ]
            if len(after) < max(2, self.config.minimum_free_flight_points - 1):
                continue
            segment = [point, *after[: self.config.minimum_free_flight_points]]
            direction = self._track_direction_consistency(segment)
            if direction >= 0.65:
                return point.frame_index
        return ordered[0].frame_index

    def _track_direction_consistency(self, points: list[TrackObservation]) -> float:
        if len(points) < 3:
            return 0.0
        vectors = []
        for first, second in zip(points, points[1:]):
            dx = second.x - first.x
            dy = second.y - first.y
            magnitude = math.hypot(dx, dy)
            if magnitude > 0:
                vectors.append((dx / magnitude, dy / magnitude))
        if len(vectors) < 2:
            return 0.0
        similarities = [
            max(-1.0, min(1.0, ax * bx + ay * by))
            for (ax, ay), (bx, by) in zip(vectors, vectors[1:])
        ]
        return max(0.0, min(1.0, statistics.fmean(similarities)))

    def _selected_temporal_quality_flags(
        self,
        hypothesis: ReleaseHypothesis,
    ) -> list[str]:
        flags: list[str] = []
        if hypothesis.late_flight_penalty > 0:
            flags.append("established_free_flight_bias_guard")
        if hypothesis.release_temporal_eligibility == "weak_localization":
            flags.append("release_localization_weak")
        if (
            hypothesis.free_flight_onset_frame is not None
            and hypothesis.frame_index < hypothesis.free_flight_onset_frame
        ):
            flags.append("release_precedes_first_reliable_observation")
            flags.append("observational_gap_near_release")
        return flags

    def _suppress_weak_single_recovered_overrides(
        self,
        hypotheses: list[ReleaseHypothesis],
    ) -> list[ReleaseHypothesis]:
        observed_floor = max(
            (
                item.arbitration_score
                for item in hypotheses
                if item.candidate_type == "OBSERVED" and item.eligible
            ),
            default=0.0,
        )
        if observed_floor < 0.5:
            return hypotheses
        adjusted: list[ReleaseHypothesis] = []
        for item in hypotheses:
            weak_single_recovered = (
                item.candidate_type == PRETRACK_RECOVERED
                and item.hand_association == "UNAVAILABLE"
                and item.separation_transition == "UNAVAILABLE"
                and "RECOVERED_SINGLE_FRAME" in item.reason_codes
                and (
                    item.free_flight_age_frames is None
                    or item.free_flight_age_frames >= -self.config.max_track_gap_frames
                )
            )
            if weak_single_recovered:
                adjusted.append(
                    replace(
                        item,
                        eligible=False,
                        reason_codes=_unique_flags(
                            [
                                *item.reason_codes,
                                "RECOVERED_REJECTED_BY_ARBITRATION",
                            ]
                        ),
                    )
                )
            else:
                adjusted.append(item)
        return adjusted

    def _build_release_hypothesis(
        self,
        score: CandidateScore,
        *,
        observed_frames: set[int],
        recovered_frames: set[int],
        recovered_count: int,
        wrist_context: dict[str, Any],
        free_flight_onset: int | None,
        first_primary_track_frame: int,
    ) -> ReleaseHypothesis:
        features = score.features
        candidate_type = self._candidate_type(score, observed_frames, recovered_frames)
        exact = candidate_type in {"OBSERVED", PRETRACK_RECOVERED}
        hand_label, hand_score = self._hand_association(features, wrist_context)
        separation_label, separation_score = self._separation_transition(features)
        trajectory = self._trajectory_score(features)
        backward_quality = self._backward_fit_quality(features)
        detector = max(0.0, min(1.0, features.detector_confidence))
        pose_quality = self._pose_score(features) if features.pose_confidence is not None else None
        free_flight_age = (
            None if free_flight_onset is None else score.frame_index - free_flight_onset
        )
        future_confirmation = features.forward_free_flight_confirmation
        event_localization = self._event_localization_strength(
            score=score,
            candidate_type=candidate_type,
            hand_label=hand_label,
            separation_label=separation_label,
            backward_quality=backward_quality,
            free_flight_age=free_flight_age,
        )
        late_penalty = self._late_flight_penalty(
            candidate_type=candidate_type,
            hand_label=hand_label,
            separation_label=separation_label,
            event_localization=event_localization,
            free_flight_age=free_flight_age,
        )
        temporal_eligibility = self._release_temporal_eligibility(
            candidate_type=candidate_type,
            hand_label=hand_label,
            separation_label=separation_label,
            event_localization=event_localization,
            late_penalty=late_penalty,
            free_flight_age=free_flight_age,
        )
        ambiguity_flags = []
        if wrist_context.get("arm_ambiguous"):
            ambiguity_flags.append("RECOVERED_ARM_AMBIGUOUS")
        if "bad_backward_trajectory_fit" in score.quality_flags:
            ambiguity_flags.append("BAD_BACKWARD_FIT")
        reason_codes: list[str] = []
        arbitration = score.score
        if exact:
            arbitration += 0.06
        else:
            arbitration -= 0.16
            reason_codes.append("NON_EXACT_TRACK_HYPOTHESIS")
        if candidate_type == "OBSERVED":
            reason_codes.append("OBSERVED_FREE_FLIGHT_SUPPORTED")
            if "ball_wrist_proximity" in score.source:
                arbitration += 0.13
                reason_codes.append("OBSERVED_HAND_REGION_CANDIDATE")
        elif candidate_type == PRETRACK_RECOVERED:
            reason_codes.append("PRETRACK_RECOVERED_CANDIDATE")
            if recovered_count >= 2:
                arbitration += 0.14
                reason_codes.append("RECOVERED_MULTI_FRAME_SUPPORTED")
            else:
                arbitration -= 0.05
                reason_codes.append("RECOVERED_SINGLE_FRAME")
        else:
            reason_codes.append("RECOVERED_REJECTED_BY_ARBITRATION")
        if hand_label == "STRONG":
            arbitration += 0.16
            reason_codes.append("RECOVERED_HAND_ASSOCIATED")
        elif hand_label == "WEAK":
            arbitration += 0.04
        else:
            reason_codes.append("HAND_ASSOCIATION_UNAVAILABLE")
        if separation_label == "CONFIRMED":
            arbitration += 0.14
            reason_codes.append("RECOVERED_SEPARATION_CONFIRMED")
        elif separation_label == "WEAK":
            arbitration += 0.04
        elif separation_label == "NEGATIVE":
            arbitration -= 0.12
        if (
            candidate_type == PRETRACK_RECOVERED
            and "pretrack_recovered_candidate" in score.source
            and "earliest_reliable_moving_segment" in score.source
            and hand_label == "UNAVAILABLE"
            and separation_label == "UNAVAILABLE"
        ):
            arbitration += 0.12
            reason_codes.append("RECOVERED_EXACT_TRACK_CANDIDATE")
        if backward_quality >= 0.8:
            arbitration += 0.06
        elif backward_quality <= 0.2:
            arbitration -= 0.12
        if features.forward_free_flight_confirmation >= 0.8:
            arbitration += 0.03
        if wrist_context.get("arm_ambiguous"):
            arbitration -= 0.08
        arbitration += 0.12 * event_localization
        arbitration -= late_penalty
        if temporal_eligibility == "onset_region":
            reason_codes.append("FREE_FLIGHT_ONSET_REGION")
        elif temporal_eligibility == "late_requires_direct_release_evidence":
            reason_codes.append("ESTABLISHED_FREE_FLIGHT_BIAS_GUARD")
        elif temporal_eligibility == "late_direct_release_supported":
            reason_codes.append("LATE_FRAME_DIRECT_RELEASE_SUPPORTED")
        if (
            score.frame_index < first_primary_track_frame
            and candidate_type == PRETRACK_RECOVERED
        ):
            reason_codes.append("RELEASE_PRECEDES_FIRST_RELIABLE_OBSERVATION")
        eligible = self._hypothesis_eligible(
            candidate_type=candidate_type,
            exact=exact,
            hand_label=hand_label,
            separation_label=separation_label,
            backward_quality=backward_quality,
            recovered_count=recovered_count,
            wrist_context=wrist_context,
            score=score,
        )
        if temporal_eligibility == "late_requires_direct_release_evidence":
            eligible = False
        if not eligible and candidate_type == PRETRACK_RECOVERED:
            reason_codes.append("RECOVERED_REJECTED_BY_ARBITRATION")
        if candidate_type == PRETRACK_RECOVERED and not (
            "RECOVERED_MULTI_FRAME_SUPPORTED" in reason_codes
            or "RECOVERED_HAND_ASSOCIATED" in reason_codes
            or "RECOVERED_SEPARATION_CONFIRMED" in reason_codes
        ):
            reason_codes.append("RECOVERED_TRAJECTORY_ONLY_WEAK")
        return ReleaseHypothesis(
            frame_index=score.frame_index,
            candidate_type=candidate_type,
            exact_observation=exact,
            trajectory_compatibility=trajectory,
            hand_association=hand_label,
            hand_association_score=hand_score,
            separation_transition=separation_label,
            separation_score=separation_score,
            pose_quality=pose_quality,
            bowling_arm_confidence=wrist_context.get("arm_confidence"),
            forward_flight=features.forward_free_flight_confirmation,
            backward_fit_quality=backward_quality,
            detector_evidence=detector,
            free_flight_onset_frame=free_flight_onset,
            free_flight_age_frames=free_flight_age,
            established_free_flight=(
                free_flight_age is not None
                and free_flight_age >= self.config.minimum_free_flight_points
            ),
            future_confirmation_strength=future_confirmation,
            event_localization_strength=event_localization,
            late_flight_penalty=late_penalty,
            release_temporal_eligibility=temporal_eligibility,
            ambiguity_flags=ambiguity_flags,
            eligible=eligible,
            arbitration_score=max(0.0, min(1.0, arbitration)),
            reason_codes=_unique_flags(reason_codes),
            source_score=score,
        )

    def _event_localization_strength(
        self,
        *,
        score: CandidateScore,
        candidate_type: str,
        hand_label: str,
        separation_label: str,
        backward_quality: float,
        free_flight_age: int | None,
    ) -> float:
        strength = 0.0
        if free_flight_age is not None:
            if free_flight_age <= 0:
                strength += 0.38
            elif free_flight_age <= 2:
                strength += 0.24
            else:
                strength += max(0.0, 0.18 - 0.05 * (free_flight_age - 2))
        if hand_label == "STRONG":
            strength += 0.30
        elif hand_label == "WEAK":
            strength += 0.14
        if separation_label == "CONFIRMED":
            strength += 0.24
        elif separation_label == "WEAK":
            strength += 0.10
        if candidate_type == PRETRACK_RECOVERED:
            strength += 0.10
        elif candidate_type == "OBSERVED" and "earliest_reliable_moving_segment" in score.source:
            strength += 0.10
        elif candidate_type == "OBSERVED" and "track_start_window" in score.source:
            strength += 0.05
        if backward_quality >= 0.8:
            strength += 0.08
        return max(0.0, min(1.0, strength))

    def _late_flight_penalty(
        self,
        *,
        candidate_type: str,
        hand_label: str,
        separation_label: str,
        event_localization: float,
        free_flight_age: int | None,
    ) -> float:
        if free_flight_age is None or free_flight_age <= 2:
            return 0.0
        direct_release_evidence = (
            hand_label in {"STRONG", "WEAK"}
            and separation_label in {"CONFIRMED", "WEAK"}
        )
        if direct_release_evidence and event_localization >= 0.58:
            return 0.0
        base = min(0.42, 0.11 * (free_flight_age - 2))
        if candidate_type != "OBSERVED":
            base *= 0.5
        return round(base, 6)

    def _release_temporal_eligibility(
        self,
        *,
        candidate_type: str,
        hand_label: str,
        separation_label: str,
        event_localization: float,
        late_penalty: float,
        free_flight_age: int | None,
    ) -> str:
        if free_flight_age is None:
            return "unknown"
        if free_flight_age <= 2:
            return "onset_region"
        direct_release_evidence = (
            hand_label in {"STRONG", "WEAK"}
            and separation_label in {"CONFIRMED", "WEAK"}
            and event_localization >= 0.58
        )
        if direct_release_evidence:
            return "late_direct_release_supported"
        if candidate_type == "OBSERVED" and late_penalty > 0:
            return "late_requires_direct_release_evidence"
        return "weak_localization"

    def _candidate_type(
        self,
        score: CandidateScore,
        observed_frames: set[int],
        recovered_frames: set[int],
    ) -> str:
        frame = score.frame_index
        if frame in recovered_frames and score.features.ball_candidate_id is not None:
            return PRETRACK_RECOVERED
        if frame in observed_frames and score.features.ball_candidate_id is not None:
            return "OBSERVED"
        if score.features.ball_candidate_id is not None:
            return "DETECTOR_CANDIDATE_ONLY"
        return "RECONSTRUCTED_OR_POSE_ONLY"

    def _hand_association(
        self,
        features: ReleaseFeatures,
        wrist_context: dict[str, Any],
    ) -> tuple[str, float | None]:
        if (
            wrist_context.get("arm_ambiguous")
            or features.wrist_confidence is None
            or features.normalized_ball_wrist_distance is None
        ):
            return "UNAVAILABLE", None
        if (
            features.wrist_confidence >= self.config.wrist_confidence_threshold
            and features.normalized_ball_wrist_distance <= 0.8
        ):
            return "STRONG", 1.0
        if features.normalized_ball_wrist_distance <= 2.4 and features.wrist_confidence >= 0.2:
            return "WEAK", max(
                0.0,
                1.0 - features.normalized_ball_wrist_distance / 2.4,
            )
        return "UNAVAILABLE", None

    def _separation_transition(
        self,
        features: ReleaseFeatures,
    ) -> tuple[str, float]:
        persistence_score = min(
            1.0,
            features.separation_persistence_frames
            / max(1, self.config.persistent_separation_frames),
        )
        velocity = features.separation_velocity
        velocity_score = 0.0
        if velocity is not None:
            velocity_score = max(0.0, min(1.0, velocity / 55.0))
        score = max(persistence_score, velocity_score)
        if persistence_score >= 0.66 and velocity_score > 0.05:
            return "CONFIRMED", score
        if score > 0.15:
            return "WEAK", score
        if velocity is not None and velocity <= 0 and features.separation_persistence_frames == 0:
            return "NEGATIVE", 0.0
        return "UNAVAILABLE", 0.0

    def _backward_fit_quality(self, features: ReleaseFeatures) -> float:
        fit = features.backward_trajectory_fit_error_px
        if fit is None:
            return 0.0
        if fit <= self.config.backward_fit_good_px:
            return 1.0
        if fit >= self.config.backward_fit_weak_px:
            return 0.0
        span = self.config.backward_fit_weak_px - self.config.backward_fit_good_px
        return max(0.0, min(1.0, 1.0 - (fit - self.config.backward_fit_good_px) / span))

    def _hypothesis_eligible(
        self,
        *,
        candidate_type: str,
        exact: bool,
        hand_label: str,
        separation_label: str,
        backward_quality: float,
        recovered_count: int,
        wrist_context: dict[str, Any],
        score: CandidateScore,
    ) -> bool:
        if candidate_type == "OBSERVED":
            return exact
        if candidate_type == PRETRACK_RECOVERED:
            reliable_arm = not wrist_context.get("arm_ambiguous")
            strong_independent = (
                backward_quality >= 0.8
                and reliable_arm
                and hand_label in {"STRONG", "WEAK"}
                and separation_label in {"CONFIRMED", "WEAK"}
            )
            multi_frame_temporal = (
                backward_quality >= 0.8
                and recovered_count >= 2
                and separation_label in {"CONFIRMED", "WEAK", "UNAVAILABLE"}
            )
            trajectory_only_supporting = (
                backward_quality >= 0.95
                and recovered_count >= 1
                and score.source == "pretrack_recovered_candidate"
                and score.features.forward_free_flight_confirmation <= 0.2
            )
            exact_recovered_temporal_support = (
                wrist_context.get("arm_ambiguous")
                and hand_label == "UNAVAILABLE"
                and separation_label == "UNAVAILABLE"
                and recovered_count >= 1
                and "pretrack_recovered_candidate" in score.source
                and "earliest_reliable_moving_segment" in score.source
                and score.score >= self.config.low_confidence_threshold
            )
            return (
                strong_independent
                or multi_frame_temporal
                or trajectory_only_supporting
                or exact_recovered_temporal_support
            )
        if candidate_type == "DETECTOR_CANDIDATE_ONLY":
            return (
                hand_label == "STRONG"
                and separation_label in {"CONFIRMED", "WEAK"}
                and backward_quality >= 0.4
            )
        return (
            score.observed
            and hand_label == "STRONG"
            and separation_label in {"CONFIRMED", "WEAK"}
        )

    def _hypothesis_disagreement_penalty(
        self,
        best: ReleaseHypothesis,
        eligible: list[ReleaseHypothesis],
    ) -> float:
        competing = [
            item
            for item in eligible
            if abs(item.frame_index - best.frame_index) >= 3
            and item.arbitration_score >= best.arbitration_score - 0.12
        ]
        return 0.015 if competing else 0.0

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
        pretrack_hypothesis: PretrackHypothesis | None = None,
        wrist_context: dict[str, Any] | None = None,
        arbitration: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        uncertainty = self._uncertainty(score, all_scores, arbitration=arbitration)
        status = "ready" if score.score >= self.config.low_confidence_threshold else "unresolved"
        features = score.features
        evidence = feature_dict(features)
        evidence.update(
            {
                "score_components": score.score_components,
                "candidate_source": score.source,
                "observed": score.observed,
                "bowler_person_id": provenance.get("bowler_id"),
                "inferred_bowling_arm": (wrist_context or {}).get("inferred_bowling_arm"),
                "arm_confidence": (wrist_context or {}).get("arm_confidence"),
                "wrist_used": (wrist_context or {}).get("wrist_used"),
                "arm_ambiguous": (wrist_context or {}).get("arm_ambiguous"),
                "pretrack_reconstruction": None
                if pretrack_hypothesis is None
                else pretrack_hypothesis.diagnostics(),
                "release_hypotheses": []
                if arbitration is None
                else arbitration.get("release_hypotheses", []),
                "arbitration": None
                if arbitration is None
                else {
                    key: value
                    for key, value in arbitration.items()
                    if key != "release_hypotheses"
                },
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
        arbitration: dict[str, Any] | None = None,
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
        if arbitration:
            onset = arbitration.get("free_flight_onset_frame")
            temporal = arbitration.get("selected_release_temporal_eligibility")
            if isinstance(onset, int) and score.frame_index < onset:
                start = min(start, score.frame_index - 4)
                end = max(end, onset + 2)
            if temporal in {"weak_localization", "late_requires_direct_release_evidence"}:
                start = min(start, score.frame_index - 4)
                end = max(end, score.frame_index + 4)
        return {"start": max(0, start), "end": max(0, end)}

    def _quality_summary(
        self,
        scores: list[CandidateScore],
        pose_sequence: BowlerPoseSequence | None,
        pretrack_hypothesis: PretrackHypothesis | None = None,
        wrist_context: dict[str, Any] | None = None,
        arbitration: dict[str, Any] | None = None,
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
            "pretrack_reconstruction": None
            if pretrack_hypothesis is None
            else pretrack_hypothesis.diagnostics(),
            "wrist_context": wrist_context or {},
            "arbitration": arbitration,
        }

    def _wrist_context(
        self,
        pose_sequence: BowlerPoseSequence | None,
    ) -> dict[str, Any]:
        if pose_sequence is None:
            return {
                "inferred_bowling_arm": None,
                "arm_confidence": None,
                "wrist_used": None,
                "arm_ambiguous": True,
            }
        arm = pose_sequence.bowling_arm
        confidence = pose_sequence.arm_confidence
        ambiguous = (
            pose_sequence.arm_ambiguous
            or arm not in {"left", "right"}
            or confidence is None
            or confidence < self.config.bowling_arm_confidence_threshold
        )
        return {
            "inferred_bowling_arm": arm,
            "arm_confidence": confidence,
            "wrist_used": None if ambiguous or arm is None else f"{arm}_wrist",
            "arm_ambiguous": ambiguous,
        }

    def _release_hypothesis_track(
        self,
        primary_track: list[TrackObservation],
        pretrack_hypothesis: PretrackHypothesis,
    ) -> list[TrackObservation]:
        recovered = [
            point.track_point
            for point in pretrack_hypothesis.recovered_points
            if point.provenance == PRETRACK_RECOVERED
        ]
        return sorted(
            [*recovered, *primary_track],
            key=lambda point: point.frame_index,
        )

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
        "diagnostics": score.diagnostics or {},
    }
