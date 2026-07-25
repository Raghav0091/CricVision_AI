from __future__ import annotations

from Backends.src.release_point.release_engine import ReleaseEstimator


def _candidate(frame: int, x: float, y: float, confidence: float = 0.82, rank: int = 1):
    return {
        "candidate_id": f"frame_{frame:06d}_candidate_{rank:03d}",
        "class_id": 0,
        "class_name": "ball",
        "confidence": confidence,
        "bbox_xyxy": [x - 4, y - 4, x + 4, y + 4],
        "bbox_normalized": {
            "x": (x - 4) / 1280,
            "y": (y - 4) / 720,
            "width": 8 / 1280,
            "height": 8 / 720,
        },
        "center": {"x": x, "y": y},
        "center_normalized": {"x": x / 1280, "y": y / 720},
        "width_pixels": 8,
        "height_pixels": 8,
        "area_pixels": 64,
        "inside_pitch_corridor": True,
    }


def _detections(
    start: int,
    points: list[tuple[float, float]],
    *,
    total_frames: int = 40,
    missing: set[int] | None = None,
    extras: dict[int, list[dict]] | None = None,
):
    missing = missing or set()
    extras = extras or {}
    frames = []
    point_by_frame = {start + offset: point for offset, point in enumerate(points)}
    for frame in range(total_frames):
        detections = []
        if frame in point_by_frame and frame not in missing:
            x, y = point_by_frame[frame]
            detections.append(_candidate(frame, x, y))
        detections.extend(extras.get(frame, []))
        frames.append(
            {
                "frame_index": frame,
                "timestamp_seconds": frame / 30,
                "processed": True,
                "detections": detections,
            }
        )
    return {
        "analysis_id": "analysis_test",
        "detector": {"key": "e4c_best_overall", "name": "E4C", "model_file": "e4c.pt"},
        "model_path_used": "Models/e4c.pt",
        "model_class_names": ["ball"],
        "settings": {
            "frame_stride": 1,
            "imgsz": 960,
            "confidence_threshold": 0.15,
            "max_det": 20,
        },
        "frames": frames,
    }


def _tracking(start: int, points: list[tuple[float, float]], *, confidence: float = 0.82):
    primary_track = []
    previous = None
    for offset, (x, y) in enumerate(points):
        frame = start + offset
        if previous is None:
            vx = 0.0
            vy = 0.0
        else:
            vx = x - previous[0]
            vy = y - previous[1]
        previous = (x, y)
        primary_track.append(
            {
                "frame_index": frame,
                "timestamp_seconds": frame / 30,
                "source": "observed",
                "provenance": "OBSERVED",
                "candidate_id": f"frame_{frame:06d}_candidate_001",
                "x": x,
                "y": y,
                "normalized_x": x / 1280,
                "normalized_y": y / 720,
                "confidence": confidence,
                "uncertainty": 0.05,
                "vx": vx / 1280,
                "vy": vy / 720,
                "prediction_error": None,
                "inside_pitch_corridor": True,
            }
        )
    return {
        "analysis_id": "analysis_test",
        "status": "ready",
        "created_at": "2026-07-23T00:00:00Z",
        "completed_at": "2026-07-23T00:00:01Z",
        "settings": {
            "motion_model": "constant_velocity_recent_median",
            "max_recoverable_gap": 6,
            "minimum_observed_points": 3,
            "static_radius_normalized": 0.012,
            "base_gate_normalized": 0.025,
            "maximum_gate_normalized": 0.16,
            "history_points": 8,
            "beam_width": 4,
            "tracker_version": "delivery_track_v2",
        },
        "primary_track": primary_track,
        "raw_primary_track": primary_track,
        "candidate_diagnostics": [],
        "bounce": None,
        "message": "ready",
    }


def _pose(
    start: int,
    *,
    wrist_confidence: float = 0.86,
    hidden_ball: bool = False,
    bowling_arm: str | None = "right",
    arm_confidence: float | None = None,
    left_wrist_confidence: float | None = None,
    quality_flags: list[str] | None = None,
):
    poses = {}
    for offset in range(0, 12):
        frame = start + offset
        wrist_x = 300 + offset * 5
        wrist_y = 250 - min(offset, 3) * 6
        if hidden_ball and offset == 0:
            wrist_x -= 18
        poses[str(frame)] = {
            "person_id": "bowler_local_1",
            "frame_index": frame,
            "timestamp_seconds": frame / 30,
            "confidence": 0.88,
            "keypoints": {
                "right_shoulder": {"x": 250, "y": 270, "confidence": 0.85},
                "left_shoulder": {"x": 210, "y": 272, "confidence": 0.82},
                "left_elbow": {"x": 230 + offset * 2, "y": 275, "confidence": 0.8},
                "left_wrist": {
                    "x": wrist_x - 90,
                    "y": wrist_y + 18,
                    "confidence": left_wrist_confidence
                    if left_wrist_confidence is not None
                    else wrist_confidence,
                },
                "right_elbow": {"x": 280 + offset * 2, "y": 255, "confidence": 0.82},
                "right_wrist": {
                    "x": wrist_x,
                    "y": wrist_y,
                    "confidence": wrist_confidence,
                },
                "right_hip": {"x": 242, "y": 360, "confidence": 0.8},
                "left_hip": {"x": 210, "y": 360, "confidence": 0.78},
            },
        }
    return {
        "bowler_id": "bowler_01",
        "selection_confidence": 0.82,
        "poses_by_frame": poses,
        "quality_flags": quality_flags or [],
        "provider": {"name": "fake_pose", "model": "fixture", "schema": "test"},
        "bowling_arm": None
        if bowling_arm is None
        else {
            "bowling_arm": bowling_arm,
            "confidence": 0.7 if arm_confidence is None else arm_confidence,
            "quality_flags": quality_flags or [],
        },
    }


def _estimate(
    points,
    *,
    track_start=10,
    pose=None,
    missing=None,
    extras=None,
    pose_evidence_real=False,
):
    estimator = ReleaseEstimator()
    return estimator.estimate(
        analysis_id="analysis_test",
        fps=30,
        detections_document=_detections(
            track_start,
            points,
            missing=missing,
            extras=extras,
        ),
        tracking_document=_tracking(track_start, points),
        bowler_pose_sequence=pose,
        provenance={
            "tracking_version": "delivery_track_v2",
            "pose_provider": "unit_real_pose" if pose_evidence_real else None,
            "pose_evidence_real": pose_evidence_real,
        },
    )


def _release_hypotheses(estimate):
    return estimate.result["evidence"].get("release_hypotheses") or []


def _hypothesis_for(estimate, frame, candidate_type=None):
    for item in _release_hypotheses(estimate):
        if item["candidate_frame"] == frame and (
            candidate_type is None or item["candidate_type"] == candidate_type
        ):
            return item
    raise AssertionError(f"missing hypothesis for frame {frame} type {candidate_type}")


def test_observed_release_uses_pose_ball_separation():
    points = [(300, 250), (330, 252), (362, 255), (395, 258), (430, 261)]

    estimate = _estimate(points, pose=_pose(10), pose_evidence_real=True)

    assert estimate.result["status"] == "ready"
    assert estimate.result["release_type"] == "OBSERVED_RELEASE"
    assert estimate.result["evidence_mode"] == "observed_pose_ball_separation"
    assert estimate.result["release_frame"] in {10, 11}


def test_ball_temporarily_missing_at_release_expands_uncertainty():
    points = [(300, 250), (330, 252), (362, 255), (395, 258), (430, 261)]

    estimate = _estimate(points, pose=_pose(10), missing={10}, pose_evidence_real=True)

    assert estimate.result["status"] == "ready"
    assert estimate.result["release_type"] == "INFERRED_RELEASE"
    assert estimate.result["frame_uncertainty"]["end"] - estimate.result["frame_uncertainty"]["start"] >= 2


def test_pose_available_but_exact_ball_hidden_uses_trajectory_pose_inferred():
    points = [(318, 250), (346, 252), (374, 255), (404, 258), (436, 261)]

    estimate = _estimate(
        points,
        pose=_pose(10, hidden_ball=True),
        missing={10},
        pose_evidence_real=True,
    )

    assert estimate.result["status"] == "ready"
    assert estimate.result["evidence_mode"] == "trajectory_pose_inferred"


def test_trajectory_only_fallback_marks_lower_certainty():
    points = [(300, 250), (330, 252), (362, 255), (395, 258), (430, 261)]

    estimate = _estimate(points)

    assert estimate.result["status"] == "ready"
    assert estimate.result["evidence_mode"] == "fallback_trajectory_only"
    assert estimate.result["confidence"] <= 0.58
    assert "trajectory_only_estimate" in estimate.result["quality_flags"]


def test_false_ball_near_bowler_does_not_override_primary_track():
    points = [(500, 300), (530, 304), (562, 309), (595, 313), (630, 318)]
    false_near_wrist = _candidate(10, 301, 249, confidence=0.95, rank=2)

    estimate = _estimate(
        points,
        pose=_pose(10),
        pose_evidence_real=True,
        extras={10: [false_near_wrist]},
    )

    assert estimate.result["status"] == "ready"
    assert estimate.result["release_point_px"]["x"] in {500, 530}
    assert estimate.result["evidence"]["ball_candidate_id"] != false_near_wrist["candidate_id"]


def test_multiple_candidate_balls_prefers_candidate_matching_track():
    points = [(300, 250), (330, 252), (362, 255), (395, 258), (430, 261)]
    distractor = _candidate(10, 900, 600, confidence=0.99, rank=2)

    estimate = _estimate(
        points,
        pose=_pose(10),
        extras={10: [distractor]},
        pose_evidence_real=True,
    )

    assert estimate.result["release_point_px"]["x"] == 300
    assert estimate.result["evidence"]["candidate_rank"] == 2


def test_short_missing_detection_gap_still_confirms_forward_free_flight():
    points = [
        (300, 250),
        (330, 252),
        (362, 255),
        (395, 258),
        (430, 261),
        (466, 264),
    ]

    estimate = _estimate(points, pose=_pose(10), missing={12}, pose_evidence_real=True)

    assert estimate.result["status"] == "ready"
    assert estimate.result["evidence"]["forward_free_flight_confirmation"] > 0.7


def test_low_confidence_wrist_forces_non_observed_or_flagged_result():
    points = [(300, 250), (330, 252), (362, 255), (395, 258), (430, 261)]

    estimate = _estimate(
        points,
        pose=_pose(10, wrist_confidence=0.2),
        pose_evidence_real=True,
    )

    assert estimate.result["status"] == "ready"
    assert estimate.result["release_type"] == "INFERRED_RELEASE"
    assert "low_confidence_wrist" in estimate.result["quality_flags"]


def test_bad_backward_trajectory_adds_quality_flag():
    points = [(300, 250), (330, 252), (362, 255), (395, 258), (430, 261)]
    off_path = _candidate(9, 80, 650, confidence=0.9)

    estimate = _estimate(
        points,
        pose=_pose(10),
        extras={9: [off_path]},
        pose_evidence_real=True,
    )
    flagged = [
        score
        for score in estimate.candidate_scores
        if "bad_backward_trajectory_fit" in score.quality_flags
    ]

    assert flagged


def test_valid_forward_free_flight_confirmation_is_exposed():
    points = [(300, 250), (330, 252), (362, 255), (395, 258), (430, 261)]

    estimate = _estimate(points, pose=_pose(10), pose_evidence_real=True)

    assert estimate.result["evidence"]["forward_free_flight_points"] >= 3
    assert estimate.result["evidence"]["forward_free_flight_confirmation"] > 0.7


def test_uncertainty_expands_when_evidence_is_weak():
    weak_points = [(300, 250), (315, 251), (330, 252), (346, 253)]

    estimate = _estimate(weak_points)
    interval = estimate.result["frame_uncertainty"]

    assert interval["end"] - interval["start"] >= 4


def test_unresolved_result_when_evidence_is_insufficient():
    estimate = _estimate([(300, 250), (301, 250)], pose=None)

    assert estimate.result["status"] == "unresolved"
    assert estimate.result["release_frame"] is None
    assert "insufficient_primary_track" in estimate.result["quality_flags"]


def test_pose_without_real_provenance_cannot_emit_observed_release():
    points = [(300, 250), (330, 252), (362, 255), (395, 258), (430, 261)]

    estimate = _estimate(points, pose=_pose(10), pose_evidence_real=False)

    assert estimate.result["evidence_mode"] == "fallback_trajectory_only"
    assert estimate.result["release_type"] == "INFERRED_RELEASE"


def test_pretrack_search_recovers_coherent_earlier_candidate():
    points = [(350, 250), (380, 252), (410, 254), (440, 256), (470, 258)]
    recovered = _candidate(14, 320, 248, confidence=0.35)

    estimate = _estimate(
        points,
        track_start=15,
        pose=_pose(10),
        extras={14: [recovered]},
        pose_evidence_real=True,
    )

    diagnostics = estimate.result["evidence"]["pretrack_reconstruction"]
    recovered_points = diagnostics["recovered_pretrack_points"]

    assert any(
        point["candidate_id"] == recovered["candidate_id"]
        and point["provenance"] == "PRETRACK_RECOVERED"
        for point in recovered_points
    )
    assert any(
        "pretrack_recovered_candidate" in score.source
        for score in estimate.candidate_scores
    )


def test_pretrack_rejects_high_confidence_spatial_false_positive():
    points = [(350, 250), (380, 252), (410, 254), (440, 256), (470, 258)]
    false_positive = _candidate(14, 900, 610, confidence=0.99, rank=1)
    coherent = _candidate(14, 320, 248, confidence=0.28, rank=2)

    estimate = _estimate(
        points,
        track_start=15,
        pose=_pose(10),
        extras={14: [false_positive, coherent]},
        pose_evidence_real=True,
    )

    gates = estimate.result["evidence"]["pretrack_reconstruction"]["candidate_gates"]
    false_gate = next(
        item for item in gates if item["candidate_id"] == false_positive["candidate_id"]
    )
    coherent_gate = next(
        item for item in gates if item["candidate_id"] == coherent["candidate_id"]
    )

    assert false_gate["accepted"] is False
    assert false_gate["rejection_reason"] == "outside_backward_projection_gate"
    assert coherent_gate["accepted"] is True


def test_lower_confidence_trajectory_consistent_pretrack_candidate_can_win_gate():
    points = [(350, 250), (380, 252), (410, 254), (440, 256), (470, 258)]
    distractor = _candidate(13, 470, 330, confidence=0.92, rank=1)
    coherent = _candidate(13, 290, 246, confidence=0.22, rank=2)

    estimate = _estimate(
        points,
        track_start=15,
        pose=_pose(10),
        extras={13: [distractor, coherent]},
        pose_evidence_real=True,
    )

    recovered_ids = {
        point["candidate_id"]
        for point in estimate.result["evidence"]["pretrack_reconstruction"][
            "recovered_pretrack_points"
        ]
    }

    assert coherent["candidate_id"] in recovered_ids
    assert distractor["candidate_id"] not in recovered_ids


def test_inferred_right_arm_binds_release_features_to_right_wrist():
    points = [(300, 250), (330, 252), (362, 255), (395, 258), (430, 261)]

    estimate = _estimate(
        points,
        pose=_pose(
            10,
            wrist_confidence=0.42,
            left_wrist_confidence=0.98,
        ),
        pose_evidence_real=True,
    )

    assert estimate.result["evidence"]["inferred_bowling_arm"] == "right"
    assert estimate.result["evidence"]["wrist_used"] == "right_wrist"
    assert estimate.result["evidence"]["wrist_confidence"] == 0.42


def test_ambiguous_bowling_arm_lowers_pose_influence():
    points = [(300, 250), (330, 252), (362, 255), (395, 258), (430, 261)]

    estimate = _estimate(
        points,
        pose=_pose(
            10,
            arm_confidence=0.05,
            quality_flags=["bowling_arm_ambiguous"],
        ),
        pose_evidence_real=True,
    )

    assert estimate.result["release_type"] == "INFERRED_RELEASE"
    assert estimate.result["evidence"]["arm_ambiguous"] is True
    assert "pose_unavailable_or_unreliable" in estimate.result["quality_flags"]


def test_pretrack_missing_detector_candidate_is_reconstructed_not_observed():
    points = [(350, 250), (380, 252), (410, 254), (440, 256), (470, 258)]

    estimate = _estimate(
        points,
        track_start=15,
        pose=_pose(10),
        pose_evidence_real=True,
    )

    recovered_points = estimate.result["evidence"]["pretrack_reconstruction"][
        "recovered_pretrack_points"
    ]
    reconstructed = [
        point for point in recovered_points if point["provenance"] == "RECONSTRUCTED"
    ]

    assert reconstructed
    assert all(point["candidate_id"] is None for point in reconstructed)
    assert all(point["provenance"] != "OBSERVED" for point in reconstructed)


def test_no_pose_fallback_remains_capped_after_v1_1_pretrack_changes():
    points = [(300, 250), (330, 252), (362, 255), (395, 258), (430, 261)]

    estimate = _estimate(points)

    assert estimate.result["status"] == "ready"
    assert estimate.result["evidence_mode"] == "fallback_trajectory_only"
    assert estimate.result["confidence"] <= 0.58


def test_v1_2_recovered_candidate_with_hand_and_separation_can_win():
    points = [(420, 250), (455, 253), (490, 256), (525, 259), (560, 262)]
    recovered = _candidate(16, 335, 234, confidence=0.55)

    estimate = _estimate(
        points,
        track_start=18,
        pose=_pose(10),
        extras={16: [recovered]},
        pose_evidence_real=True,
    )

    arbitration = estimate.result["evidence"]["arbitration"]

    assert estimate.result["release_frame"] == 16
    assert arbitration["selected_candidate_type"] == "PRETRACK_RECOVERED"
    assert "RECOVERED_HAND_ASSOCIATED" in arbitration["selected_reason_codes"]
    assert "RECOVERED_SEPARATION_CONFIRMED" in arbitration["selected_reason_codes"]


def test_v1_2_ambiguous_arm_recovered_candidate_cannot_override_observed():
    points = [(350, 250), (380, 252), (410, 254), (440, 256), (470, 258)]
    recovered = _candidate(13, 290, 246, confidence=0.35)

    estimate = _estimate(
        points,
        track_start=15,
        pose=_pose(10, arm_confidence=0.05, quality_flags=["bowling_arm_ambiguous"]),
        extras={13: [recovered]},
        pose_evidence_real=True,
    )

    recovered_hypothesis = _hypothesis_for(estimate, 13, "PRETRACK_RECOVERED")

    assert estimate.result["evidence"]["arbitration"]["selected_candidate_type"] == "OBSERVED"
    assert recovered_hypothesis["final_arbitration_eligibility"] is False
    assert "RECOVERED_REJECTED_BY_ARBITRATION" in recovered_hypothesis[
        "arbitration_reason_codes"
    ]
    assert recovered_hypothesis["hand_association_evidence"] == "UNAVAILABLE"


def test_v1_2_single_isolated_recovered_candidate_cannot_large_shift_without_pose():
    points = [(420, 250), (455, 253), (490, 256), (525, 259), (560, 262)]
    recovered = _candidate(15, 330, 234, confidence=0.9)

    estimate = _estimate(
        points,
        track_start=18,
        pose=None,
        extras={15: [recovered]},
    )

    recovered_hypothesis = _hypothesis_for(estimate, 15, "PRETRACK_RECOVERED")

    assert estimate.result["evidence"]["arbitration"]["selected_candidate_type"] == "OBSERVED"
    assert estimate.result["release_frame"] >= 18
    assert recovered_hypothesis["final_arbitration_eligibility"] is False
    assert recovered_hypothesis["hand_association_evidence"] == "UNAVAILABLE"


def test_v1_2_multi_frame_recovered_sequence_strengthens_no_pose_fallback():
    points = [(420, 250), (455, 253), (490, 256), (525, 259), (560, 262)]
    extras = {
        15: [_candidate(15, 330, 234, confidence=0.55)],
        16: [_candidate(16, 360, 236, confidence=0.5)],
        17: [_candidate(17, 390, 239, confidence=0.45)],
    }

    estimate = _estimate(
        points,
        track_start=18,
        pose=None,
        extras=extras,
    )

    arbitration = estimate.result["evidence"]["arbitration"]

    assert estimate.result["release_frame"] == 15
    assert arbitration["selected_candidate_type"] == "PRETRACK_RECOVERED"
    assert "RECOVERED_MULTI_FRAME_SUPPORTED" in arbitration["selected_reason_codes"]
    assert _hypothesis_for(estimate, 15, "PRETRACK_RECOVERED")[
        "hand_association_evidence"
    ] == "UNAVAILABLE"


def test_v1_2_competing_hypotheses_lower_confidence():
    points = [(420, 250), (455, 253), (490, 256), (525, 259), (560, 262)]
    recovered = _candidate(15, 330, 234, confidence=0.55)

    estimate = _estimate(
        points,
        track_start=18,
        pose=_pose(10),
        extras={15: [recovered]},
        pose_evidence_real=True,
    )

    arbitration = estimate.result["evidence"]["arbitration"]
    selected = _hypothesis_for(estimate, estimate.result["release_frame"])

    assert arbitration["confidence_disagreement_penalty"] > 0
    assert estimate.result["confidence"] < selected["release_feature_score"]
    assert "competing_hypotheses_disagree" in estimate.result["quality_flags"]


def test_v1_2_pretrack_provenance_remains_distinct_from_observed():
    points = [(420, 250), (455, 253), (490, 256), (525, 259), (560, 262)]
    recovered = _candidate(16, 335, 234, confidence=0.55)

    estimate = _estimate(
        points,
        track_start=18,
        pose=_pose(10),
        extras={16: [recovered]},
        pose_evidence_real=True,
    )

    selected = _hypothesis_for(estimate, 16, "PRETRACK_RECOVERED")

    assert estimate.result["evidence"]["tracker_provenance"] == "PRETRACK_RECOVERED"
    assert selected["candidate_type"] == "PRETRACK_RECOVERED"
    assert any(
        item["candidate_type"] == "OBSERVED"
        for item in estimate.result["evidence"]["release_hypotheses"]
    )


def test_v1_3_late_free_flight_cannot_beat_onset_without_direct_evidence():
    points = [
        (420, 250),
        (455, 253),
        (490, 256),
        (525, 259),
        (560, 262),
        (595, 265),
        (630, 268),
        (665, 271),
    ]

    estimate = _estimate(points, track_start=18, pose=None)
    hypotheses = estimate.result["evidence"]["release_hypotheses"]
    late = _hypothesis_for(estimate, 23, "OBSERVED")

    assert estimate.result["release_frame"] == 18
    assert estimate.result["evidence"]["arbitration"]["free_flight_onset_frame"] == 18
    assert late["release_temporal_eligibility"] == "late_requires_direct_release_evidence"
    assert late["final_arbitration_eligibility"] is False
    assert "ESTABLISHED_FREE_FLIGHT_BIAS_GUARD" in late["arbitration_reason_codes"]
    assert any(item["future_confirmation_strength"] == 1.0 for item in hypotheses)


def test_v1_3_future_confirmation_supports_onset_without_shifting_later():
    points = [
        (420, 250),
        (455, 253),
        (490, 256),
        (525, 259),
        (560, 262),
        (595, 265),
        (630, 268),
        (665, 271),
    ]

    estimate = _estimate(points, track_start=18, pose=None)
    selected = _hypothesis_for(estimate, 18, "OBSERVED")

    assert selected["future_confirmation_strength"] == 1.0
    assert selected["event_localization_strength"] > 0
    assert selected["release_temporal_eligibility"] == "onset_region"
    assert estimate.result["release_frame"] == selected["candidate_frame"]


def test_v1_3_observation_gap_widens_uncertainty_for_recovered_release():
    points = [(420, 250), (455, 253), (490, 256), (525, 259), (560, 262)]
    recovered = _candidate(16, 335, 234, confidence=0.55)

    estimate = _estimate(
        points,
        track_start=18,
        pose=_pose(10),
        extras={16: [recovered]},
        pose_evidence_real=True,
    )

    interval = estimate.result["frame_uncertainty"]

    assert estimate.result["release_frame"] == 16
    assert interval["end"] >= 20
    assert "release_precedes_first_reliable_observation" in estimate.result["quality_flags"]
    assert "observational_gap_near_release" in estimate.result["quality_flags"]


def test_v1_3_direct_hand_separation_can_override_late_bias_guard():
    points = [
        (420, 250),
        (455, 253),
        (490, 256),
        (525, 259),
        (560, 262),
        (595, 265),
        (630, 268),
        (665, 271),
    ]
    pose = _pose(18)
    for frame in (21, 22, 23, 24, 25):
        wrist = pose["poses_by_frame"][str(frame)]["keypoints"]["right_wrist"]
        wrist["x"] = 525
        wrist["y"] = 259
        wrist["confidence"] = 0.9

    estimate = _estimate(points, track_start=18, pose=pose, pose_evidence_real=True)
    selected = _hypothesis_for(estimate, estimate.result["release_frame"], "OBSERVED")

    assert estimate.result["release_frame"] == 21
    assert selected["free_flight_age_frames"] >= 3
    assert selected["release_temporal_eligibility"] == "late_direct_release_supported"
    assert selected["late_flight_penalty"] == 0.0
    assert "RECOVERED_HAND_ASSOCIATED" in selected["arbitration_reason_codes"]
    assert "RECOVERED_SEPARATION_CONFIRMED" in selected["arbitration_reason_codes"]


def test_v1_3_false_early_detector_only_shift_remains_blocked():
    points = [(420, 250), (455, 253), (490, 256), (525, 259), (560, 262)]
    early_detector_only = _candidate(17, 340, 234, confidence=0.9)

    estimate = _estimate(
        points,
        track_start=18,
        pose=_pose(10, arm_confidence=0.05, quality_flags=["bowling_arm_ambiguous"]),
        extras={17: [early_detector_only]},
        pose_evidence_real=True,
    )

    early = _hypothesis_for(estimate, 17, "DETECTOR_CANDIDATE_ONLY")

    assert estimate.result["evidence"]["arbitration"]["selected_candidate_type"] == "OBSERVED"
    assert early["final_arbitration_eligibility"] is False
    assert "RECOVERED_REJECTED_BY_ARBITRATION" in early["arbitration_reason_codes"]
