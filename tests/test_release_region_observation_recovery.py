from __future__ import annotations

from Backends.src.release_point.features import (
    ReleasePointConfig,
    parse_bowler_pose_sequence,
    parse_detection_observations,
    parse_track_observations,
)
from Backends.src.release_point.release_region_recovery import (
    NO_CREDIBLE_RELEASE_REGION_CHAIN,
    OBSERVED_RELEASE_RECOVERED,
    augment_tracking_with_recovery,
    recover_release_region_observations,
)


def _candidate(frame: int, x: float, y: float, confidence: float, rank: int = 1):
    return {
        "candidate_id": f"frame_{frame:06d}_candidate_{rank:03d}",
        "class_name": "ball",
        "confidence": confidence,
        "bbox_xyxy": [x - 4, y - 4, x + 4, y + 4],
        "center": {"x": x, "y": y},
        "center_normalized": {"x": x / 1280, "y": y / 720},
        "inside_pitch_corridor": True,
    }


def _detections(frames: dict[int, list[dict]], total_frames: int = 30):
    return {
        "analysis_id": "analysis_test",
        "frames": [
            {
                "frame_index": frame,
                "timestamp_seconds": frame / 30,
                "processed": True,
                "detections": frames.get(frame, []),
            }
            for frame in range(total_frames)
        ],
    }


def _tracking(start: int = 10, points=None):
    points = points or [
        (350, 250),
        (380, 252),
        (410, 254),
        (440, 256),
        (470, 258),
    ]
    primary_track = []
    previous = None
    for offset, (x, y) in enumerate(points):
        frame = start + offset
        vx = 0.0 if previous is None else (x - previous[0]) / 1280
        vy = 0.0 if previous is None else (y - previous[1]) / 720
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
                "confidence": 0.82,
                "uncertainty": 0.05,
                "vx": vx,
                "vy": vy,
                "inside_pitch_corridor": True,
            }
        )
    return {
        "analysis_id": "analysis_test",
        "status": "ready",
        "settings": {"tracker_version": "delivery_track_v2"},
        "primary_track": primary_track,
    }


def _pose(start: int = 6, *, ambiguous: bool = False):
    poses = {}
    for frame in range(start, start + 8):
        offset = frame - start
        poses[str(frame)] = {
            "person_id": "bowler",
            "frame_index": frame,
            "timestamp_seconds": frame / 30,
            "confidence": 0.88,
            "keypoints": {
                "right_shoulder": {"x": 240, "y": 270, "confidence": 0.86},
                "left_shoulder": {"x": 200, "y": 272, "confidence": 0.84},
                "right_elbow": {"x": 260 + offset * 8, "y": 250, "confidence": 0.84},
                "right_wrist": {"x": 280 + offset * 18, "y": 246, "confidence": 0.9},
                "left_elbow": {"x": 205, "y": 310, "confidence": 0.8},
                "left_wrist": {"x": 198, "y": 360, "confidence": 0.78},
                "right_hip": {"x": 245, "y": 390, "confidence": 0.82},
                "left_hip": {"x": 205, "y": 390, "confidence": 0.8},
            },
        }
    return {
        "bowler_id": "bowler_01",
        "selection_confidence": 0.82,
        "poses_by_frame": poses,
        "quality_flags": ["bowling_arm_ambiguous"] if ambiguous else [],
        "provider": {"name": "rtmpose_mmpose", "model": "fixture", "schema": "coco17"},
        "bowling_arm": {
            "bowling_arm": "right",
            "confidence": 0.05 if ambiguous else 0.8,
            "quality_flags": ["bowling_arm_ambiguous"] if ambiguous else [],
        },
    }


def _recover(detections, tracking=None, pose=None):
    return recover_release_region_observations(
        detections_by_frame=parse_detection_observations(detections),
        primary_track=parse_track_observations(tracking or _tracking()),
        pose_sequence=parse_bowler_pose_sequence(pose),
        config=ReleasePointConfig(),
    )


def test_low_confidence_temporal_path_beats_isolated_high_confidence_false_positive():
    detections = _detections(
        {
            7: [_candidate(7, 260, 244, 0.28, 2), _candidate(7, 900, 610, 0.98, 1)],
            8: [_candidate(8, 290, 246, 0.30, 2)],
            9: [_candidate(9, 320, 248, 0.32, 2)],
        }
    )

    recovery = _recover(detections)
    recovered_ids = {
        point.observation.candidate_id for point in recovery.recovered_observations
    }

    assert recovery.status == "ready"
    assert "frame_000007_candidate_002" in recovered_ids
    assert "frame_000007_candidate_001" not in recovered_ids


def test_static_repeated_false_positive_is_rejected():
    detections = _detections(
        {
            7: [_candidate(7, 300, 300, 0.9)],
            8: [_candidate(8, 300, 300, 0.92)],
            9: [_candidate(9, 300, 300, 0.91)],
        }
    )

    recovery = _recover(detections)

    assert recovery.status == NO_CREDIBLE_RELEASE_REGION_CHAIN
    assert recovery.recovered_observations == []
    assert recovery.reason in {
        "static_or_low_motion_candidate_path",
        "insufficient_multi_frame_temporal_support",
    }


def test_backward_and_forward_agreement_strengthens_recovery():
    detections = _detections(
        {
            7: [_candidate(7, 260, 244, 0.35)],
            8: [_candidate(8, 290, 246, 0.35)],
            9: [_candidate(9, 320, 248, 0.35)],
        }
    )

    recovery = _recover(detections)

    assert recovery.status == "ready"
    assert recovery.path_score > 0.55
    assert all(point.forward_compatibility > 0.4 for point in recovery.recovered_observations)
    assert all(point.backward_compatibility > 0.4 for point in recovery.recovered_observations)


def test_temporally_inconsistent_candidate_path_is_rejected():
    detections = _detections(
        {
            7: [_candidate(7, 260, 244, 0.5)],
            8: [_candidate(8, 610, 500, 0.5)],
        }
    )

    recovery = _recover(detections)

    assert recovery.status == NO_CREDIBLE_RELEASE_REGION_CHAIN


def test_recovery_operates_without_pose():
    detections = _detections(
        {
            7: [_candidate(7, 260, 244, 0.28)],
            8: [_candidate(8, 290, 246, 0.30)],
            9: [_candidate(9, 320, 248, 0.32)],
        }
    )

    recovery = _recover(detections, pose=None)

    assert recovery.status == "ready"
    assert len(recovery.recovered_observations) == 3
    assert all(point.pose_support is None for point in recovery.recovered_observations)


def test_reliable_bowling_hand_evidence_strengthens_plausible_chain():
    detections = _detections({8: [_candidate(8, 290, 246, 0.34)]})

    recovery = _recover(detections, pose=_pose())

    assert recovery.status == "ready"
    assert len(recovery.recovered_observations) == 1
    assert recovery.recovered_observations[0].pose_support is not None


def test_ambiguous_pose_cannot_dominate_trajectory_evidence():
    detections = _detections(
        {
            7: [_candidate(7, 260, 244, 0.28, 2), _candidate(7, 318, 246, 0.95, 1)],
            8: [_candidate(8, 290, 246, 0.30)],
            9: [_candidate(9, 320, 248, 0.32)],
        }
    )

    recovery = _recover(detections, pose=_pose(ambiguous=True))
    recovered_ids = {
        point.observation.candidate_id for point in recovery.recovered_observations
    }

    assert recovery.status == "ready"
    assert "frame_000007_candidate_002" in recovered_ids
    assert all(point.pose_support is None for point in recovery.recovered_observations)


def test_large_unsupported_gap_returns_reconstruction_uncertainty_only():
    recovery = _recover(_detections({}))

    assert recovery.status == NO_CREDIBLE_RELEASE_REGION_CHAIN
    assert recovery.recovered_observations == []
    assert recovery.reconstructed_points


def test_no_credible_candidates_returns_explicit_status():
    detections = _detections({7: [_candidate(7, 40, 650, 0.2)]})

    recovery = _recover(detections)

    assert recovery.status == NO_CREDIBLE_RELEASE_REGION_CHAIN


def test_recovered_provenance_remains_distinct_from_primary_observed():
    detections = _detections(
        {
            7: [_candidate(7, 260, 244, 0.28)],
            8: [_candidate(8, 290, 246, 0.30)],
            9: [_candidate(9, 320, 248, 0.32)],
        }
    )
    tracking = _tracking()

    recovery = _recover(detections, tracking=tracking)
    augmented = augment_tracking_with_recovery(tracking, recovery)
    provenances = {point["provenance"] for point in augmented["primary_track"]}

    assert OBSERVED_RELEASE_RECOVERED in provenances
    assert "OBSERVED" in provenances
    assert all(
        point["provenance"] != "RECONSTRUCTED"
        for point in augmented["primary_track"]
    )
