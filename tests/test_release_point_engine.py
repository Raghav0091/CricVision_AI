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


def _pose(start: int, *, wrist_confidence: float = 0.86, hidden_ball: bool = False):
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
        "quality_flags": [],
        "provider": {"name": "fake_pose", "model": "fixture", "schema": "test"},
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
