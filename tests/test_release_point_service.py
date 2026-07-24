from __future__ import annotations

import pytest

from services.api.schemas.release_point import ReleaseAnalysisInput
from Backends.src.release_point.pose_provider import (
    Keypoint,
    PoseProviderInfo,
    PosePerson,
    PoseSequence,
)
from services.api.services.video_release_point_service import (
    POSE_PROVIDER_ENV,
    VideoReleasePointError,
    _provenance,
    _resolve_pose_context,
)
from services.api.services import video_release_point_service


def _release_input(raw_path: str = "outputs/video_analysis/analysis_1/raw/original.mp4"):
    return ReleaseAnalysisInput(
        analysis_id="analysis_test",
        raw_video_path=raw_path,
        fps=30.0,
        frame_count=40,
        width=1280,
        height=720,
        detections_path="outputs/video_analysis/analysis_1/detections/detections.json",
        tracking_path="outputs/video_analysis/analysis_1/tracking/tracking_result.json",
        calibration_path="outputs/video_analysis/analysis_1/calibration/calibration.json",
    )


def _detections(detector_key: str = "e3_motion_blur"):
    return {
        "analysis_id": "analysis_test",
        "detector": {"key": detector_key, "name": "E3", "model_file": "e3.pt"},
        "model_path_used": "Models/e3.pt",
        "frames": [],
    }


def _tracking():
    return {
        "analysis_id": "analysis_test",
        "settings": {"tracker_version": "delivery_track_v2"},
    }


def test_no_configured_pose_provider_is_explicit_not_run(monkeypatch):
    monkeypatch.delenv(POSE_PROVIDER_ENV, raising=False)

    context = _resolve_pose_context(
        _release_input(),
        calibration={},
        calibration_v2=None,
        camera_pose=None,
        tracking=_tracking(),
        bowler_pose_sequence=None,
    )

    assert context.bowler_pose_sequence is None
    assert context.provenance["status"] == "not_run"
    assert context.provenance["evidence_real"] is False
    assert context.quality_flags == ["pose_not_run"]


def test_configured_but_unavailable_pose_provider_falls_back_honestly(monkeypatch):
    monkeypatch.setenv(POSE_PROVIDER_ENV, "rtmpose")

    context = _resolve_pose_context(
        _release_input(),
        calibration={},
        calibration_v2=None,
        camera_pose=None,
        tracking=_tracking(),
        bowler_pose_sequence=None,
    )

    assert context.bowler_pose_sequence is None
    assert context.provenance["status"] == "unavailable"
    assert context.provenance["configured_provider"] == "rtmpose"
    assert "pose_provider_unavailable" in context.quality_flags
    assert context.provenance["evidence_real"] is False


def test_fake_pose_provider_is_rejected_at_production_boundary(monkeypatch):
    monkeypatch.delenv(POSE_PROVIDER_ENV, raising=False)
    fake_bowler_pose = {
        "bowler_id": "bowler_01",
        "selection_confidence": 0.9,
        "poses_by_frame": {},
        "provider": {"name": "fake_pose", "model": "unit", "schema": "test"},
    }

    with pytest.raises(VideoReleasePointError, match="test-only"):
        _resolve_pose_context(
            _release_input(),
            calibration={},
            calibration_v2=None,
            camera_pose=None,
            tracking=_tracking(),
            bowler_pose_sequence=fake_bowler_pose,
        )


def test_clean_video_path_guard_rejects_overlay_for_pose_context(monkeypatch):
    monkeypatch.delenv(POSE_PROVIDER_ENV, raising=False)

    with pytest.raises(ValueError, match="clean original"):
        _resolve_pose_context(
            _release_input("outputs/video_analysis/analysis_1/tracking/tracking_debug.mp4"),
            calibration={},
            calibration_v2=None,
            camera_pose=None,
            tracking=_tracking(),
            bowler_pose_sequence=None,
        )


def test_release_provenance_preserves_detector_metadata(monkeypatch):
    monkeypatch.delenv(POSE_PROVIDER_ENV, raising=False)
    context = _resolve_pose_context(
        _release_input(),
        calibration={"mode": "automatic_visual"},
        calibration_v2=None,
        camera_pose=None,
        tracking=_tracking(),
        bowler_pose_sequence=None,
    )

    provenance = _provenance(
        _detections("e2_baseline"),
        _tracking(),
        {"mode": "automatic_visual"},
        None,
        None,
        context,
    )

    assert provenance["ball_detector_model_key"] == "e2_baseline"
    assert provenance["tracking_version"] == "delivery_track_v2"
    assert provenance["pose_provider"] is None
    assert provenance["pose_evidence_real"] is False
    assert provenance["quality_flags"] == ["pose_not_run"]


def test_configured_real_provider_can_create_real_pose_context(monkeypatch):
    monkeypatch.setenv(POSE_PROVIDER_ENV, "rtmpose")

    class Provider:
        provider_info = PoseProviderInfo(
            "rtmpose_mmpose",
            "rtmpose-m_8xb256-420e_body8-256x192",
            "coco17",
        )

        def __init__(self, _config):
            pass

        def estimate_sequence(self, _path, frame_window, fps):
            frames = {}
            for frame in frame_window[:4]:
                frames[frame] = [
                    PosePerson(
                        person_id=f"person_{frame}",
                        frame_index=frame,
                        timestamp_seconds=frame / fps,
                        bbox_xyxy=(210, 140, 420, 700),
                        confidence=0.92,
                        keypoints={
                            "left_shoulder": Keypoint(260, 250, 0.8),
                            "right_shoulder": Keypoint(310, 250, 0.82),
                            "left_elbow": Keypoint(255, 305, 0.78),
                            "right_elbow": Keypoint(330 + frame, 230, 0.8),
                            "left_wrist": Keypoint(250, 360, 0.76),
                            "right_wrist": Keypoint(350 + frame * 12, 210, 0.84),
                            "left_hip": Keypoint(270, 430, 0.8),
                            "right_hip": Keypoint(315, 430, 0.8),
                            "left_knee": Keypoint(260, 530, 0.76),
                            "right_knee": Keypoint(320, 525, 0.76),
                            "left_ankle": Keypoint(255, 600, 0.75),
                            "right_ankle": Keypoint(325, 598, 0.75),
                        },
                        provider=self.provider_info,
                    )
                ]
            return PoseSequence(frames=frames, provider=self.provider_info)

    monkeypatch.setattr(video_release_point_service, "RTMPoseProvider", Provider)
    context = _resolve_pose_context(
        _release_input(),
        calibration={
            "image_width": 1280,
            "image_height": 720,
            "non_striker_wicket": {"bottom_center": {"x": 0.25, "y": 0.83}},
            "striker_wicket": {"bottom_center": {"x": 0.75, "y": 0.83}},
            "pitch_geometry": {
                "corridor": [
                    {"x": 0.1, "y": 0.7},
                    {"x": 0.5, "y": 0.7},
                    {"x": 0.5, "y": 0.95},
                    {"x": 0.1, "y": 0.95},
                ]
            },
        },
        calibration_v2=None,
        camera_pose=None,
        tracking={
            "primary_track": [
                {"frame_index": 10, "center": {"x": 360, "y": 210}},
                {"frame_index": 11, "center": {"x": 390, "y": 215}},
                {"frame_index": 12, "center": {"x": 420, "y": 220}},
            ]
        },
        bowler_pose_sequence=None,
    )

    assert context.bowler_pose_sequence is not None
    assert context.provenance["name"] == "rtmpose_mmpose"
    assert context.provenance["evidence_real"] is True
    assert context.provenance["bowling_arm"]["bowling_arm"] == "right"
