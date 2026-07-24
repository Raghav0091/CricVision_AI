from __future__ import annotations

import pytest

from Backends.src.release_point.pose_provider import (
    CORE_KEYPOINTS,
    FakePoseProvider,
    Keypoint,
    PoseProvider,
    PoseProviderInfo,
    PosePerson,
    _validate_clean_original_video_path,
)


def test_fake_pose_provider_outputs_stable_contract():
    provider = FakePoseProvider(width=1280, height=720)

    sequence = provider.estimate_sequence("unused.mp4", range(3), fps=30.0)

    assert sequence.provider == provider.provider_info
    assert sequence.frame_indices() == [0, 1, 2]
    pose = sequence.persons_at(1)[0]
    assert pose.frame_index == 1
    assert pose.timestamp_seconds == pytest.approx(1 / 30)
    assert pose.provider.to_dict() == {
        "name": "fake_pose",
        "model": "deterministic_cricket_pose_v1",
        "schema": "cricvision_core_2d",
    }
    assert set(CORE_KEYPOINTS) <= set(pose.keypoints)
    assert {"x", "y", "confidence"} == set(pose.keypoint("right_wrist").to_dict())
    assert len(pose.bbox_xyxy) == 4
    assert 0 <= pose.confidence <= 1


def test_fake_pose_provider_is_deterministic():
    provider = FakePoseProvider(width=1280, height=720)

    first = provider.estimate_sequence("unused.mp4", range(5), fps=25.0).to_dict()
    second = provider.estimate_sequence("unused.mp4", range(5), fps=25.0).to_dict()

    assert first == second


def test_fake_pose_provider_preserves_custom_missing_keypoints():
    provider_info = PoseProviderInfo("custom", "unit", "partial")
    custom_pose = PosePerson(
        person_id="partial_person",
        frame_index=7,
        timestamp_seconds=0.28,
        bbox_xyxy=(10.0, 20.0, 80.0, 180.0),
        confidence=0.7,
        keypoints={"right_wrist": Keypoint(70.0, 40.0, 0.2)},
        provider=provider_info,
    )
    provider = FakePoseProvider(persons_by_frame={7: [custom_pose]})

    pose = provider.estimate_sequence("unused.mp4", [7], fps=25.0).persons_at(7)[0]

    assert pose.keypoints == {"right_wrist": Keypoint(70.0, 40.0, 0.2)}
    assert pose.provider.name == "fake_pose"


def test_base_pose_sequence_rejects_overlay_video_paths():
    class MinimalProvider(PoseProvider):
        provider_name = "minimal"
        model_name = "none"
        keypoint_schema = "none"

        def estimate_frame(self, frame_bgr, frame_index, timestamp_seconds):
            return []

    provider = MinimalProvider()

    with pytest.raises(ValueError, match="clean original"):
        provider.estimate_sequence(
            "outputs/video_analysis/analysis_1/tracking/tracking_debug.mp4",
            [0],
            fps=30.0,
        )

    with pytest.raises(ValueError, match="clean original"):
        provider.estimate_sequence(
            "outputs/video_analysis/analysis_1/replay/delivery_replay.mp4",
            [0],
            fps=30.0,
        )


def test_clean_original_path_validator_accepts_raw_analysis_path():
    _validate_clean_original_video_path(
        "outputs/video_analysis/analysis_1/raw/original_upload.mp4"
    )
