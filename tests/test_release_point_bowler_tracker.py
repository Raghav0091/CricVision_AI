from __future__ import annotations

from Backends.src.release_point.bowler_tracker import BowlerTracker
from Backends.src.release_point.pose_provider import (
    CORE_KEYPOINTS,
    Keypoint,
    PoseProviderInfo,
    PosePerson,
    PoseSequence,
)


PROVIDER = PoseProviderInfo("fake_pose", "unit", "cricvision_core_2d")


def _pose(
    person_id: str,
    frame_index: int,
    foot_x: float,
    foot_y: float = 600.0,
    *,
    confidence: float = 0.9,
    wrist_confidence: float = 0.8,
    missing: set[str] | None = None,
) -> PosePerson:
    missing = missing or set()
    cx = foot_x
    shoulder_y = foot_y - 330.0
    keypoints = {
        "left_shoulder": Keypoint(cx - 35, shoulder_y, 0.85),
        "right_shoulder": Keypoint(cx + 35, shoulder_y, 0.86),
        "left_elbow": Keypoint(cx - 45, shoulder_y + 55, 0.8),
        "right_elbow": Keypoint(cx + 55, shoulder_y - 20, 0.82),
        "left_wrist": Keypoint(cx - 55, shoulder_y + 110, wrist_confidence),
        "right_wrist": Keypoint(cx + 45, shoulder_y - 85, wrist_confidence),
        "left_hip": Keypoint(cx - 25, foot_y - 190, 0.84),
        "right_hip": Keypoint(cx + 25, foot_y - 190, 0.84),
        "left_knee": Keypoint(cx - 35, foot_y - 95, 0.78),
        "right_knee": Keypoint(cx + 35, foot_y - 100, 0.78),
        "left_ankle": Keypoint(cx - 45, foot_y, 0.76),
        "right_ankle": Keypoint(cx + 45, foot_y - 4, 0.76),
    }
    for name in missing:
        keypoints.pop(name, None)
    return PosePerson(
        person_id=person_id,
        frame_index=frame_index,
        timestamp_seconds=frame_index / 30.0,
        bbox_xyxy=(cx - 80, foot_y - 420, cx + 80, foot_y + 10),
        confidence=confidence,
        keypoints=keypoints,
        provider=PROVIDER,
    )


def _calibration() -> dict:
    return {
        "image_width": 1280,
        "image_height": 720,
        "non_striker_wicket": {"bottom_center": {"x": 0.25, "y": 0.83}},
        "striker_wicket": {"bottom_center": {"x": 0.75, "y": 0.83}},
        "pitch_geometry": {
            "corridor": [
                {"x": 0.18, "y": 0.72},
                {"x": 0.82, "y": 0.72},
                {"x": 0.82, "y": 0.95},
                {"x": 0.18, "y": 0.95},
            ]
        },
    }


def test_bowler_tracker_selects_bowler_from_multiple_people():
    frames = {}
    for frame_index in range(4):
        frames[frame_index] = [
            _pose("bowler_local", frame_index, 320 + frame_index * 12),
            _pose("batter", frame_index, 960),
            _pose("umpire", frame_index, 640, foot_y=250, confidence=0.88),
        ]
    sequence = PoseSequence(frames=frames, provider=PROVIDER)

    result = BowlerTracker().track(sequence, scene_calibration=_calibration())

    assert result.bowler_id == "bowler_01"
    assert result.frames == [0, 1, 2, 3]
    assert result.selection_confidence >= 0.55
    assert {pose.person_id for pose in result.poses_by_frame.values()} == {"bowler_01"}
    assert "bowling_end_assignment_uncertain" in result.quality_flags


def test_bowler_tracker_maintains_identity_when_provider_ids_change():
    frames = {
        frame_index: [_pose(f"provider_person_{frame_index}", frame_index, 320 + frame_index * 10)]
        for frame_index in range(5)
    }
    sequence = PoseSequence(frames=frames, provider=PROVIDER)

    result = BowlerTracker().track(sequence, scene_calibration=_calibration())

    assert result.bowler_id == "bowler_01"
    assert result.frames == [0, 1, 2, 3, 4]


def test_bowler_tracker_does_not_select_when_bowler_evidence_is_uncertain():
    frames = {
        frame_index: [
            _pose("person_a", frame_index, 300, confidence=0.86),
            _pose("person_b", frame_index, 900, confidence=0.86),
        ]
        for frame_index in range(3)
    }
    sequence = PoseSequence(frames=frames, provider=PROVIDER)

    result = BowlerTracker().track(sequence)

    assert result.bowler_id == "unknown"
    assert result.poses_by_frame == {}
    assert "insufficient_bowler_evidence" in result.quality_flags


def test_bowler_tracker_flags_missing_keypoints():
    sequence = PoseSequence(
        frames={
            frame_index: [
                _pose("bowler", frame_index, 320 + frame_index * 8, missing={"left_ankle"})
            ]
            for frame_index in range(4)
        },
        provider=PROVIDER,
    )

    result = BowlerTracker().track(sequence, scene_calibration=_calibration())

    assert result.bowler_id == "bowler_01"
    assert "missing_core_keypoints" in result.quality_flags
    assert "left_ankle" in CORE_KEYPOINTS


def test_bowler_tracker_flags_low_confidence_wrist():
    sequence = PoseSequence(
        frames={
            frame_index: [
                _pose("bowler", frame_index, 320 + frame_index * 8, wrist_confidence=0.2)
            ]
            for frame_index in range(4)
        },
        provider=PROVIDER,
    )

    result = BowlerTracker().track(sequence, scene_calibration=_calibration())

    assert result.bowler_id == "bowler_01"
    assert "low_confidence_wrist" in result.quality_flags


def test_bowler_tracker_keeps_continuity_across_missing_pose_frame():
    sequence = PoseSequence(
        frames={
            0: [_pose("bowler", 0, 320)],
            1: [_pose("bowler", 1, 330)],
            3: [_pose("bowler", 3, 350)],
            4: [_pose("bowler", 4, 360)],
        },
        provider=PROVIDER,
    )

    result = BowlerTracker().track(sequence, scene_calibration=_calibration())

    assert result.bowler_id == "bowler_01"
    assert result.frames == [0, 1, 3, 4]
    assert "missing_pose_frames" in result.quality_flags
