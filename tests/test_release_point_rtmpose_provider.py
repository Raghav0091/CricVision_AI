from __future__ import annotations

from Backends.src.release_point.pose_provider import Keypoint, PoseProviderInfo, PosePerson
from Backends.src.release_point import rtmpose_provider
from Backends.src.release_point.rtmpose_provider import (
    RTMPoseProvider,
    RTMPoseProviderConfig,
    infer_bowling_arm,
)


class _FakeInferencer:
    def __call__(self, *_args, **_kwargs):
        yield {
            "predictions": [
                [
                    {
                        "keypoints": [
                            [0, 0],
                            [0, 0],
                            [0, 0],
                            [0, 0],
                            [0, 0],
                            [210, 270],
                            [250, 270],
                            [215, 300],
                            [280, 250],
                            [220, 330],
                            [300, 240],
                            [212, 360],
                            [245, 360],
                            [210, 470],
                            [250, 470],
                            [205, 590],
                            [255, 590],
                        ],
                        "keypoint_scores": [0.1] * 5 + [0.8] * 12,
                        "bboxes": [[190, 150, 330, 610]],
                        "bbox_scores": [0.91],
                    }
                ]
            ]
        }


def test_rtmpose_provider_normalizes_mmpose_inferencer_output():
    provider = RTMPoseProvider(RTMPoseProviderConfig(device="cpu"))
    provider._inferencer = _FakeInferencer()

    people = provider.estimate_frame(
        frame_bgr=object(),
        frame_index=12,
        timestamp_seconds=0.4,
    )

    assert len(people) == 1
    person = people[0]
    assert person.provider.name == "rtmpose_mmpose"
    assert person.provider.schema == "coco17"
    assert person.confidence == 0.91
    assert person.bbox_xyxy == (190.0, 150.0, 330.0, 610.0)
    assert person.keypoint("right_wrist").to_dict() == {
        "x": 300.0,
        "y": 240.0,
        "confidence": 0.8,
    }


def test_rtmpose_provider_availability_checks_optional_stack(monkeypatch):
    monkeypatch.setattr(
        rtmpose_provider,
        "find_spec",
        lambda package: object() if package != "mmpose" else None,
    )

    assert RTMPoseProvider.is_available() is False


def test_bowling_arm_heuristic_uses_temporal_wrist_activity():
    info = PoseProviderInfo("rtmpose_mmpose", "unit", "coco17")
    poses = {}
    for frame in range(4):
        poses[frame] = PosePerson(
            person_id="bowler",
            frame_index=frame,
            timestamp_seconds=frame / 30,
            bbox_xyxy=(100, 100, 300, 600),
            confidence=0.9,
            keypoints={
                "left_wrist": Keypoint(150 + frame, 250, 0.75),
                "right_wrist": Keypoint(220 + frame * 18, 230 - frame * 5, 0.85),
            },
            provider=info,
        )

    result = infer_bowling_arm(poses)

    assert result["bowling_arm"] == "right"
    assert result["confidence"] > 0.5
