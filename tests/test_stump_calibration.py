"""Stump calibration consumes synthetic detections and never runs YOLO."""

from Backends.src.calibration.stump_calibration import estimate_stump_reference


def test_estimate_stump_reference_uses_dummy_detection():
    result = estimate_stump_reference(
        frame_detections=[
            {
                "frame_index": 0,
                "stump_detections": [
                    {
                        "bbox": [100, 200, 140, 300],
                        "confidence": 0.85,
                    }
                ],
            }
        ],
        frame_width=640,
        frame_height=360,
    )
    stump = result["stump_reference"]

    assert result["bbox"] == stump["bbox"]
    assert result["confidence"] == stump["confidence"]
    assert stump["center"] == [120.0, 250.0]
    assert stump["source"] == "auto"
    assert stump["status"] == "detected"


def test_estimate_stump_reference_handles_empty_detections():
    result = estimate_stump_reference([], frame_width=640, frame_height=360)

    assert result["stump_reference"]["source"] == "estimated"
    assert result["stump_reference"]["confidence"] < 0.3


def test_estimate_stump_reference_handles_malformed_shapes():
    result = estimate_stump_reference(
        first_frame_detections=[
            None,
            {"bbox": ["bad", 2, 3, 4]},
            {"center": [1]},
            {"center": [float("nan"), 10]},
        ],
        frame_width="bad",
        frame_height=0,
    )

    assert result["stump_reference"]["status"] == "estimated"
    assert result["frame_width"] == 1280
    assert result["frame_height"] == 720
