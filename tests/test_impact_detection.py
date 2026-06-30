"""Tests for bat-ball impact detection."""

from Backends.src.analysis.impact_detection import detect_bat_ball_impact


def test_no_detections_returns_not_detected():
    result = detect_bat_ball_impact([])
    assert result["impact_detected"] is False
    assert result["impact_frame"] is None
    assert result["impact_confidence"] == "Not Detected"


def test_ball_and_bat_close_returns_possible_impact():
    timeline = [
        {
            "frame_index": 5,
            "ball_detections": [
                {
                    "box": [100, 100, 110, 110],
                    "center": [105, 105],
                    "confidence": 0.9,
                }
            ],
            "bat_detections": [
                {
                    "box": [102, 102, 130, 130],
                    "center": [116, 116],
                    "confidence": 0.85,
                }
            ],
        }
    ]
    result = detect_bat_ball_impact(timeline, fps=25)
    assert result["impact_frame"] == 5
    assert result["min_ball_bat_distance_px"] is not None
    assert result["min_ball_bat_distance_px"] < 60


def test_missing_bat_does_not_crash():
    timeline = [
        {
            "frame_index": 0,
            "ball_detections": [{"center": [50, 50], "confidence": 0.9, "box": [45, 45, 55, 55]}],
            "bat_detections": [],
        }
    ]
    result = detect_bat_ball_impact(timeline)
    assert result["impact_detected"] is False
    assert "bat" in result["reason"].lower()


def test_missing_ball_does_not_crash():
    timeline = [
        {
            "frame_index": 0,
            "ball_detections": [],
            "bat_detections": [{"center": [50, 50], "confidence": 0.9, "bbox": [30, 30, 70, 70]}],
        }
    ]
    result = detect_bat_ball_impact(timeline)
    assert result["impact_detected"] is False
    assert "ball" in result["reason"].lower()
