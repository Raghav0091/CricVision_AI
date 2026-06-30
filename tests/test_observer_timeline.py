"""Tests for observer timeline / detection quality summary."""

from Backends.src.agents.observer_timeline import build_observer_timeline


def test_observer_timeline_empty_detections():
    result = build_observer_timeline([], total_frames=10, fps=25)
    assert result["total_frames"] == 10
    assert result["processed_frames"] == 0
    assert result["ball_detected_frames"] == 0
    assert result["detection_quality"] == "Unknown"
    assert isinstance(result["observer_notes"], str)


def test_observer_timeline_counts_ball_and_bat_frames():
    timeline = [
        {
            "frame_index": 0,
            "ball_detections": [{"center": [10, 10], "confidence": 0.9}],
            "bat_detections": [{"center": [12, 12], "confidence": 0.8}],
            "stump_detections": [],
        },
        {
            "frame_index": 1,
            "ball_detections": [{"center": [15, 15], "confidence": 0.85}],
            "bat_detections": [],
            "stump_detections": [],
        },
    ]
    result = build_observer_timeline(timeline, total_frames=2, fps=30)
    assert result["ball_detected_frames"] == 2
    assert result["bat_detected_frames"] == 1
    assert result["ball_tracking_coverage"] == 100.0


def test_observer_timeline_detects_missing_ball_frames():
    timeline = [
        {"frame_index": 0, "ball_detections": [{"center": [10, 10], "confidence": 0.9}]},
        {"frame_index": 1, "ball_detections": []},
        {"frame_index": 2, "ball_detections": [{"center": [20, 20], "confidence": 0.9}]},
    ]
    result = build_observer_timeline(timeline, total_frames=3)
    assert result["missing_ball_frames"] == 1


def test_observer_timeline_detection_quality_safe():
    timeline = [
        {"frame_index": i, "ball_detections": [{"center": [10 + i, 10], "confidence": 0.9}]}
        for i in range(10)
    ]
    result = build_observer_timeline(timeline, total_frames=10)
    assert result["detection_quality"] in {"High", "Medium", "Low", "Unknown"}
