"""Tests for frame timeline helpers and detection box normalization."""

from Backends.src.analysis.analysis_speed import extract_detection_box, normalize_detections
from Backends.src.analysis.frame_detection_utils import (
    best_detection_center,
    calculate_point_distance,
    find_ball_center_at_or_before,
    normalize_frame_detections,
)


def test_normalize_frame_detections_sorts_and_preserves_types():
    timeline = [
        {"frame_index": 2, "balls": [{"center": [20, 20], "confidence": 0.5}]},
        {"frame_index": 0, "ball_detections": [{"center": [10, 10], "confidence": 0.9}]},
    ]
    frames = normalize_frame_detections(timeline)
    assert [item["frame_index"] for item in frames] == [0, 2]
    assert len(frames[0]["ball_detections"]) == 1
    assert len(frames[1]["ball_detections"]) == 1


def test_best_detection_center_from_box():
    center = best_detection_center([{"box": [90, 90, 110, 110], "confidence": 0.8}])
    assert center == (100.0, 100.0)


def test_best_detection_center_from_bbox():
    center = best_detection_center([{"bbox": [0, 0, 20, 40], "confidence": 0.7}])
    assert center == (10.0, 20.0)


def test_best_detection_center_from_xyxy():
    center = best_detection_center([{"xyxy": [50, 60, 70, 100], "confidence": 0.6}])
    assert center == (60.0, 80.0)


def test_best_detection_center_from_xy_fields():
    center = best_detection_center(
        [{"x1": 10, "y1": 20, "x2": 30, "y2": 60, "confidence": 0.95}]
    )
    assert center is None  # frame_detection_utils reads box/bbox only


def test_normalize_detections_handles_box_bbox_xyxy_and_xy_fields():
    stats = {"invalid_detection_count": 0}
    detections = normalize_detections(
        [
            {"box": [1, 2, 3, 4], "confidence": 0.5},
            {"bbox": [10, 20, 30, 40], "confidence": 0.6},
            {"xyxy": [5, 5, 15, 25], "confidence": 0.7},
            {"x1": 100, "y1": 110, "x2": 120, "y2": 140, "confidence": 0.8},
            {"confidence": 0.1},
        ],
        stats=stats,
    )
    assert len(detections) == 4
    assert stats["invalid_detection_count"] == 1
    assert detections[0]["box"] == (1, 2, 3, 4)
    assert detections[3]["box"] == (100, 110, 120, 140)


def test_extract_detection_box_formats():
    assert extract_detection_box({"box": [1, 2, 3, 4]})[1] == [1.0, 2.0, 3.0, 4.0]
    assert extract_detection_box({"bbox": [1, 2, 3, 4]})[0] == "bbox"
    assert extract_detection_box({"xyxy": [1, 2, 3, 4]})[0] == "xyxy"
    assert extract_detection_box({"x1": 1, "y1": 2, "x2": 3, "y2": 4})[0] == "xy_fields"
    assert extract_detection_box(None) == (None, None)


def test_find_ball_center_at_or_before():
    frames = normalize_frame_detections(
        [
            {"frame_index": 0, "ball_detections": [{"center": [10, 10], "confidence": 0.5}]},
            {"frame_index": 1, "ball_detections": [{"center": [20, 20], "confidence": 0.9}]},
            {"frame_index": 2, "ball_detections": []},
        ]
    )
    center = find_ball_center_at_or_before(frames, 1)
    assert center == (20.0, 20.0)


def test_invalid_detection_does_not_crash():
    assert best_detection_center([{"confidence": 0.5}]) is None
    assert calculate_point_distance(None, (1, 2)) is None
    assert normalize_frame_detections(["bad", 123]) == []
