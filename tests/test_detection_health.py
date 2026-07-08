"""Detection health helper tests (no models, GPU, or video files)."""

from pathlib import Path

from Backends.src.detection_health import (
    REQUIRED_HEALTH_KEYS,
    _parse_ball_detections_from_result,
    _sample_frame_indices,
    build_detection_health,
    classify_failure_type,
)


def test_build_detection_health_returns_required_keys():
    health = build_detection_health(
        {
            "total_frames": 120,
            "ball_detected_frames": 40,
            "total_ball_detections": 55,
            "ball_detection_rate": 33.3,
            "ball_tracking_rate": 52.0,
            "overall_tracking_quality": "Good",
            "kalman_predicted_frames": 2,
            "interpolated_ball_frames": 5,
            "tracker_recoveries": 1,
            "low_confidence_ball_frames": 3,
            "review_flags": ["low_ball_coverage"],
            "visual_observer_repair": {
                "repair_confidence": "Medium",
                "repaired_frames": 2,
            },
            "speed_mode": "Smart Balanced",
            "active_preset": "Balanced Mode",
            "ball_model_used": "Current Best Ball + Stump Model",
        },
        model_key="current_best",
        confidence_threshold=0.25,
        imgsz=768,
    )

    for key in REQUIRED_HEALTH_KEYS:
        assert key in health


def test_detector_failure_when_no_raw_detections():
    health = build_detection_health(
        {
            "total_frames": 100,
            "total_ball_detections": 0,
            "ball_detected_frames": 0,
            "ball_tracking_rate": 0,
        }
    )
    assert health["failure_type"] == "detector_failure"


def test_tracker_failure_when_raw_exists_but_few_selected_points():
    health = build_detection_health(
        {
            "total_frames": 100,
            "total_ball_detections": 12,
            "ball_detected_frames": 3,
            "ball_tracking_rate": 10,
        }
    )
    assert health["failure_type"] == "tracker_failure"


def test_partial_track_when_selected_points_exist_but_rate_low():
    health = build_detection_health(
        {
            "total_frames": 100,
            "total_ball_detections": 20,
            "ball_detected_frames": 12,
            "ball_tracking_rate": 30,
        }
    )
    assert health["failure_type"] == "partial_track"


def test_good_track_when_tracking_rate_acceptable():
    health = build_detection_health(
        {
            "total_frames": 100,
            "total_ball_detections": 30,
            "ball_detected_frames": 50,
            "ball_tracking_rate": 55,
        }
    )
    assert health["failure_type"] == "good_track"


def test_helper_handles_missing_keys():
    health = build_detection_health({})
    assert health["failure_type"] in {"unknown", "detector_failure"}
    assert health["model_name"] == "Unknown"
    assert health["review_flags"] == []
    assert isinstance(health["visual_observer_summary"], str)


def test_classify_failure_type_fraction_and_percent_rates():
    assert classify_failure_type(
        raw_ball_detections=10,
        selected_ball_points=8,
        ball_tracking_rate=0.50,
        total_frames=100,
    ) == "good_track"
    assert classify_failure_type(
        raw_ball_detections=10,
        selected_ball_points=8,
        ball_tracking_rate=50,
        total_frames=100,
    ) == "good_track"
    assert classify_failure_type(
        raw_ball_detections=0,
        selected_ball_points=0,
        ball_tracking_rate=None,
        total_frames=0,
    ) == "unknown"


def test_sample_frame_indices_evenly_spreads_samples():
    indices = _sample_frame_indices(120, 8)
    assert 6 <= len(indices) <= 12
    assert indices[0] == 0
    assert indices == sorted(indices)
    assert all(0 <= index < 120 for index in indices)


def test_parse_ball_detections_from_result_filters_non_ball_classes():
    class TensorLike:
        def __init__(self, value):
            self._value = value

        def cpu(self):
            return self

        def numpy(self):
            return self._value

    class Box:
        def __init__(self, class_id, confidence, xyxy):
            self.cls = [TensorLike(class_id)]
            self.conf = [TensorLike(confidence)]
            self.xyxy = [TensorLike(xyxy)]

    class FakeResult:
        def __init__(self, boxes):
            self.boxes = boxes

    class_names = {0: "ball", 1: "stump"}
    result = FakeResult(
        [
            Box(0, 0.82, [10, 20, 30, 40]),
            Box(1, 0.91, [50, 60, 70, 80]),
        ]
    )

    detections = _parse_ball_detections_from_result(result, class_names)
    assert len(detections) == 1
    assert detections[0]["class_id"] == 0
    assert detections[0]["class_name"] == "ball"
    assert detections[0]["confidence"] == 0.82
    assert detections[0]["bbox"] == (10, 20, 30, 40)
    assert detections[0]["center"] == (20, 30)


def test_benchmark_scaffold_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "benchmarks" / "README.md").is_file()
    assert (root / "benchmarks" / "expected_results.csv").is_file()
    assert (root / "benchmarks" / "clips").is_dir()
    assert (root / "benchmarks" / "notes").is_dir()

    header = (root / "benchmarks" / "expected_results.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "clip_name" in header
    assert "expected_tracking_quality" in header
