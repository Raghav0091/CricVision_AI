"""Trajectory replay visualization tests (no models, GPU, or video files)."""

from Backends.src.trajectory_replay import _metric_cards, build_trajectory_replay_image


def _fake_points(count: int = 8):
    return [(120 + index * 18, 80 + index * 35) for index in range(count)]


def test_build_trajectory_replay_image_saves_file(tmp_path):
    output_path = tmp_path / "replay.png"
    image = build_trajectory_replay_image(
        _fake_points(),
        bounce_point=(210, 260),
        health={
            "overall_tracking_quality": "Good",
            "ball_detection_rate": 42.5,
            "ball_tracking_rate": 0.51,
            "raw_ball_detections": 18,
            "selected_ball_points": 12,
        },
        output_path=output_path,
        width=720,
        height=1280,
    )
    assert image is not None
    assert output_path.is_file()
    assert image.shape == (1280, 720, 3)


def test_empty_trajectory_returns_none():
    assert build_trajectory_replay_image([]) is None
    assert build_trajectory_replay_image([(10, 20), (30, 40)]) is None


def test_missing_health_is_safe():
    image = build_trajectory_replay_image(_fake_points())
    assert image is not None
    assert image.shape[2] == 3


def test_dimensions_are_respected():
    image = build_trajectory_replay_image(_fake_points(), width=480, height=960)
    assert image is not None
    assert image.shape == (960, 480, 3)


def test_speed_swing_spin_are_not_numeric():
    cards = dict(
        _metric_cards(
            {
                "speed_kmh": 142.3,
                "swing_degrees": 12.5,
                "spin_rpm": 2200,
            }
        )
    )
    assert cards["Speed"] == "Not calibrated"
    assert cards["Swing"] == "Unknown"
    assert cards["Spin"] == "Unknown"
    assert "142" not in cards["Speed"]
    assert "12.5" not in cards["Swing"]
