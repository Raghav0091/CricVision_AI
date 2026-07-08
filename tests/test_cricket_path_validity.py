"""Unit tests for cricket path validity / safe trajectory segments (no YOLO/GPU/video)."""

from Backends.src.cricket_path_validity import (
    build_safe_trajectory_segments,
    normalize_points,
    points_as_xy,
    prepare_safe_trajectory_for_draw,
    validate_cricket_path,
)


def _smooth_delivery(count=12, start_y=80, step_y=28, x0=640):
    return [(x0 + (i % 3) - 1, start_y + i * step_y) for i in range(count)]


def test_empty_points_unavailable():
    result = validate_cricket_path([])
    assert result["is_valid"] is False
    assert result["quality"] == "Unavailable"
    assert result["valid_points"] == []
    assert result["main_rejection_reason"] == "too_few_points"


def test_few_points_poor_or_unavailable():
    result = validate_cricket_path([(10, 20), (12, 40), (14, 60)])
    assert result["is_valid"] is False
    assert result["quality"] in {"Unavailable", "Poor"}
    assert result["main_rejection_reason"] == "too_few_points"


def test_smooth_valid_path_good_or_partial():
    points = _smooth_delivery(12)
    result = validate_cricket_path(points, frame_size={"width": 1280, "height": 720})
    assert result["is_valid"] is True
    assert result["quality"] in {"Good", "Partial"}
    assert len(result["valid_points"]) >= 5
    assert "speed" not in result
    assert "swing" not in result
    assert "spin" not in result
    assert "lbw" not in result


def test_accepts_mixed_point_formats():
    points = [
        {"x": 640, "y": 100, "frame_index": 0},
        {"center": (642, 130)},
        (644, 160),
        [646, 190],
        {"x": 648, "y": 220, "frame_index": 4},
        {"x": 650, "y": 250},
        {"x": 652, "y": 280},
        {"x": 654, "y": 310},
    ]
    normalized = normalize_points(points)
    assert len(normalized) == 8
    result = validate_cricket_path(points, frame_size=(1280, 720))
    assert result["is_valid"] is True
    assert points_as_xy(result["valid_points"])


def test_rejects_out_of_bounds():
    points = _smooth_delivery(8)
    points[3] = (-50, points[3][1])
    result = validate_cricket_path(points, frame_size={"width": 1280, "height": 720})
    reasons = {item["reason"] for item in result["rejected_points"]}
    assert "out_of_bounds" in reasons


def test_rejects_huge_consecutive_jumps():
    points = _smooth_delivery(8)
    points[4] = (points[3][0] + 500, points[3][1] + 20)
    result = validate_cricket_path(points, frame_size={"width": 1280, "height": 720}, max_gap_px=80)
    reasons = {item["reason"] for item in result["rejected_points"]}
    assert "huge_jump" in reasons


def test_rejects_sudden_reversals():
    points = _smooth_delivery(8, start_y=100, step_y=30)
    # Force a large reverse jump opposite to progressive +y delivery.
    points[5] = (points[4][0], points[4][1] - 80)
    result = validate_cricket_path(points, frame_size={"width": 1280, "height": 720})
    reasons = {item["reason"] for item in result["rejected_points"]}
    assert "sudden_reversal" in reasons


def test_rejects_extreme_sideways_jumps():
    points = _smooth_delivery(8, start_y=100, step_y=10)
    points[4] = (points[3][0] + 120, points[3][1] + 5)
    result = validate_cricket_path(points, frame_size={"width": 1280, "height": 720})
    reasons = {item["reason"] for item in result["rejected_points"]}
    assert "extreme_sideways" in reasons


def test_rejects_far_outside_pitch_roi():
    points = _smooth_delivery(8, x0=640)
    points[3] = (50, points[3][1])
    roi = {"bbox": [500, 50, 780, 700]}
    result = validate_cricket_path(
        points,
        frame_size={"width": 1280, "height": 720},
        pitch_roi=roi,
    )
    reasons = {item["reason"] for item in result["rejected_points"]}
    assert "outside_pitch_corridor" in reasons


def test_never_crashes_on_garbage_input():
    result = validate_cricket_path(
        [None, "bad", {"x": "nope"}, object(), {"center": None}],
        frame_size="not-a-size",
        pitch_roi=123,
        stump_context="nope",
    )
    assert result["is_valid"] is False
    assert "quality" in result
    assert isinstance(result["notes"], list)


def test_draw_allowed_false_when_invalid():
    prepared = prepare_safe_trajectory_for_draw([(1, 1), (2, 2)])
    assert prepared["draw_allowed"] is False
    assert prepared["draw_segments"] == []
    assert prepared["ui_summary"]["draw_allowed"] is False


def test_partial_segments_skip_large_gaps():
    left = [(640, 80 + i * 20) for i in range(6)]
    right = [(640, 420 + i * 20) for i in range(6)]
    # Large spatial gap between clusters should not be bridged.
    valid = [{"x": x, "y": y} for x, y in (left + right)]
    segments = build_safe_trajectory_segments(
        valid,
        frame_size={"width": 1280, "height": 720},
        max_segment_gap_px=40,
    )
    assert segments["draw_allowed"] is True
    assert len(segments["segments"]) >= 2
    assert segments["fit_quality"] in {"Partial", "Good"}


def test_prepare_safe_trajectory_draws_only_safe_segments():
    points = _smooth_delivery(10)
    points.insert(5, (points[4][0] + 400, points[4][1] + 5))
    prepared = prepare_safe_trajectory_for_draw(
        points,
        frame_size={"width": 1280, "height": 720},
        max_gap_px=90,
    )
    assert prepared["quality"] in {"Good", "Partial", "Poor", "Unavailable"}
    if prepared["draw_allowed"]:
        assert prepared["draw_segments"]
        for segment in prepared["draw_segments"]:
            assert len(segment) >= 2
            for index in range(1, len(segment)):
                dx = segment[index][0] - segment[index - 1][0]
                dy = segment[index][1] - segment[index - 1][1]
                assert (dx * dx + dy * dy) ** 0.5 < 250


def test_no_fake_speed_swing_spin_lbw_fields():
    prepared = prepare_safe_trajectory_for_draw(_smooth_delivery(10))
    blob = str(prepared)
    for banned in ("speed_kmh", "swing", "spin", "lbw"):
        assert banned not in blob
