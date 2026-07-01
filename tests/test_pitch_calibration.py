"""Pitch calibration tests use approximate frame geometry only."""

from Backends.src.calibration.pitch_calibration import (
    estimate_line_reference,
    estimate_pitch_corridor,
)


def _stump_reference():
    return {
        "bbox": [600, 300, 680, 450],
        "center": [640, 375],
        "confidence": 0.8,
        "source": "auto",
    }


def test_estimate_pitch_corridor_returns_valid_polygon_and_bbox():
    result = estimate_pitch_corridor(
        _stump_reference(),
        1280,
        720,
        camera_view="umpire_end",
    )
    corridor = result["pitch_corridor"]

    assert result["polygon"] == corridor["polygon"]
    assert result["bbox"] == corridor["bbox"]
    assert len(corridor["polygon"]) == 4
    assert len(corridor["bbox"]) == 4
    assert all(
        isinstance(value, (int, float))
        for point in corridor["polygon"]
        for value in point
    )
    assert result["pitch_ends"]["batter_end_y"] is not None


def test_estimate_line_reference_returns_three_stump_positions():
    result = estimate_line_reference(_stump_reference(), "right")

    assert result["off_stump_x"] < result["middle_stump_x"]
    assert result["middle_stump_x"] < result["leg_stump_x"]


def test_missing_stump_reference_returns_low_confidence_defaults():
    corridor = estimate_pitch_corridor(None, None, None)
    line_reference = estimate_line_reference(None)

    assert corridor["pitch_corridor"]["confidence"] <= 0.1
    assert len(corridor["pitch_corridor"]["polygon"]) == 4
    assert line_reference["source"] == "missing"
    assert line_reference["middle_stump_x"] is None
