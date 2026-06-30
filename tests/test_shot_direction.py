"""Tests for shot direction / field zone estimation."""

from Backends.src.analysis.shot_direction import estimate_shot_direction_zone


def _moving_timeline(impact_frame, start, end, frame_count=8):
    x0, y0 = start
    x1, y1 = end
    frames = []
    for index in range(frame_count):
        frame_index = impact_frame + index
        t = 0.0 if index == 0 else index / (frame_count - 1)
        cx = int(x0 + (x1 - x0) * t)
        cy = int(y0 + (y1 - y0) * t)
        frames.append(
            {
                "frame_index": frame_index,
                "ball_detections": [
                    {
                        "center": [cx, cy],
                        "confidence": 0.9,
                        "box": [cx - 5, cy - 5, cx + 5, cy + 5],
                    }
                ],
            }
        )
    return frames


def test_no_impact_returns_unknown():
    result = estimate_shot_direction_zone([], {}, shot_result={})
    assert result["field_zone"] == "Unknown"
    assert result["shot_direction"] == "Unknown"


def test_off_side_movement_for_right_hander():
    timeline = _moving_timeline(0, (100, 100), (180, 100))
    impact = {"impact_detected": True, "impact_frame": 0, "impact_confidence": "High"}
    shot = {"shot_type": "Cut Shot", "shot_confidence": "Medium"}
    result = estimate_shot_direction_zone(timeline, impact, shot_result=shot, batter_handedness="right")
    assert result["field_zone"] != "Unknown"
    assert result["shot_direction"] in {"Off Side", "Straight", "Behind Square", "Unknown"}


def test_leg_side_movement_for_right_hander():
    timeline = _moving_timeline(0, (200, 100), (120, 100))
    impact = {"impact_detected": True, "impact_frame": 0, "impact_confidence": "High"}
    shot = {"shot_type": "Flick", "shot_confidence": "Medium"}
    result = estimate_shot_direction_zone(timeline, impact, shot_result=shot, batter_handedness="right")
    assert result["field_zone"] != "Unknown"
    assert result["shot_direction"] in {"Leg Side", "Straight", "Behind Square", "Unknown"}


def test_missing_frame_data_does_not_crash():
    result = estimate_shot_direction_zone(
        [{"frame_index": 0, "ball_detections": []}],
        {"impact_detected": True, "impact_frame": 0},
        shot_result={"shot_type": "Defence"},
    )
    assert result["field_zone"] == "Unknown"
