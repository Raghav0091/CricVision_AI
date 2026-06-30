"""Tests for rule-based outcome prediction."""

from Backends.src.analysis.outcome_prediction import predict_shot_outcome


def _timeline_with_path(impact_frame, points):
    frames = []
    for index, (cx, cy) in enumerate(points):
        frames.append(
            {
                "frame_index": impact_frame + index,
                "ball_detections": [
                    {
                        "center": [cx, cy],
                        "confidence": 0.9,
                        "box": [cx - 4, cy - 4, cx + 4, cy + 4],
                    }
                ],
            }
        )
    return frames


def test_no_impact_returns_unknown():
    result = predict_shot_outcome([], {}, {})
    assert result["predicted_outcome"] == "Unknown"


def test_small_movement_predicts_dot_ball():
    timeline = _timeline_with_path(0, [(100, 100), (105, 102), (108, 103)])
    impact = {"impact_detected": True, "impact_frame": 0, "ball_center": [100, 100]}
    shot = {"shot_type": "Defence", "shot_height": "Grounded", "shot_confidence": "Medium"}
    result = predict_shot_outcome(timeline, impact, shot)
    assert result["predicted_outcome"] == "Dot Ball"
    assert result["run_estimate"] == 0


def test_large_grounded_movement_can_predict_four():
    points = [(100, 100)]
    for step in range(1, 16):
        points.append((100 + step * 30, 100))
    timeline = _timeline_with_path(0, points)
    impact = {"impact_detected": True, "impact_frame": 0, "ball_center": [100, 100]}
    shot = {"shot_type": "Cover Drive", "shot_height": "Grounded", "shot_confidence": "High"}
    result = predict_shot_outcome(timeline, impact, shot)
    assert result["predicted_outcome"] in {"Four", "Three", "Double", "Single"}
    if result["predicted_outcome"] == "Four":
        assert result["boundary_chance"] == "High"


def test_aerial_shot_can_predict_six_or_caught_chance():
    points = [(100, 100)]
    for step in range(1, 12):
        points.append((100 + step * 20, 100 - step * 15))
    timeline = _timeline_with_path(0, points)
    impact = {"impact_detected": True, "impact_frame": 0, "ball_center": [100, 100]}
    shot = {"shot_type": "Lofted Shot", "shot_height": "Aerial", "shot_confidence": "High"}
    result = predict_shot_outcome(timeline, impact, shot)
    assert result["predicted_outcome"] in {"Six", "Caught Chance", "Unknown", "Four"}
