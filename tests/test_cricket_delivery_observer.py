from Backends.src.cricket_delivery_observer import (
    extract_ball_candidates_from_frame_detections,
    fit_observer_path,
    select_best_cricket_path,
)


def test_extract_ball_candidates_from_frame_detections_varied_shapes():
    frames = [
        {
            "frame_index": 0,
            "ball_detections": [
                {"center": [100, 200], "confidence": 0.8, "bbox": [95, 195, 105, 205]},
                {"box": [110, 210, 120, 220], "confidence": 0.6},
                {"xyxy": [130, 230, 140, 240], "confidence": 0.7},
                {"x1": 150, "y1": 250, "x2": 160, "y2": 260, "confidence": 0.9},
            ],
        }
    ]
    candidates = extract_ball_candidates_from_frame_detections(frames)
    assert len(candidates) == 4
    assert all(item["source"] == "raw_detection" for item in candidates)
    assert all("x" in item and "y" in item for item in candidates)


def test_extract_ball_candidates_empty_is_safe():
    assert extract_ball_candidates_from_frame_detections([]) == []
    assert extract_ball_candidates_from_frame_detections(None) == []


def test_selector_unavailable_for_too_few_candidates():
    candidates = [
        {"frame_index": idx, "x": 10 + idx, "y": 20 + idx, "confidence": 0.5}
        for idx in range(4)
    ]
    selected = select_best_cricket_path(candidates)
    assert selected["path_quality"] == "Unavailable"
    assert selected["observer_path"] == []


def test_selector_prefers_smooth_forward_path():
    smooth = [
        {"frame_index": i, "x": 200 + i * 2, "y": 100 + i * 14, "confidence": 0.75}
        for i in range(10)
    ]
    noisy = [
        {"frame_index": i, "x": 50 + (i % 2) * 120, "y": 120 + (i % 3) * 2, "confidence": 0.95}
        for i in range(10)
    ]
    selected = select_best_cricket_path(smooth + noisy, frame_size={"width": 1280, "height": 720})
    assert len(selected["observer_path"]) >= 5
    xs = [point["x"] for point in selected["observer_path"]]
    assert max(xs) - min(xs) < 120


def test_selector_rejects_backward_and_huge_jump_candidates():
    # Soft DP penalties skip bad alternatives when a smooth forward path exists.
    smooth = [
        {"frame_index": i, "x": 200 + i * 2, "y": 100 + i * 14, "confidence": 0.75}
        for i in range(8)
    ]
    bad = [
        {"frame_index": 3, "x": 180, "y": 40, "confidence": 0.99},  # backward
        {"frame_index": 5, "x": 900, "y": 650, "confidence": 0.99},  # huge jump
    ]
    selected = select_best_cricket_path(
        smooth + bad, frame_size={"width": 1280, "height": 720}
    )
    assert len(selected["observer_path"]) >= 5
    path_keys = {
        (point["frame_index"], point["x"], point["y"])
        for point in selected["observer_path"]
    }
    assert (3, 180.0, 40.0) not in path_keys
    assert (5, 900.0, 650.0) not in path_keys
    rejected_reasons = {item["reason"] for item in selected["rejected_candidates"]}
    assert "not_selected" in rejected_reasons


def test_fit_observer_path_safe_output():
    observer_path = [
        {"frame_index": i, "x": 300 + i * 2, "y": 100 + i * 15, "confidence": 0.8}
        for i in range(8)
    ]
    fitted = fit_observer_path(observer_path, frame_size={"width": 1280, "height": 720})
    assert fitted["fit_quality"] in {"Good", "Partial", "Poor"}
    assert len(fitted["fitted_path"]) == len(observer_path)
    assert all("x" in point and "y" in point for point in fitted["fitted_path"])


def test_observer_outputs_no_speed_swing_spin_lbw():
    candidates = [
        {"frame_index": i, "x": 200 + i * 2, "y": 100 + i * 14, "confidence": 0.7}
        for i in range(8)
    ]
    selected = select_best_cricket_path(candidates)
    fitted = fit_observer_path(selected.get("observer_path"))
    forbidden = {"speed_kmh", "swing", "spin", "lbw"}
    assert forbidden.isdisjoint(selected.keys())
    assert forbidden.isdisjoint(fitted.keys())
