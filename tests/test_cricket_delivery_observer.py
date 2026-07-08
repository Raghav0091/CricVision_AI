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
    candidates = [
        {"frame_index": 0, "x": 200, "y": 100, "confidence": 0.8},
        {"frame_index": 1, "x": 202, "y": 112, "confidence": 0.8},
        {"frame_index": 2, "x": 204, "y": 126, "confidence": 0.8},
        {"frame_index": 3, "x": 206, "y": 141, "confidence": 0.8},
        {"frame_index": 4, "x": 208, "y": 156, "confidence": 0.8},
        {"frame_index": 5, "x": 210, "y": 130, "confidence": 0.8},  # backward
        {"frame_index": 6, "x": 500, "y": 600, "confidence": 0.8},  # huge jump
    ]
    selected = select_best_cricket_path(candidates, frame_size={"width": 1280, "height": 720})
    assert len(selected["observer_path"]) >= 5
    rejected_reasons = {item["reason"] for item in selected["rejected_candidates"]}
    assert rejected_reasons


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
