"""Unit tests for the physics-assisted trajectory fitter (no Streamlit/YOLO/video)."""

from Backends.src.physics_trajectory import (
    PROJECTED_PATH_NOTE,
    build_physics_trajectory_report,
    estimate_bounce_point,
    fit_physics_trajectory,
    normalize_trajectory_points,
    project_path_after_impact,
    split_pre_impact_path,
)

FRAME_SIZE = {"width": 1280, "height": 720}


def _smooth_delivery(count=12, start_y=80, step_y=28, x0=640):
    return [
        {"frame_index": i, "x": x0 + (i % 3) - 1, "y": start_y + i * step_y}
        for i in range(count)
    ]


def test_empty_points_returns_unavailable():
    result = fit_physics_trajectory([], frame_size=FRAME_SIZE)
    assert result["physics_quality"] == "Unavailable"
    assert result["fitted_path"] == []
    report = build_physics_trajectory_report([], frame_size=FRAME_SIZE)
    assert report["physics_quality"] == "Unavailable"
    assert report["fitted_delivery_path"] == []


def test_fewer_than_five_points_poor_or_unavailable():
    result = fit_physics_trajectory(_smooth_delivery(4), frame_size=FRAME_SIZE)
    assert result["physics_quality"] in {"Poor", "Unavailable"}
    assert result["fitted_path"] == []


def test_smooth_forward_path_fits_successfully():
    result = fit_physics_trajectory(_smooth_delivery(14), frame_size=FRAME_SIZE)
    assert result["physics_quality"] in {"Good", "Partial"}
    assert len(result["fitted_path"]) >= 5
    assert result["path_score"] > 0


def test_huge_sideways_outlier_is_rejected():
    points = _smooth_delivery(12)
    points[6] = {"frame_index": 6, "x": points[6]["x"] + 500, "y": points[6]["y"]}
    result = fit_physics_trajectory(points, frame_size=FRAME_SIZE)
    reasons = {item["reason"] for item in result["rejected_points"]}
    assert reasons & {"huge_jump", "impossible_sideways"}
    # Outlier must not drag the fitted path far sideways.
    assert all(abs(point["x"] - 640) < 120 for point in result["fitted_path"])


def test_impact_frame_splits_pre_and_post_points():
    points = _smooth_delivery(12)
    split = split_pre_impact_path(points, impact_frame=7)
    assert split["impact_detected"] is True
    assert split["impact_frame"] == 7
    assert [p["frame_index"] for p in split["pre_impact_points"]] == list(range(8))
    assert [p["frame_index"] for p in split["post_impact_observed_points"]] == list(range(8, 12))


def test_post_impact_points_not_used_in_delivery_fitting():
    points = _smooth_delivery(10)
    # Post-impact shot flies back up and away; must not appear in the fit.
    points += [
        {"frame_index": 10 + i, "x": 640 + i * 60, "y": 360 - i * 50}
        for i in range(1, 6)
    ]
    report = build_physics_trajectory_report(points, impact_frame=9, frame_size=FRAME_SIZE)
    assert len(report["post_impact_observed_points"]) == 5
    used_frames = {p.get("frame_index") for p in report["fitted_delivery_path"]}
    assert all(frame is None or frame <= 9 for frame in used_frames)


def test_projection_unavailable_with_too_few_points():
    result = project_path_after_impact(_smooth_delivery(3), impact_frame=2, frame_size=FRAME_SIZE)
    assert result["projection_quality"] == "Unavailable"
    assert result["projected_path"] == []


def test_projection_stays_inside_frame():
    # Path heading toward the bottom edge; projection must stop at the boundary.
    points = [{"frame_index": i, "x": 640, "y": 600 + i * 15} for i in range(8)]
    result = project_path_after_impact(points, impact_frame=7, frame_size=FRAME_SIZE)
    for point in result["projected_path"]:
        assert 0 <= point["x"] <= 1280
        assert 0 <= point["y"] <= 720
    if result["projected_path"]:
        assert PROJECTED_PATH_NOTE in result["notes"]


def test_bounce_estimate_does_not_overclaim():
    too_few = estimate_bounce_point(_smooth_delivery(3), frame_size=FRAME_SIZE)
    assert too_few["bounce_detected"] is False
    assert too_few["confidence"] in {"Unknown", "Low"}

    monotonic = estimate_bounce_point(_smooth_delivery(12), frame_size=FRAME_SIZE)
    assert monotonic["bounce_detected"] is False

    # Clear V-shape: ball drops then rises.
    bouncing = [
        {"frame_index": i, "x": 640, "y": 100 + i * 40 if i <= 6 else 340 - (i - 6) * 30}
        for i in range(12)
    ]
    bounced = estimate_bounce_point(bouncing, frame_size=FRAME_SIZE)
    if bounced["bounce_detected"]:
        assert bounced["confidence"] in {"Low", "Medium"}
        assert bounced["bounce_point"] is not None


def test_no_official_drs_speed_swing_spin_lbw_claims():
    points = _smooth_delivery(14)
    report = build_physics_trajectory_report(
        points,
        impact_frame=12,
        frame_size=FRAME_SIZE,
        pitch_roi={"bbox": [500, 50, 780, 700]},
    )
    blob = str(report).lower()
    for banned in ("speed_kph", "speed_kmh", "lbw", "drs", "swing", "spin", "hawkeye", "hawk-eye"):
        assert banned not in blob, f"report must not contain '{banned}'"


def test_normalize_handles_mixed_and_garbage_formats():
    points = [
        {"x": 640, "y": 100, "frame_index": 3},
        {"center": (642, 130)},
        {"centroid": {"x": 644, "y": 160}},
        {"bbox": [650, 180, 660, 190]},
        (646, 200),
        (5, 648, 230, 0.8),
        None,
        "bad",
        {"x": "nope", "y": 10},
        object(),
    ]
    normalized = normalize_trajectory_points(points)
    assert len(normalized) == 6
    for point in normalized:
        assert isinstance(point["x"], float)
        assert isinstance(point["y"], float)
    bbox_point = next(p for p in normalized if p["x"] == 655.0)
    assert bbox_point["y"] == 185.0
