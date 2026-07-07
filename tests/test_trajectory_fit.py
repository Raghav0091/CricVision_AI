from Backends.src.tracking.trajectory_fit import fit_delivery_trajectory


def _points(coords, *, start_frame=0):
    return [
        {
            "frame_index": start_frame + index,
            "x": x,
            "y": y,
            "confidence": 0.9,
        }
        for index, (x, y) in enumerate(coords)
    ]


def _jump_sum(points):
    total = 0.0
    for index in range(1, len(points)):
        x1, y1 = points[index - 1]
        x2, y2 = points[index]
        total += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return total


def test_smooth_ten_point_segment_returns_partial_fit():
    points = _points([(100 + i * 10, 200 + i * 2) for i in range(10)])
    result = fit_delivery_trajectory(points, frame_size=(1280, 720), fps=25.0)
    assert result["trajectory_fit_quality"] == "Partial"
    assert result["trajectory_visualization_mode"] == "partial_fit"
    assert len(result["fitted_trajectory_points"]) >= 8
    assert result["best_segment_point_count"] == 10


def test_smooth_twenty_point_segment_returns_good_fit():
    points = _points([(100 + i * 8, 200 + i * 3) for i in range(20)])
    result = fit_delivery_trajectory(points, frame_size=(1280, 720), fps=25.0)
    assert result["trajectory_fit_quality"] == "Good"
    assert result["trajectory_visualization_mode"] == "full_fit"
    assert result["best_segment_point_count"] == 20


def test_first_tiny_segment_loses_to_later_long_smooth_segment():
    tiny = _points([(20, 20), (22, 21), (24, 21)], start_frame=0)
    longer = _points([(100 + i * 8, 200 + i * 2) for i in range(14)], start_frame=30)

    result = fit_delivery_trajectory(
        [*tiny, *longer],
        frame_size=(1280, 720),
        fps=25.0,
    )

    assert result["best_segment_start_frame"] == 30
    assert result["best_segment_end_frame"] == 43
    assert result["best_segment_point_count"] == 14
    assert result["selected_segment_score"] > 0


def test_noisy_ten_point_segment_is_smoothed():
    noisy = _points(
        [
            (100, 200),
            (110, 203),
            (120, 195),
            (130, 210),
            (140, 206),
            (150, 218),
            (160, 214),
            (170, 226),
            (180, 223),
            (190, 236),
        ]
    )
    result = fit_delivery_trajectory(noisy, frame_size=(1280, 720), fps=25.0)
    observed_jumps = _jump_sum(result["observed_trajectory_points"])
    fitted_jumps = _jump_sum(result["fitted_trajectory_points"][: len(result["observed_trajectory_points"])])
    assert fitted_jumps < observed_jumps


def test_duplicate_static_points_removed_before_fit():
    points = _points(
        [
            (100, 200),
            (100, 200),
            (101, 201),
            (101, 201),
            (112, 203),
            (124, 206),
        ]
    )
    result = fit_delivery_trajectory(points, frame_size=(1280, 720), fps=25.0)
    assert result["observed_point_count"] < len(points)


def test_outlier_does_not_badly_distort_fit():
    points = _points(
        [
            (100, 200),
            (110, 202),
            (120, 204),
            (500, 20),
            (130, 206),
            (140, 208),
            (150, 210),
        ]
    )
    result = fit_delivery_trajectory(points, frame_size=(1280, 720), fps=25.0)
    fit_points = result["fitted_trajectory_points"]
    assert all(abs(fit_points[i][0] - fit_points[i - 1][0]) <= 60 for i in range(1, len(fit_points)))


def test_too_few_points_returns_poor_without_fake_trajectory():
    result = fit_delivery_trajectory(
        _points([(100, 200), (110, 203)]),
        frame_size=(1280, 720),
        fps=25.0,
    )
    assert result["trajectory_fit_quality"] == "Poor"
    assert result["fitted_trajectory_points"] == []


def test_no_wild_extrapolation_outside_frame_bounds():
    points = _points([(1200 + i * 4, 680 + i * 2) for i in range(10)])
    result = fit_delivery_trajectory(points, frame_size=(1280, 720), fps=25.0)
    for x, y in result["fitted_trajectory_points"]:
        assert 0 <= x < 1280
        assert 0 <= y < 720
