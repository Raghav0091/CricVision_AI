import pandas as pd


MAX_INTERPOLATION_GAP = 8
EDGE_FILL_LIMIT = 2
MIN_BOUNCE_POINTS = 6


def _point_to_xy(point):
    if point is None:
        return None, None

    x, y = point
    return float(x), float(y)


def _to_int_point(x, y):
    if pd.isna(x) or pd.isna(y):
        return None

    return int(round(x)), int(round(y))


def interpolate_missing_positions(ball_positions):
    rows = [_point_to_xy(point) for point in ball_positions]
    positions_df = pd.DataFrame(rows, columns=["x", "y"])

    if positions_df.empty:
        return []

    positions_df[["x", "y"]] = positions_df[["x", "y"]].apply(
        pd.to_numeric,
        errors="coerce",
    )

    positions_df[["x", "y"]] = positions_df[["x", "y"]].interpolate(
        method="linear",
        limit=MAX_INTERPOLATION_GAP,
        limit_area="inside",
    )
    positions_df[["x", "y"]] = positions_df[["x", "y"]].ffill(limit=EDGE_FILL_LIMIT)
    positions_df[["x", "y"]] = positions_df[["x", "y"]].bfill(limit=EDGE_FILL_LIMIT)

    return [
        _to_int_point(row.x, row.y)
        for row in positions_df.itertuples(index=False)
    ]


def smooth_trajectory(trajectory_points, window_size=5):
    if not trajectory_points:
        return []

    interpolated_points = interpolate_missing_positions(trajectory_points)
    rows = [_point_to_xy(point) for point in interpolated_points]
    trajectory_df = pd.DataFrame(rows, columns=["x", "y"])

    if trajectory_df.empty:
        return []

    trajectory_df[["x", "y"]] = trajectory_df[["x", "y"]].apply(
        pd.to_numeric,
        errors="coerce",
    )

    window_size = max(1, int(window_size))
    trajectory_df[["x", "y"]] = (
        trajectory_df[["x", "y"]]
        .rolling(window=window_size, min_periods=1, center=True)
        .mean()
    )

    smoothed_points = []

    for original_point, row in zip(interpolated_points, trajectory_df.itertuples(index=False)):
        if original_point is None:
            smoothed_points.append(None)
        else:
            smoothed_points.append(_to_int_point(row.x, row.y))

    return smoothed_points


def detect_bounce_by_direction_change(trajectory_points):
    smoothed_points = smooth_trajectory(trajectory_points)
    valid_points = [
        (index, point)
        for index, point in enumerate(smoothed_points)
        if point is not None
    ]

    if len(valid_points) < MIN_BOUNCE_POINTS:
        return None

    lowest_position, (lowest_index, lowest_point) = max(
        enumerate(valid_points),
        key=lambda item: item[1][1][1],
    )
    y_by_index = {index: point[1] for index, point in valid_points}
    candidate_indices = []

    for index, point in valid_points[1:-1]:
        previous_y = y_by_index.get(index - 1)
        next_y = y_by_index.get(index + 1)

        if previous_y is None or next_y is None:
            continue

        moving_down_before = point[1] - previous_y >= 1
        moving_up_after = next_y - point[1] <= -1

        if moving_down_before and moving_up_after:
            candidate_indices.append(index)

    if candidate_indices:
        bounce_index = min(candidate_indices, key=lambda index: abs(index - lowest_index))
        return {
            "point": smoothed_points[bounce_index],
            "frame_index": bounce_index,
            "method": "direction_change",
        }

    if lowest_position < 2 or lowest_position > len(valid_points) - 3:
        return None

    return {
        "point": lowest_point,
        "frame_index": lowest_index,
        "method": "max_y_fallback",
    }


def calculate_tracking_quality(trajectory_points, total_frames):
    if total_frames <= 0:
        return {
            "detected_frames": 0,
            "interpolated_frames": 0,
            "usable_frames": 0,
            "tracking_rate": 0,
        }

    detected_frames = sum(1 for point in trajectory_points if point is not None)
    interpolated_positions = interpolate_missing_positions(trajectory_points)
    usable_frames = sum(1 for point in interpolated_positions if point is not None)
    interpolated_frames = max(usable_frames - detected_frames, 0)

    return {
        "detected_frames": detected_frames,
        "interpolated_frames": interpolated_frames,
        "usable_frames": usable_frames,
        "tracking_rate": (usable_frames / total_frames) * 100,
    }
