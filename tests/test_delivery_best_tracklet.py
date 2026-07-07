"""Tests for online best-tracklet integration helpers."""

from __future__ import annotations

from Backends.src.tracking.trajectory_scorer import (
    TrajectoryBallSelector,
    build_ball_positions_from_tracklet,
    count_after_track_terminated_from_frame,
    select_online_best_tracklet,
    should_enable_online_best_tracklet,
)


def _det(center: tuple[int, int], confidence: float = 0.85) -> dict:
    x, y = center
    return {
        "center": center,
        "confidence": confidence,
        "box": (x - 5, y - 5, x + 5, y + 5),
        "class_name": "ball",
    }


def test_should_enable_for_accuracy_and_debug_full_frame() -> None:
    assert should_enable_online_best_tracklet(
        ball_tracking_mode="Accuracy / Small Ball",
        speed_mode="Smart Balanced",
    )
    assert should_enable_online_best_tracklet(
        ball_tracking_mode="Balanced",
        speed_mode="Debug Full Frame",
    )
    assert not should_enable_online_best_tracklet(
        ball_tracking_mode="Balanced",
        speed_mode="Smart Balanced",
    )


def test_ranked_segment_builds_ball_positions() -> None:
    frame_candidates = {
        index: [_det((150 + index * 20, 200 + index * 10))]
        for index in range(40, 55)
    }
    early = {
        index: [_det((200, 200))]
        for index in range(9, 26)
    }

    result = select_online_best_tracklet(
        {**early, **frame_candidates},
        1280,
        720,
        fps=60.0,
    )

    assert result["applied"] is True
    positions = build_ball_positions_from_tracklet(
        result["tracklet_points"],
        total_frames=120,
    )
    assert positions[40] is not None
    assert positions[54] is not None
    assert result["best_segment_start_frame"] >= 40


def test_after_track_terminated_applies_after_best_segment_end() -> None:
    frame_candidates = {
        index: [_det((150 + index * 20, 200 + index * 10))]
        for index in range(40, 55)
    }
    for index in range(56, 80):
        frame_candidates[index] = [_det((400, 400))]

    result = select_online_best_tracklet(
        frame_candidates,
        1280,
        720,
        fps=60.0,
    )
    assert result["applied"] is True
    end_frame = int(result["best_segment_end_frame"])

    after_count = count_after_track_terminated_from_frame(
        frame_candidates,
        result["tracklet_points"],
        frame_width=1280,
        frame_height=720,
        segment_end_frame=end_frame,
    )
    assert after_count > 0


def test_fallback_when_no_ranked_segment() -> None:
    frame_candidates = {
        0: [_det((100, 100))],
        20: [_det((500, 500))],
    }

    result = select_online_best_tracklet(
        frame_candidates,
        640,
        480,
        max_frame_gap=2,
        max_link_distance_px=20.0,
        min_tracklet_points=3,
    )

    assert result["applied"] is False
    assert result["fallback_reason"] == "no_valid_ranked_segment"


def test_apply_ranked_tracklet_updates_selector_state() -> None:
    selector = TrajectoryBallSelector(1280, 720)
    tracklet_points = [
        {"frame_index": 60, "x": 200, "y": 300},
        {"frame_index": 61, "x": 220, "y": 310},
        {"frame_index": 62, "x": 240, "y": 320},
    ]
    selector.apply_ranked_tracklet(tracklet_points, segment_end_frame=62)

    assert selector.accepted_point_count == 3
    assert selector.selected_track_start_frame() == 60
    assert selector.selected_track_end_frame() == 62
    assert selector._track_terminated is True


def test_replay_and_delivery_share_select_online_best_tracklet() -> None:
    frame_candidates = {
        index: [_det((120 + index * 20, 200 + index * 10))]
        for index in range(8)
    }
    shared = select_online_best_tracklet(frame_candidates, 640, 480, fps=25.0)
    ranking = shared["ranking"]

    assert shared["applied"] is True
    assert ranking["winner"]["start_frame"] == shared["best_segment_start_frame"]
    assert ranking["winner"]["final_segment_score"] == shared["best_segment_score"]
