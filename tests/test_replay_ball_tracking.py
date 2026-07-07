"""Tests for offline ball-tracking replay from debug CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from Backends.src.tracking.trajectory_scorer import rank_candidate_tracklets
from scripts.replay_ball_tracking_from_csv import (
    ReplayConfig,
    load_ball_candidates_from_csv,
    run_replay,
)


def _write_csv(path: Path, rows: list[dict], *, extra_fields: list[str] | None = None) -> None:
    fieldnames = [
        "frame_index",
        "timestamp_sec",
        "class_name",
        "confidence",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "center_x",
        "center_y",
        "frame_width",
        "frame_height",
        "candidate_id",
        "candidate_selected",
        "candidate_rejected",
        "rejection_reason",
        "tracking_quality_so_far",
    ]
    if extra_fields:
        fieldnames.extend(extra_fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _ball_row(
    frame_index: int,
    x: int,
    y: int,
    *,
    confidence: float = 0.8,
    timestamp: float | None = None,
    candidate_rejected: bool = False,
    rejection_reason: str = "",
    candidate_selected: bool = False,
) -> dict:
    return {
        "frame_index": frame_index,
        "timestamp_sec": timestamp if timestamp is not None else frame_index / 25.0,
        "class_name": "ball",
        "confidence": confidence,
        "bbox_x1": x - 5,
        "bbox_y1": y - 5,
        "bbox_x2": x + 5,
        "bbox_y2": y + 5,
        "center_x": x,
        "center_y": y,
        "frame_width": 640,
        "frame_height": 480,
        "candidate_id": f"{frame_index}-{x}-{y}",
        "candidate_selected": candidate_selected,
        "candidate_rejected": candidate_rejected,
        "rejection_reason": rejection_reason,
        "tracking_quality_so_far": "Poor",
    }


def _det(center: tuple[int, int], confidence: float = 0.85) -> dict:
    x, y = center
    return {
        "center": center,
        "confidence": confidence,
        "box": (x - 5, y - 5, x + 5, y + 5),
        "class_name": "ball",
    }


def test_replay_ignores_candidate_rejected_when_loading(tmp_path: Path) -> None:
    csv_path = tmp_path / "rejected.csv"
    rows = [
        _ball_row(40, 200, 300, candidate_rejected=True, rejection_reason="static_false_positive"),
        _ball_row(41, 220, 310, candidate_rejected=True, rejection_reason="after_track_terminated"),
        _ball_row(42, 240, 320, candidate_rejected=True, rejection_reason="far_from_track"),
    ]
    _write_csv(csv_path, rows)

    frame_candidates, metadata = load_ball_candidates_from_csv(csv_path)

    assert metadata["raw_candidate_count_used_for_replay"] == 3
    assert len(frame_candidates) == 3


def test_replay_ignores_after_track_terminated_rejection_reason(tmp_path: Path) -> None:
    csv_path = tmp_path / "terminated.csv"
    rows = [
        _ball_row(
            frame_index,
            200 + (frame_index - 50) * 20,
            300 + (frame_index - 50) * 10,
            candidate_rejected=True,
            rejection_reason="after_track_terminated",
        )
        for frame_index in range(50, 56)
    ]
    _write_csv(csv_path, rows)

    result = run_replay(csv_path, config=ReplayConfig(top_k=5))
    ranking = result["ranking"]

    assert ranking["top_segments"]
    assert ranking["top_segments"][0]["start_frame"] >= 50


def test_later_smooth_chain_ranked_even_if_online_rejected(tmp_path: Path) -> None:
    csv_path = tmp_path / "moving.csv"
    rows = [
        _ball_row(10, 200, 200, candidate_rejected=False, candidate_selected=True),
        _ball_row(11, 202, 201, candidate_rejected=False, candidate_selected=True),
        _ball_row(12, 203, 202, candidate_rejected=False, candidate_selected=True),
    ]
    for frame_index in range(40, 50):
        rows.append(
            _ball_row(
                frame_index,
                200 + (frame_index - 40) * 20,
                300 + (frame_index - 40) * 10,
                candidate_rejected=True,
                rejection_reason="after_track_terminated",
            )
        )
    _write_csv(csv_path, rows)

    result = run_replay(csv_path, config=ReplayConfig(top_k=5))
    winner = result["ranking"]["winner"]

    assert winner is not None
    assert winner["start_frame"] >= 40
    assert winner["end_frame"] >= 49


def test_small_frame_gaps_are_allowed() -> None:
    frame_candidates = {
        0: [_det((150, 200))],
        2: [_det((170, 215))],
        4: [_det((190, 230))],
        6: [_det((210, 245))],
    }

    result = rank_candidate_tracklets(
        frame_candidates,
        640,
        480,
        fps=25.0,
        max_frame_gap=5,
        min_tracklet_points=3,
    )

    assert result["top_segments"]
    assert result["top_segments"][0]["point_count"] >= 3


def test_static_repeated_points_are_penalised() -> None:
    static = {
        index: [_det((200, 200))]
        for index in range(5)
    }
    moving = {
        index: [_det((150 + index * 20, 200 + index * 10))]
        for index in range(10, 18)
    }

    static_result = rank_candidate_tracklets(static, 640, 480, fps=25.0, top_n=3)
    moving_result = rank_candidate_tracklets(moving, 640, 480, fps=25.0, top_n=3)
    combined = rank_candidate_tracklets({**static, **moving}, 640, 480, fps=25.0, top_n=5)

    assert static_result["winner"] is None
    assert static_result["rejected_static_segment_count"] >= 1
    assert moving_result["winner"] is not None
    assert moving_result["winner"]["static_penalty"] == 0.0
    assert combined["winner"] is not None
    assert combined["winner"]["start_frame"] >= 10


def test_81_point_static_segment_rejected_as_near_static() -> None:
    frame_candidates = {
        frame_index: [_det((348, 473))]
        for frame_index in range(116, 197)
    }

    result = rank_candidate_tracklets(frame_candidates, 1280, 720, fps=60.0, top_n=10)

    assert result["winner"] is None
    assert result["rejected_static_segment_count"] >= 1
    rejected = result["rejected_static_segments"][0]
    assert rejected["point_count"] == 81
    assert rejected["total_movement"] < 12.0
    assert rejected["rejection_reason"] in {
        "insufficient_total_movement",
        "insufficient_average_movement",
        "near_static_segment",
    }


def test_long_static_segment_cannot_beat_shorter_moving_segment() -> None:
    static_long = {
        frame_index: [_det((300, 400))]
        for frame_index in range(100, 190)
    }
    moving_short = {
        frame_index: [_det((200 + (frame_index - 40) * 18, 300 + (frame_index - 40) * 12))]
        for frame_index in range(40, 55)
    }

    result = rank_candidate_tracklets(
        {**static_long, **moving_short},
        1280,
        720,
        fps=60.0,
        top_n=5,
    )

    assert result["winner"] is not None
    assert result["winner"]["start_frame"] == 40
    assert result["winner"]["total_movement"] >= 12.0
    assert result["winner"]["point_count"] < 81


def test_repeated_same_center_positions_are_rejected() -> None:
    frame_candidates = {
        index: [_det((220, 330))]
        for index in range(20)
    }

    result = rank_candidate_tracklets(frame_candidates, 640, 480, fps=25.0, top_n=5)

    assert result["top_segments"] == []
    assert result["rejected_static_segment_count"] >= 1
    assert result["rejected_static_segments"][0]["unique_center_count"] == 1
    assert result["rejected_static_segments"][0]["rejection_reason"] in {
        "near_static_segment",
        "insufficient_total_movement",
        "insufficient_average_movement",
    }


def test_moving_segment_with_enough_movement_is_still_ranked() -> None:
    frame_candidates = {
        index: [_det((120 + index * 20, 200 + index * 10))]
        for index in range(6)
    }

    result = rank_candidate_tracklets(frame_candidates, 640, 480, fps=25.0, top_n=5)

    assert result["winner"] is not None
    assert result["winner"]["total_movement"] >= 12.0
    assert result["winner"]["reason"].startswith("selected_winner")


def test_replay_output_includes_rejected_static_segment_reason(tmp_path: Path) -> None:
    csv_path = tmp_path / "static_only.csv"
    rows = [_ball_row(index, 348, 473) for index in range(116, 130)]
    _write_csv(csv_path, rows)

    result = run_replay(csv_path, config=ReplayConfig(top_k=5))
    rejected = json.loads(
        Path(result["output_files"]["rejected_static_segments_json"]).read_text()
    )

    assert rejected
    assert rejected[0]["rejection_reason"] in {
        "insufficient_total_movement",
        "insufficient_average_movement",
        "near_static_segment",
    }
    assert result["summary"]["ranking"]["rejected_static_segment_count"] >= 1


def test_all_static_segments_explains_why_no_valid_winner() -> None:
    frame_candidates = {
        index: [_det((200, 200))]
        for index in range(10)
    }

    result = rank_candidate_tracklets(frame_candidates, 640, 480, fps=25.0, top_n=5)

    assert result["winner"] is None
    assert result["rejected_static_segment_count"] >= 1
    assert result["why_no_segments"] is not None
    assert result["why_no_segments"]["static_rejection_reason_counts"]


def test_ranked_segments_not_empty_for_clear_moving_chain(tmp_path: Path) -> None:
    csv_path = tmp_path / "clear_chain.csv"
    rows = [
        _ball_row(index, 120 + index * 20, 200 + index * 10)
        for index in range(8)
    ]
    _write_csv(csv_path, rows)

    result = run_replay(csv_path, config=ReplayConfig(top_k=5))
    ranked = json.loads(Path(result["output_files"]["ranked_segments_json"]).read_text())

    assert ranked
    assert ranked[0]["rank"] == 1
    assert ranked[0]["point_count"] >= 3


def test_why_no_segments_populated_when_no_segment_exists() -> None:
    frame_candidates = {
        0: [_det((100, 100), confidence=0.9)],
        10: [_det((400, 400), confidence=0.9)],
    }

    result = rank_candidate_tracklets(
        frame_candidates,
        640,
        480,
        max_frame_gap=2,
        max_link_distance_px=30.0,
        min_tracklet_points=3,
        top_n=5,
    )

    assert result["top_segments"] == []
    assert result["why_no_segments"] is not None
    assert result["why_no_segments"]["summary"]
    assert result["nearest_failed_segments"] is not None


def test_replay_parses_tiny_fake_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "tiny.csv"
    rows = [
        _ball_row(0, 100, 200),
        _ball_row(1, 105, 205),
        {"frame_index": 2, "timestamp_sec": 0.08, "class_name": "person", "center_x": 1, "center_y": 1},
        _ball_row(2, 110, 210),
    ]
    _write_csv(csv_path, rows)

    frame_candidates, metadata = load_ball_candidates_from_csv(csv_path)

    assert metadata["raw_candidate_count_used_for_replay"] == 3
    assert len(frame_candidates) == 3


def test_run_replay_writes_output_files(tmp_path: Path) -> None:
    csv_path = tmp_path / "replay.csv"
    rows = [_ball_row(index, 100 + index * 20, 200 + index * 10) for index in range(6)]
    _write_csv(csv_path, rows)

    result = run_replay(csv_path, config=ReplayConfig(top_k=5))

    for path in result["output_files"].values():
        assert Path(path).exists()

    ranked = json.loads(Path(result["output_files"]["ranked_segments_json"]).read_text())
    assert ranked
    assert ranked[0]["rank"] == 1
