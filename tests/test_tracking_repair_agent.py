"""Tracking repair tests use only small synthetic detection timelines."""

from copy import deepcopy

from Backends.src.agents.tracking_repair_agent import repair_ball_tracking


def _frame(frame_index, center=None, confidence=0.9):
    detections = []
    if center is not None:
        x, y = center
        detections = [
            {
                "center": [x, y],
                "bbox": [x - 2, y - 2, x + 2, y + 2],
                "confidence": confidence,
            }
        ]
    return {
        "frame_index": frame_index,
        "ball_detections": detections,
        "bat_detections": [],
        "stump_detections": [],
    }


def test_stable_tracking_needs_no_repair_and_preserves_input():
    frames = [_frame(index, (100 + index * 5, 80)) for index in range(5)]
    original = deepcopy(frames)

    result = repair_ball_tracking(frames, frame_width=640, frame_height=360)

    assert result["repair_report"]["repaired_frames"] == 0
    assert result["repair_report"]["original_coverage"] == 100.0
    assert result["repair_report"]["repaired_coverage"] == 100.0
    assert frames == original


def test_short_missing_gap_is_interpolated_and_marked():
    frames = [
        _frame(1, (10, 20)),
        _frame(2, (20, 20)),
        _frame(3),
        _frame(4),
        _frame(5, (50, 20)),
    ]

    result = repair_ball_tracking(frames)
    repaired = {
        frame["frame_index"]: frame["ball_detections"]
        for frame in result["repaired_frame_detections"]
    }

    assert result["repair_report"]["repaired_frames"] == 2
    for frame_index in (3, 4):
        detection = repaired[frame_index][0]
        assert detection["repaired"] is True
        assert detection["source"] == "observer_repair"
        assert detection["trusted"] is True


def test_long_missing_gap_is_not_repaired():
    frames = [_frame(0, (10, 20))]
    frames.extend(_frame(index) for index in range(1, 6))
    frames.append(_frame(6, (70, 20)))

    result = repair_ball_tracking(frames, max_gap_frames=4)

    assert result["repair_report"]["repaired_frames"] == 0
    assert result["repair_report"]["missing_frames"] == 5


def test_isolated_impossible_jump_is_downgraded():
    frames = [
        _frame(0, (100, 100)),
        _frame(1, (110, 100)),
        _frame(2, (900, 700)),
        _frame(3, (130, 100)),
        _frame(4, (140, 100)),
    ]

    result = repair_ball_tracking(frames, frame_width=1000, frame_height=800)
    jump_frame = result["repaired_frame_detections"][2]
    suspicious = jump_frame["ball_detections"][0]

    assert suspicious["trusted"] is False
    assert suspicious["anomaly_type"] == "impossible_jump"
    assert result["repair_report"]["suspicious_detections"] == 1
    assert any(
        detection.get("source") == "observer_repair"
        for detection in jump_frame["ball_detections"]
    )


def test_empty_timeline_returns_safe_report():
    result = repair_ball_tracking([])

    assert result["repaired_frame_detections"] == []
    assert result["repair_report"]["original_coverage"] == 0.0
    assert result["repair_report"]["repaired_frames"] == 0
