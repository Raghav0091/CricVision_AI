"""Trajectory-aware ball selection tests use only fake detections."""

from Backends.src.tracking.trajectory_scorer import TrajectoryBallSelector


def _ball(center, confidence=0.9):
    x, y = center
    return {
        "center": (x, y),
        "confidence": confidence,
        "box": (x - 2, y - 2, x + 2, y + 2),
    }


def test_reject_impossible_jump():
    selector = TrajectoryBallSelector(frame_width=640, frame_height=360)
    first = selector.select([_ball((100, 100), 0.9)])
    assert first is not None

    jumped = selector.select(
        [
            _ball((100, 105), 0.9),
            _ball((500, 300), 0.99),
        ],
        previous_center=first["center"],
    )

    assert jumped is not None
    assert jumped["center"] == (100, 105)
    assert selector.rejection_reasons["impossible_jump"] >= 1


def test_reject_static_false_candidate():
    selector = TrajectoryBallSelector(frame_width=640, frame_height=360)
    static_spot = (40, 40)

    selector.select([_ball((100, 100), 0.8)])
    selector.select(
        [_ball((110, 100), 0.75), _ball(static_spot, 0.95)],
        previous_center=(100, 100),
    )
    selector.select(
        [_ball((120, 100), 0.7), _ball(static_spot, 0.96)],
        previous_center=(110, 100),
    )
    chosen = selector.select(
        [_ball((130, 100), 0.65), _ball(static_spot, 0.97)],
        previous_center=(120, 100),
    )

    assert chosen is not None
    assert chosen["center"] == (130, 100)
    assert selector.rejection_reasons["static_false_positive"] >= 1


def test_prefer_smooth_lower_confidence_over_impossible_high_confidence():
    selector = TrajectoryBallSelector(frame_width=640, frame_height=360)
    selector.select([_ball((100, 100), 0.9)])

    chosen = selector.select(
        [
            _ball((112, 102), 0.45),
            _ball((620, 340), 0.98),
        ],
        previous_center=(100, 100),
    )

    assert chosen is not None
    assert chosen["center"] == (112, 102)
    assert chosen["confidence"] == 0.45


def test_poor_tracking_when_no_reliable_path_exists():
    selector = TrajectoryBallSelector(frame_width=640, frame_height=360)
    selector.select([_ball((100, 100), 0.9)])
    selector.select(
        [_ball((620, 340), 0.98)],
        previous_center=(100, 100),
    )
    selector.select(
        [_ball((10, 10), 0.97)],
        previous_center=(100, 100),
    )

    summary = selector.debug_summary("Poor")

    assert selector.has_reliable_track() is False
    assert summary["tracking_quality"] == "Poor"
    assert summary["accepted_ball_point_count"] == 1
    assert summary["rejected_candidate_count"] >= 2
    assert summary["main_rejection_reasons"]
