"""Trajectory-aware ball selection tests use only fake detections."""

from Backends.src.tracking.trajectory_scorer import TrajectoryBallSelector


def _ball(center, confidence=0.9):
    x, y = center
    return {
        "center": (x, y),
        "confidence": confidence,
        "box": (x - 2, y - 2, x + 2, y + 2),
    }


def _bootstrap_moving_track(
    selector,
    start=(100, 100),
    step=(12, 2),
    count=3,
    *,
    start_frame=0,
    frame_step=1,
):
    previous = None
    chosen = None
    for index in range(count):
        center = (start[0] + step[0] * index, start[1] + step[1] * index)
        chosen = selector.select(
            [_ball(center, 0.8)],
            previous_center=previous,
            frame_index=start_frame + index * frame_step,
        )
        if chosen is not None:
            previous = chosen["center"]
    return chosen


def test_delivery_track_terminates_after_long_frame_gap():
    selector = TrajectoryBallSelector(frame_width=640, frame_height=360)
    _bootstrap_moving_track(selector, count=5, start_frame=0)

    accepted_before = selector.accepted_point_count
    assert accepted_before >= 5
    assert selector.has_reliable_track() is True

    selector.select([], frame_index=20)

    assert selector._track_terminated is True
    assert selector.accepted_point_count == accepted_before
    assert selector.rejection_reasons["delivery_track_lost"] >= 1


def test_static_cluster_after_termination_not_appended():
    selector = TrajectoryBallSelector(frame_width=640, frame_height=360)
    _bootstrap_moving_track(selector, count=5, start_frame=0)
    selector.select([], frame_index=20)
    static = (216, 183)

    for frame in range(30, 36):
        chosen = selector.select([_ball(static, 0.99)], frame_index=frame)
        assert chosen is None

    assert selector.accepted_point_count == 5
    assert selector.rejection_reasons["after_track_terminated"] >= 6


def test_far_cluster_after_termination_not_appended():
    selector = TrajectoryBallSelector(frame_width=640, frame_height=360)
    _bootstrap_moving_track(selector, count=5, start_frame=0)
    selector.select([], frame_index=20)
    far_centers = [(500, 300), (520, 310), (540, 320)]

    for index, center in enumerate(far_centers):
        chosen = selector.select(
            [_ball(center, 0.99)],
            frame_index=30 + index,
        )
        assert chosen is None

    assert selector.accepted_point_count == 5
    assert selector.rejection_reasons["after_track_terminated"] >= 3


def test_clean_short_delivery_path_stays_reliable():
    selector = TrajectoryBallSelector(frame_width=640, frame_height=360)
    centers = [(100, 100), (115, 102), (130, 105), (145, 108), (160, 110)]

    previous = None
    for index, center in enumerate(centers):
        chosen = selector.select(
            [_ball(center, 0.8)],
            previous_center=previous,
            frame_index=index,
        )
        if chosen is not None:
            previous = chosen["center"]

    summary = selector.debug_summary("Medium")

    assert selector.has_reliable_track() is True
    assert summary["trajectory_reliable"] is True
    assert summary["tracking_quality"] == "Medium"
    assert summary["accepted_ball_point_count"] == len(centers)
    assert selector._track_terminated is False


def test_termination_rejection_reasons_in_diagnostics():
    selector = TrajectoryBallSelector(frame_width=640, frame_height=360)
    _bootstrap_moving_track(selector, count=5, start_frame=0)
    selector.select([], frame_index=20)
    diagnostics = []

    selector.select(
        [_ball((216, 183), 0.99)],
        frame_index=30,
        diagnostics=diagnostics,
    )

    assert selector._track_terminated is True
    assert any(
        "after_track_terminated" in item["rejection_reason"]
        for item in diagnostics
    )
    assert (
        selector.rejection_reasons["delivery_track_lost"] >= 1
        or selector.rejection_reasons["after_track_terminated"] >= 1
    )


def test_reject_impossible_jump():
    selector = TrajectoryBallSelector(frame_width=640, frame_height=360)
    _bootstrap_moving_track(selector)

    jumped = selector.select(
        [
            _ball((112, 102), 0.9),
            _ball((500, 300), 0.99),
        ],
        previous_center=selector.accepted_positions[-1],
    )

    assert jumped is not None
    assert jumped["center"] == (112, 102)
    assert selector.rejection_reasons["impossible_jump"] >= 1


def test_reject_static_false_candidate():
    selector = TrajectoryBallSelector(frame_width=640, frame_height=360)
    static_spot = (40, 40)

    _bootstrap_moving_track(selector)
    selector.select(
        [_ball((136, 106), 0.75), _ball(static_spot, 0.95)],
        previous_center=selector.accepted_positions[-1],
    )
    selector.select(
        [_ball((148, 108), 0.7), _ball(static_spot, 0.96)],
        previous_center=selector.accepted_positions[-1],
    )
    chosen = selector.select(
        [_ball((160, 110), 0.65), _ball(static_spot, 0.97)],
        previous_center=selector.accepted_positions[-1],
    )

    assert chosen is not None
    assert chosen["center"] == (160, 110)
    assert selector.rejection_reasons["static_false_positive"] >= 1


def test_prefer_smooth_lower_confidence_over_impossible_high_confidence():
    selector = TrajectoryBallSelector(frame_width=640, frame_height=360)
    _bootstrap_moving_track(selector)

    chosen = selector.select(
        [
            _ball((136, 106), 0.45),
            _ball((620, 340), 0.98),
        ],
        previous_center=selector.accepted_positions[-1],
    )

    assert chosen is not None
    assert chosen["center"] == (136, 106)
    assert chosen["confidence"] == 0.45


def test_poor_tracking_when_no_reliable_path_exists():
    selector = TrajectoryBallSelector(frame_width=640, frame_height=360)
    selector.select([_ball((100, 100), 0.9)])
    selector.select([_ball((620, 340), 0.98)])
    selector.select([_ball((10, 10), 0.97)])

    summary = selector.debug_summary("Poor")

    assert selector.has_reliable_track() is False
    assert summary["tracking_quality"] == "Poor"
    assert summary["accepted_ball_point_count"] == 0
    assert summary["bootstrap_established"] is False
    assert summary["rejected_candidate_count"] >= 2
    assert summary["main_rejection_reasons"]


def test_identical_early_frames_do_not_bootstrap():
    selector = TrajectoryBallSelector(frame_width=640, frame_height=360)
    static = (216, 183)

    for _ in range(6):
        chosen = selector.select([_ball(static, 0.99)])
        assert chosen is None

    summary = selector.debug_summary("Poor")
    assert selector.has_reliable_track() is False
    assert summary["bootstrap_established"] is False
    assert summary["accepted_ball_point_count"] == 0


def test_static_high_confidence_first_frame_not_reliable_without_movement():
    selector = TrajectoryBallSelector(frame_width=640, frame_height=360)
    static = (40, 40)

    for _ in range(8):
        selector.select([_ball(static, 0.99)])

    assert selector.has_reliable_track() is False
    assert selector.accepted_point_count == 0
    assert selector.rejection_reasons["static_bootstrap_rejected"] >= 1


def test_two_static_early_candidates_do_not_establish_reliable_track():
    selector = TrajectoryBallSelector(frame_width=640, frame_height=360)
    static = (80, 80)

    assert selector.select([_ball(static, 0.96)]) is None
    assert selector.select([_ball(static, 0.95)]) is None

    summary = selector.debug_summary("Poor")
    assert selector.has_reliable_track() is False
    assert summary["bootstrap_established"] is False
    assert summary["bootstrap_provisional_point_count"] == 2
    assert summary["accepted_ball_point_count"] == 0


def test_two_static_plus_far_jump_do_not_bootstrap():
    selector = TrajectoryBallSelector(frame_width=640, frame_height=360)
    static = (216, 183)

    assert selector.select([_ball(static, 0.99)]) is None
    assert selector.select([_ball(static, 0.98)]) is None
    assert selector.select([_ball((345, 284), 0.97)]) is None

    summary = selector.debug_summary("Poor")
    assert summary["bootstrap_established"] is False
    assert summary["accepted_ball_point_count"] == 0
    assert selector.rejection_reasons["bootstrap_pending_no_valid_chain"] >= 3


def test_three_moving_candidates_establish_bootstrap():
    selector = TrajectoryBallSelector(frame_width=640, frame_height=360)
    centers = [(100, 100), (112, 102), (130, 105)]

    previous = None
    chosen = None
    for center in centers:
        chosen = selector.select([_ball(center, 0.8)], previous_center=previous)
        if chosen is not None:
            previous = chosen["center"]

    assert chosen is not None
    assert chosen["center"] == (130, 105)
    summary = selector.debug_summary("Poor")
    assert summary["bootstrap_established"] is True
    assert summary["accepted_ball_point_count"] == 3


def test_smooth_five_point_chain_establishes_bootstrap():
    selector = TrajectoryBallSelector(frame_width=640, frame_height=360)
    centers = [(100, 100), (115, 102), (130, 105), (145, 108), (160, 110)]

    previous = None
    for center in centers[:3]:
        chosen = selector.select([_ball(center, 0.8)], previous_center=previous)
        if chosen is not None:
            previous = chosen["center"]

    summary = selector.debug_summary("Poor")
    assert summary["bootstrap_established"] is True
    assert selector.accepted_point_count == 3
    assert selector.accepted_positions == centers[:3]


def test_edge_touching_candidates_are_suspicious_during_bootstrap():
    selector = TrajectoryBallSelector(frame_width=640, frame_height=360)
    diagnostics = []

    chosen = selector.select([_ball((1, 100), 0.99)], diagnostics=diagnostics)

    assert chosen is None
    assert selector.has_reliable_track() is False
    assert selector.rejection_reasons["bootstrap_edge_candidate"] == 1
    assert diagnostics[0]["rejected"] is True
    assert "bootstrap_edge_candidate" in diagnostics[0]["rejection_reason"]


def test_later_candidate_can_restart_after_two_static_bootstrap_points():
    selector = TrajectoryBallSelector(frame_width=640, frame_height=360)
    static = (40, 40)

    assert selector.select([_ball(static, 0.98)]) is None
    assert selector.select([_ball(static, 0.97)]) is None
    assert selector.select([_ball(static, 0.96)]) is None

    centers = [(180, 120), (195, 123), (210, 126)]
    previous = None
    later = None
    for center in centers:
        later = selector.select([_ball(center, 0.7)], previous_center=previous)
        if later is not None:
            previous = later["center"]

    assert later is not None
    assert later["center"] == (210, 126)
    assert selector.debug_summary("Poor")["bootstrap_established"] is True
    assert selector.rejection_reasons["bootstrap_pending_no_valid_chain"] >= 3


def test_moving_candidate_confirmed_after_short_movement():
    selector = TrajectoryBallSelector(frame_width=640, frame_height=360)
    centers = [(100, 100), (115, 100), (130, 100), (145, 100), (160, 100)]

    chosen = None
    for center in centers:
        chosen = selector.select([_ball(center, 0.8)])
        if center == centers[2]:
            assert chosen is not None
            assert chosen["center"] == center
        elif center in centers[:2]:
            assert chosen is None

    assert selector.accepted_point_count == len(centers)
    assert selector.has_reliable_track() is True


def test_bootstrap_established_false_when_trajectory_unreliable():
    selector = TrajectoryBallSelector(frame_width=640, frame_height=360)
    static = (216, 183)
    sequence = [
        static,
        static,
        (345, 284),
        (436, 282),
        (436, 282),
        (349, 282),
    ]

    for center in sequence:
        selector.select([_ball(center, 0.95)])

    summary = selector.debug_summary("Poor")
    assert summary["trajectory_reliable"] is False
    assert summary["bootstrap_established"] is False
    assert summary["accepted_ball_point_count"] == 0


def test_select_can_report_candidate_diagnostics():
    selector = TrajectoryBallSelector(frame_width=640, frame_height=360)
    _bootstrap_moving_track(selector)
    diagnostics = []

    chosen = selector.select(
        [
            _ball((136, 106), 0.45),
            _ball((620, 340), 0.98),
        ],
        previous_center=selector.accepted_positions[-1],
        diagnostics=diagnostics,
    )

    assert chosen["center"] == (136, 106)
    assert any(item["selected"] for item in diagnostics)
    assert any(
        item["rejected"] and "impossible_jump" in item["rejection_reason"]
        for item in diagnostics
    )
