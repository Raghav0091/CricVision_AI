from collections import Counter

from scripts import debug_ball_tracking as debug


def _candidate():
    return {
        "_debug_candidate_id": 0,
        "class_name": "ball",
        "confidence": 0.91,
        "box": (10, 20, 18, 28),
        "center": (14, 24),
    }


def test_candidate_rows_include_required_debug_columns():
    rows = debug.build_candidate_rows(
        frame_index=3,
        fps=25,
        width=640,
        height=360,
        model_path="Models/cricket_objects/best.pt",
        class_names={0: "ball", 1: "stump"},
        detection_source="primary",
        detection_ran=True,
        used_roi=True,
        roi_box=(0, 10, 200, 300),
        raw_ball_detections=[_candidate()],
        raw_stump_count=1,
        diagnostics=[
            {
                "candidate_id": 0,
                "selected": True,
                "rejected": False,
                "rejection_reason": "",
                "score": 1.25,
                "reference_center": (12, 23),
                "predicted_center": (15, 25),
            }
        ],
        previous_center=(12, 23),
        calibration_context={},
        accepted_track_count=2,
        tracking_quality="Medium",
    )

    assert set(debug.CSV_COLUMNS) == set(rows[0])
    assert rows[0]["candidate_selected"] is True
    assert rows[0]["inside_roi_or_corridor_if_available"] is True
    assert rows[0]["distance_from_previous"] > 0


def test_missing_candidate_row_keeps_frame_context():
    rows = debug.build_candidate_rows(
        frame_index=4,
        fps=25,
        width=640,
        height=360,
        model_path="Models/cricket_objects/best.pt",
        class_names={0: "ball", 1: "stump"},
        detection_source="primary",
        detection_ran=True,
        used_roi=False,
        roi_box=None,
        raw_ball_detections=[],
        raw_stump_count=0,
        diagnostics=[],
        previous_center=None,
        calibration_context={},
        accepted_track_count=0,
        tracking_quality="Poor",
    )

    assert rows[0]["raw_ball_candidate_count"] == 0
    assert rows[0]["rejection_reason"] == "no_raw_ball_candidates"
    assert rows[0]["candidate_selected"] is False


def test_parse_small_ball_diagnostic_options():
    args = debug.parse_args(
        [
            "sample.mp4",
            "--conf",
            "0.12",
            "--imgsz",
            "1280",
            "--max-frames",
            "25",
            "--every-nth-frame",
            "2",
            "--no-roi",
            "--speed-mode",
            "Debug Full Frame",
        ]
    )

    assert args.conf == 0.12
    assert args.imgsz == 1280
    assert args.max_frames == 25
    assert args.every_nth_frame == 2
    assert args.no_roi is True


def test_summary_includes_small_ball_diagnostic_settings():
    summary = debug.build_summary(
        video_path="sample.mp4",
        model_path="Models/cricket_objects/best.pt",
        class_names={0: "ball", 1: "stump"},
        total_frames=100,
        processed_frames=50,
        total_raw_ball_candidates=8,
        selected_ball_points=4,
        rejected_candidate_count=2,
        rejection_reasons=Counter(),
        stump_detections_count=20,
        calibration_context={"calibration_quality": "Disabled"},
        final_tracking_quality="Poor",
        selector_debug_summary={
            "tracking_quality": "Poor",
            "bootstrap_established": False,
            "delivery_track_found": False,
            "delivery_track_terminated": False,
            "selected_track_start_frame": None,
            "selected_track_end_frame": None,
            "selected_track_frame_count": 0,
            "short_track_reason": None,
            "trajectory_fit_quality": "Poor",
            "trajectory_visualization_mode": "hidden",
            "fitted_trajectory_point_count": 0,
            "observed_track_point_count": 0,
            "best_segment_start_frame": None,
            "best_segment_end_frame": None,
            "best_segment_point_count": 0,
            "selected_segment_score": 0.0,
        },
        timing={"total_time_sec": 1.0},
        roi_detected_frames=0,
        local_recovery_frames=0,
        kalman_predicted_frames=0,
        speed_settings={"mode": "Debug Full Frame"},
        confidence_threshold=0.12,
        ball_confidence_threshold=0.12,
        image_size=1280,
        full_frame_roi_mode="full_frame_no_roi",
        every_nth_frame=2,
    )

    assert summary["confidence_threshold_used"] == 0.12
    assert summary["image_size_used"] == 1280
    assert summary["full_frame_roi_mode"] == "full_frame_no_roi"
    assert summary["total_raw_ball_candidates"] == 8
    assert summary["selected_ball_points"] == 4
    assert summary["final_tracking_quality"] == "Poor"
    assert summary["trajectory_fit_quality"] == "Poor"
    assert summary["trajectory_visualization_mode"] == "hidden"
    assert summary["best_segment_point_count"] == 0
    assert summary["online_best_tracklet_enabled"] is False
    assert summary["best_tracklet_applied"] is False
    assert summary["best_tracklet_fallback_reason"] is None


def _det(center: tuple[int, int], confidence: float = 0.85) -> dict:
    x, y = center
    return {
        "center": center,
        "confidence": confidence,
        "box": (x - 5, y - 5, x + 5, y + 5),
        "class_name": "ball",
    }


def _moving_segment_candidates(start: int, end: int) -> dict[int, list[dict]]:
    return {
        index: [_det((150 + index * 20, 200 + index * 10))]
        for index in range(start, end)
    }


def test_apply_best_tracklet_post_pass_applies_ranked_segment():
    selector = debug.TrajectoryBallSelector(1280, 720)
    for frame_index in range(10, 16):
        offset = (frame_index - 10) * 8
        selector.select(
            [_det((200 + offset, 200 + offset))],
            frame_index=frame_index,
        )

    raw_candidates = {
        index: [_det((200, 200))]
        for index in range(9, 26)
    }
    raw_candidates.update(_moving_segment_candidates(40, 55))

    result = debug.apply_best_tracklet_post_pass(
        selector=selector,
        raw_frame_ball_candidates=raw_candidates,
        width=1280,
        height=720,
        fps=60.0,
        speed_mode="Debug Full Frame",
        ball_positions=[None] * 120,
        calibration_context={},
    )

    assert result["online_best_tracklet_enabled"] is True
    assert result["best_tracklet_applied"] is True
    assert result["best_segment_start_frame"] >= 40
    assert result["best_segment_point_count"] >= 3
    assert result["fit_result"] is not None
    assert result["online_selector_original_start_frame"] == 10
    assert result["online_selector_original_end_frame"] == 15
    assert selector.selected_track_start_frame() == result["best_segment_start_frame"]
    assert selector.selected_track_end_frame() == result["best_segment_end_frame"]


def test_summary_reports_best_tracklet_applied():
    summary = debug.build_summary(
        video_path="sample.mp4",
        model_path="Models/cricket_objects/best.pt",
        class_names={0: "ball", 1: "stump"},
        total_frames=100,
        processed_frames=50,
        total_raw_ball_candidates=8,
        selected_ball_points=15,
        rejected_candidate_count=2,
        rejection_reasons=Counter(),
        stump_detections_count=20,
        calibration_context={"calibration_quality": "Disabled"},
        final_tracking_quality="Medium",
        selector_debug_summary={
            "tracking_quality": "Medium",
            "bootstrap_established": True,
            "delivery_track_found": True,
            "delivery_track_terminated": True,
            "selected_track_start_frame": 40,
            "selected_track_end_frame": 54,
            "selected_track_frame_count": 15,
            "short_track_reason": None,
            "trajectory_fit_quality": "Good",
            "trajectory_visualization_mode": "fitted",
            "fitted_trajectory_point_count": 15,
            "observed_track_point_count": 15,
            "best_segment_start_frame": 40,
            "best_segment_end_frame": 54,
            "best_segment_point_count": 15,
            "best_segment_duration_sec": 0.25,
            "selected_segment_score": 12.5,
            "selected_segment_reason": "ranked_best_tracklet",
        },
        timing={"total_time_sec": 1.0},
        roi_detected_frames=0,
        local_recovery_frames=0,
        kalman_predicted_frames=0,
        speed_settings={"mode": "Debug Full Frame"},
        confidence_threshold=0.12,
        ball_confidence_threshold=0.12,
        image_size=1280,
        full_frame_roi_mode="full_frame_no_roi",
        every_nth_frame=2,
        best_tracklet_info={
            "online_best_tracklet_enabled": True,
            "best_tracklet_applied": True,
            "fallback_reason": None,
            "best_segment_score": 12.5,
            "best_segment_reason": "ranked_best_tracklet",
            "candidate_segment_count": 3,
            "rejected_static_segment_count": 1,
            "online_selector_original_start_frame": 10,
            "online_selector_original_end_frame": 12,
            "online_selector_after_track_terminated_count": 4,
        },
    )

    assert summary["best_tracklet_applied"] is True
    assert summary["online_best_tracklet_enabled"] is True
    assert summary["best_segment_score"] == 12.5
    assert summary["online_selector_original_start_frame"] == 10
    assert summary["online_selector_original_end_frame"] == 12


def test_apply_best_tracklet_post_pass_fallback_when_no_ranked_segment():
    selector = debug.TrajectoryBallSelector(640, 480)
    raw_candidates = {
        0: [_det((100, 100))],
        20: [_det((500, 500))],
    }

    result = debug.apply_best_tracklet_post_pass(
        selector=selector,
        raw_frame_ball_candidates=raw_candidates,
        width=640,
        height=480,
        fps=25.0,
        speed_mode="Debug Full Frame",
        ball_positions=[None] * 30,
        calibration_context={},
    )

    assert result["best_tracklet_applied"] is False
    assert result["fallback_reason"] == "no_valid_ranked_segment"
    assert result["online_selector_original_start_frame"] is None
    assert result["online_selector_original_end_frame"] is None


def test_debug_and_replay_share_select_online_best_tracklet():
    from Backends.src.tracking.trajectory_scorer import select_online_best_tracklet

    raw_candidates = _moving_segment_candidates(8, 20)
    shared = select_online_best_tracklet(raw_candidates, 640, 480, fps=25.0)

    selector = debug.TrajectoryBallSelector(640, 480)
    result = debug.apply_best_tracklet_post_pass(
        selector=selector,
        raw_frame_ball_candidates=raw_candidates,
        width=640,
        height=480,
        fps=25.0,
        speed_mode="Debug Full Frame",
        ball_positions=[None] * 30,
        calibration_context={},
    )

    assert shared["applied"] is True
    assert result["best_tracklet_applied"] is True
    assert result["best_segment_start_frame"] == shared["best_segment_start_frame"]
    assert result["best_segment_end_frame"] == shared["best_segment_end_frame"]
    assert result["best_segment_score"] == shared["best_segment_score"]


def test_build_debug_overlay_status_uses_final_tracking_not_poor():
    status = debug.build_debug_overlay_status(
        final_tracking_quality="Partial",
        fit_result={
            "trajectory_fit_quality": "Good",
            "trajectory_visualization_mode": "full_fit",
            "best_segment_start_frame": 60,
            "best_segment_end_frame": 103,
        },
        best_tracklet_info={
            "best_tracklet_applied": True,
            "best_segment_start_frame": 60,
            "best_segment_end_frame": 103,
        },
        calibration_context={"calibration_quality": "Disabled"},
        selector_debug_summary={"tracking_quality": "Poor"},
    )

    assert status["tracking_quality"] == "Partial"
    assert status["tracking_quality"] != "Poor"
    assert status["trajectory_fit_quality"] == "Good"
    assert status["track_label"] == "Track: Partial"
    assert status["fit_label"] == "Fit: Good"
    assert status["segment_label"] == "Segment: 60-103"
    assert status["online_selector_tracking_quality"] == "Poor"


def test_build_debug_overlay_status_keeps_unknown_metrics_without_calibration():
    status = debug.build_debug_overlay_status(
        final_tracking_quality="Partial",
        fit_result={"trajectory_fit_quality": "Good"},
        calibration_context={"calibration_quality": "Disabled"},
    )

    assert status["line"] == "Unknown"
    assert status["length"] == "Unknown"
    assert status["bounce"] == "Unknown"


def test_summary_includes_overlay_status():
    overlay_status = debug.build_debug_overlay_status(
        final_tracking_quality="Partial",
        fit_result={"trajectory_fit_quality": "Good"},
        best_tracklet_info={
            "best_tracklet_applied": True,
            "best_segment_start_frame": 60,
            "best_segment_end_frame": 103,
        },
        calibration_context={"calibration_quality": "Disabled"},
        selector_debug_summary={"tracking_quality": "Poor"},
    )
    summary = debug.build_summary(
        video_path="sample.mp4",
        model_path="Models/cricket_objects/best.pt",
        class_names={0: "ball"},
        total_frames=100,
        processed_frames=50,
        total_raw_ball_candidates=8,
        selected_ball_points=44,
        rejected_candidate_count=0,
        rejection_reasons=Counter(),
        stump_detections_count=0,
        calibration_context={"calibration_quality": "Disabled"},
        final_tracking_quality="Partial",
        selector_debug_summary={"tracking_quality": "Poor", "trajectory_fit_quality": "Poor"},
        timing={"total_time_sec": 1.0},
        roi_detected_frames=0,
        local_recovery_frames=0,
        kalman_predicted_frames=0,
        speed_settings={"mode": "Debug Full Frame"},
        confidence_threshold=0.12,
        ball_confidence_threshold=0.12,
        image_size=1280,
        full_frame_roi_mode="full_frame_no_roi",
        every_nth_frame=1,
        best_tracklet_info={"best_tracklet_applied": True},
        overlay_status=overlay_status,
    )

    assert summary["final_tracking_quality"] == "Partial"
    assert summary["trajectory_fit_quality"] == "Poor"
    assert summary["overlay_status"]["tracking_quality"] == "Partial"
    assert summary["overlay_status"]["trajectory_fit_quality"] == "Good"
    assert summary["overlay_status"]["fit_label"] == "Fit: Good"
    assert summary["overlay_status"]["online_selector_tracking_quality"] == "Poor"


def test_draw_debug_overlay_passes_final_tracking_quality(monkeypatch):
    import numpy as np

    captured: dict = {}

    def _capture_overlay(frame, **kwargs):
        captured.update(kwargs)
        return frame

    monkeypatch.setattr(
        debug,
        "draw_fitted_trajectory_overlay",
        _capture_overlay,
    )
    monkeypatch.setattr(debug, "draw_debug_status_labels", lambda *args, **kwargs: None)
    monkeypatch.setattr(debug, "draw_trajectory_lines", lambda *args, **kwargs: None)

    overlay_status = debug.build_debug_overlay_status(
        final_tracking_quality="Partial",
        fit_result={
            "trajectory_fit_quality": "Good",
            "trajectory_visualization_mode": "full_fit",
            "fitted_trajectory_points": [(10, 10), (20, 20)],
            "observed_trajectory_points": [(10, 10)],
        },
        best_tracklet_info={
            "best_tracklet_applied": True,
            "best_segment_start_frame": 60,
            "best_segment_end_frame": 103,
        },
        calibration_context={"calibration_quality": "Disabled"},
        selector_debug_summary={"tracking_quality": "Poor"},
    )
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    fit_result = {
        "trajectory_fit_quality": "Good",
        "trajectory_visualization_mode": "full_fit",
        "fitted_trajectory_points": [(10, 10), (20, 20)],
        "observed_trajectory_points": [(10, 10)],
    }
    debug.draw_debug_overlay(
        frame,
        [],
        [],
        [],
        [(10, 10), (20, 20)],
        fit_result=fit_result,
        roi_box=None,
        search_roi=None,
        overlay_status=overlay_status,
    )

    assert captured["tracking_quality"] == "Partial"
    assert captured["tracking_quality"] != "Poor"
    assert captured["line"] == "Unknown"
    assert captured["length"] == "Unknown"


def _rejected_detection():
    return {
        "_debug_candidate_id": 0,
        "class_name": "ball",
        "confidence": 0.42,
        "box": (100, 100, 120, 120),
        "center": (110, 110),
    }


def test_clean_best_tracklet_overlay_hides_online_rejections(monkeypatch):
    import numpy as np

    calls = {"rejected": 0, "ranked": 0}

    def _reject(*args, **kwargs):
        calls["rejected"] += 1

    def _ranked(*args, **kwargs):
        calls["ranked"] += 1

    monkeypatch.setattr(debug, "draw_rejected_candidate_debug", _reject)
    monkeypatch.setattr(debug, "draw_ranked_tracklet_point", _ranked)
    monkeypatch.setattr(debug, "draw_fitted_trajectory_overlay", lambda *a, **k: None)
    monkeypatch.setattr(debug, "draw_debug_status_labels", lambda *a, **k: None)
    monkeypatch.setattr(debug, "draw_trajectory_lines", lambda *a, **k: None)

    overlay_status = debug.build_debug_overlay_status(
        final_tracking_quality="Partial",
        fit_result={"trajectory_fit_quality": "Good"},
        best_tracklet_info={
            "best_tracklet_applied": True,
            "best_segment_start_frame": 60,
            "best_segment_end_frame": 103,
        },
        selector_debug_summary={"tracking_quality": "Poor"},
    )
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    debug.draw_debug_overlay(
        frame,
        [_rejected_detection()],
        [],
        [{"candidate_id": 0, "rejected": True, "rejection_reason": "after_track_terminated"}],
        [],
        fit_result={"trajectory_visualization_mode": "full_fit"},
        roi_box=None,
        search_roi=None,
        frame_index=60,
        label_rejections=False,
        overlay_status=overlay_status,
        ranked_frame_points={60: (110, 110)},
    )

    assert calls["rejected"] == 0
    assert calls["ranked"] == 1


def test_label_rejections_shows_dim_rejected_candidates(monkeypatch):
    import numpy as np

    calls = {"rejected": 0, "ranked": 0}

    monkeypatch.setattr(
        debug,
        "draw_rejected_candidate_debug",
        lambda *args, **kwargs: calls.__setitem__("rejected", calls["rejected"] + 1),
    )
    monkeypatch.setattr(
        debug,
        "draw_ranked_tracklet_point",
        lambda *args, **kwargs: calls.__setitem__("ranked", calls["ranked"] + 1),
    )
    monkeypatch.setattr(debug, "draw_fitted_trajectory_overlay", lambda *a, **k: None)
    monkeypatch.setattr(debug, "draw_debug_status_labels", lambda *a, **k: None)
    monkeypatch.setattr(debug, "draw_trajectory_lines", lambda *a, **k: None)

    overlay_status = debug.build_debug_overlay_status(
        final_tracking_quality="Partial",
        fit_result={"trajectory_fit_quality": "Good"},
        best_tracklet_info={"best_tracklet_applied": True},
        selector_debug_summary={"tracking_quality": "Poor"},
    )
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    debug.draw_debug_overlay(
        frame,
        [_rejected_detection()],
        [],
        [{"candidate_id": 0, "rejected": True, "rejection_reason": "after_track_terminated"}],
        [],
        fit_result={"trajectory_visualization_mode": "full_fit"},
        roi_box=None,
        search_roi=None,
        frame_index=60,
        label_rejections=True,
        overlay_status=overlay_status,
        ranked_frame_points={60: (110, 110)},
    )

    assert calls["rejected"] == 1
    assert calls["ranked"] == 1


def test_rejected_candidates_hidden_without_label_rejections(monkeypatch):
    import numpy as np

    calls = {"rejected": 0}

    monkeypatch.setattr(
        debug,
        "draw_rejected_candidate_debug",
        lambda *args, **kwargs: calls.__setitem__("rejected", calls["rejected"] + 1),
    )
    monkeypatch.setattr(debug, "draw_fitted_trajectory_overlay", lambda *a, **k: None)
    monkeypatch.setattr(debug, "draw_debug_status_labels", lambda *a, **k: None)
    monkeypatch.setattr(debug, "draw_trajectory_lines", lambda *a, **k: None)
    monkeypatch.setattr(debug, "draw_label", lambda *a, **k: None)

    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    debug.draw_debug_overlay(
        frame,
        [_rejected_detection()],
        [],
        [{"candidate_id": 0, "rejected": True, "rejection_reason": "after_track_terminated"}],
        [],
        fit_result={},
        roi_box=None,
        search_roi=None,
        label_rejections=False,
        overlay_status={"best_tracklet_applied": False},
    )

    assert calls["rejected"] == 0


def test_overlay_status_uses_final_ranked_quality_not_online_poor():
    status = debug.build_debug_overlay_status(
        final_tracking_quality="Partial",
        fit_result={"trajectory_fit_quality": "Good"},
        best_tracklet_info={
            "best_tracklet_applied": True,
            "best_segment_start_frame": 60,
            "best_segment_end_frame": 103,
        },
        selector_debug_summary={"tracking_quality": "Poor"},
    )

    assert status["tracking_quality"] == "Partial"
    assert status["trajectory_fit_quality"] == "Good"
    assert status["segment_label"] == "Segment: 60-103"
    assert status["online_selector_tracking_quality"] == "Poor"
    assert status["best_tracklet_applied"] is True


def test_apply_best_tracklet_post_pass_populates_extension_fields():
    tracklet, raw_candidates = _build_moving_segment_for_extension()
    selector = debug.TrajectoryBallSelector(1280, 720)

    result = debug.apply_best_tracklet_post_pass(
        selector=selector,
        raw_frame_ball_candidates=raw_candidates,
        width=1280,
        height=720,
        fps=25.0,
        speed_mode="Debug Full Frame",
        ball_positions=[None] * 120,
        calibration_context={},
    )

    assert result["best_tracklet_applied"] is True
    assert result["extension_enabled"] is True
    assert "backward_extension_points" in result
    assert "extension_rejection_reasons" in result
    assert "extension_preserved_original_segment" in result
    assert "trajectory_fit_quality_after_extension" in result
    assert result["extended_segment_point_count"] >= result["best_segment_point_count"]


def _build_moving_segment_for_extension():
    candidates = {}
    for frame_index in range(40, 55):
        center = (120 + frame_index * 15, 200 + frame_index * 8)
        candidates[frame_index] = [_det(center)]
    for frame_index in range(37, 40):
        center = (120 + frame_index * 15, 200 + frame_index * 8)
        candidates[frame_index] = [_det(center)]
    return [], candidates
