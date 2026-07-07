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
