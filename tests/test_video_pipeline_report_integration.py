"""Contract tests for all reports over one synthetic detection timeline."""

from Backends.src.video_pipeline.report_pipeline import build_video_reports


EXPECTED_REPORT_KEYS = {
    "calibration_context",
    "visual_observer_repair",
    "observer_timeline",
    "impact_result",
    "shot_result",
    "direction_result",
    "outcome_result",
    "agent_result",
}


def _detection(center, box_size=10, confidence=0.9):
    center_x, center_y = center
    half = box_size // 2
    return {
        "center": [center_x, center_y],
        "box": [
            center_x - half,
            center_y - half,
            center_x + half,
            center_y + half,
        ],
        "confidence": confidence,
    }


def _synthetic_detection_timeline():
    frames = []
    impact_frame = 8
    stump = _detection((285, 150), box_size=28, confidence=0.95)
    for frame_index in range(18):
        if frame_index <= impact_frame:
            ball_center = (55 + frame_index * 12, 135)
        else:
            ball_center = (
                55 + impact_frame * 12 + (frame_index - impact_frame) * 9,
                135 - (frame_index - impact_frame) * 7,
            )
        bat_detections = []
        if abs(frame_index - impact_frame) <= 2:
            bat_detections = [_detection((153, 140), box_size=42, confidence=0.86)]
        frames.append(
            {
                "frame_index": frame_index,
                "ball_detections": [_detection(ball_center)],
                "bat_detections": bat_detections,
                "stump_detections": [stump],
            }
        )
    return frames


def test_report_pipeline_contract_with_synthetic_timeline():
    result = build_video_reports(
        _synthetic_detection_timeline(),
        fps=25,
        total_frames=18,
        batter_handedness="right",
        delivery_report={"estimated_line": "Middle", "estimated_length": "Full"},
    )

    assert isinstance(result, dict)
    assert EXPECTED_REPORT_KEYS <= result.keys()
    assert result["observer_timeline"]["processed_frames"] == 18
    for key in EXPECTED_REPORT_KEYS:
        assert isinstance(result[key], dict)


def test_report_pipeline_handles_missing_detection_data():
    result = build_video_reports(
        [{"frame_index": 0}, {"frame_index": 1, "ball_detections": []}],
        fps=25,
        total_frames=2,
    )

    assert EXPECTED_REPORT_KEYS <= result.keys()
    assert result["observer_timeline"]["processed_frames"] == 2
