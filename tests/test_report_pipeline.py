"""Report pipeline tests use synthetic detections and no model files."""

from Backends.src.video_pipeline.report_pipeline import build_video_reports


def _dummy_frame_detections():
    return [
        {
            "frame_index": 0,
            "ball_detections": [
                {
                    "center": [100, 100],
                    "confidence": 0.9,
                    "box": [95, 95, 105, 105],
                }
            ],
            "bat_detections": [
                {
                    "center": [108, 108],
                    "confidence": 0.85,
                    "box": [98, 98, 118, 118],
                }
            ],
            "stump_detections": [],
        },
        {
            "frame_index": 1,
            "ball_detections": [
                {
                    "center": [128, 99],
                    "confidence": 0.88,
                    "box": [123, 94, 133, 104],
                }
            ],
            "bat_detections": [],
            "stump_detections": [],
        },
    ]


def test_report_pipeline_returns_safe_result_keys():
    result = build_video_reports(
        _dummy_frame_detections(),
        fps=25,
        total_frames=2,
        batter_handedness="right",
        delivery_report={"estimated_line": "Middle"},
    )

    expected_keys = {
        "calibration_context",
        "visual_observer_repair",
        "observer_timeline",
        "impact_result",
        "shot_result",
        "direction_result",
        "outcome_result",
        "agent_result",
    }
    assert expected_keys <= result.keys()
    assert result["visual_observer_repair"]["repair_confidence"] in {"Low", "Medium", "High"}
    assert result["calibration_context"]["enabled"] is False
    assert result["observer_timeline"]["processed_frames"] == 2
    assert result["impact_result"]["impact_detected"] in {True, False}
    assert result["shot_result"]["shot_type"]
    assert result["direction_result"]["field_zone"]
    assert result["outcome_result"]["predicted_outcome"]
    assert result["agent_result"]["agent_quality"]


def test_report_pipeline_uses_repaired_timeline_for_reports():
    frames = _dummy_frame_detections()
    frames.insert(
        1,
        {
            "frame_index": 1,
            "ball_detections": [],
            "bat_detections": [],
            "stump_detections": [],
        },
    )
    frames[-1]["frame_index"] = 2

    result = build_video_reports(frames, fps=25, total_frames=3)

    repaired_detection = result["frame_detections"][1]["ball_detections"][0]
    assert repaired_detection["source"] == "observer_repair"
    assert result["visual_observer_repair"]["repaired_frames"] == 1
    assert result["observer_timeline"]["ball_tracking_coverage"] == 100.0


def test_report_pipeline_falls_back_to_raw_timeline_when_repair_fails(monkeypatch):
    def fail_repair(*args, **kwargs):
        raise RuntimeError("synthetic repair failure")

    monkeypatch.setattr(
        "Backends.src.agents.visual_observer_agent.repair_ball_tracking",
        fail_repair,
    )
    frames = _dummy_frame_detections()

    result = build_video_reports(frames, fps=25, total_frames=2)

    assert result["raw_frame_detections"] == frames
    assert [
        frame["ball_detections"] for frame in result["frame_detections"]
    ] == [
        frame["ball_detections"] for frame in frames
    ]
    assert result["visual_observer_repair"]["repair_confidence"] == "Low"
    assert "raw detections" in result["visual_observer_repair"]["agent_decision"]


def test_report_pipeline_normalizes_custom_calibration_context():
    result = build_video_reports(
        _dummy_frame_detections(),
        fps=25,
        total_frames=2,
        calibration_context={
            "enabled": True,
            "camera_view": "Umpire End",
            "batter_handedness": "Left-handed",
            "calibration_score": 0.75,
        },
    )

    calibration = result["calibration_context"]
    assert calibration["enabled"] is True
    assert calibration["camera_view"] == "umpire_end"
    assert calibration["batter_handedness"] == "left"
    assert calibration["calibration_quality"] == "Good"
