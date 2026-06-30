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
        "observer_timeline",
        "impact_result",
        "shot_result",
        "direction_result",
        "outcome_result",
        "agent_result",
    }
    assert expected_keys <= result.keys()
    assert result["observer_timeline"]["processed_frames"] == 2
    assert result["impact_result"]["impact_detected"] in {True, False}
    assert result["shot_result"]["shot_type"]
    assert result["direction_result"]["field_zone"]
    assert result["outcome_result"]["predicted_outcome"]
    assert result["agent_result"]["agent_quality"]
