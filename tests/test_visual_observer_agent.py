"""Visual Observer wrapper tests require no models or video files."""

from Backends.src.agents.visual_observer_agent import run_visual_observer_repair


def test_visual_observer_returns_repaired_timeline_and_report():
    frames = [
        {
            "frame_index": 0,
            "ball_detections": [{"center": [10, 10], "confidence": 0.9}],
        },
        {"frame_index": 1, "ball_detections": []},
        {
            "frame_index": 2,
            "ball_detections": [{"center": [30, 10], "confidence": 0.9}],
        },
    ]

    result = run_visual_observer_repair(frames)

    assert {"frame_detections", "raw_frame_detections", "repair_report"} <= result.keys()
    assert {
        "original_coverage",
        "repaired_coverage",
        "missing_frames",
        "repaired_frames",
        "repair_confidence",
        "agent_decision",
        "notes",
    } <= result["repair_report"].keys()
    assert result["repair_report"]["repaired_frames"] == 1


def test_empty_visual_observer_report_is_low_confidence():
    result = run_visual_observer_repair([])

    assert result["frame_detections"] == []
    assert result["repair_report"]["repair_confidence"] == "Low"
    assert result["repair_report"]["notes"]
