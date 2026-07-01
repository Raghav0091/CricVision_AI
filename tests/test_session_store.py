"""Tests for local session JSON persistence."""

import json

from Backends.src.storage.session_store import (
    load_session_results,
    normalize_session_result,
    save_session_result,
)


def test_load_session_results_missing_file_returns_empty(temp_session_store):
    assert not temp_session_store.exists()
    assert load_session_results() == []


def test_normalize_session_result_handles_old_records():
    legacy = {"estimated_line": "Outside Off", "impact_info": {"impact_frame": 12}}
    normalized = normalize_session_result(legacy)
    assert normalized["line"] == "Outside Off"
    assert normalized["impact_frame"] == 12
    assert normalized["shot_type"] == "Unknown"
    assert normalized["smart_pipeline_used"] is False
    assert normalized["visual_observer_repair"] == {}
    assert normalized["calibration_context"]["enabled"] is False


def test_save_session_result_writes_lightweight_json(temp_session_store):
    saved = save_session_result(
        {
            "id": "test-id-1",
            "source_type": "Video Analysis",
            "video_name": "clip.mp4",
            "line": "Middle",
            "length": "Good Length",
        }
    )
    assert saved["id"] == "test-id-1"
    payload = json.loads(temp_session_store.read_text(encoding="utf-8"))
    assert isinstance(payload["results"], list)
    assert payload["results"][-1]["video_name"] == "clip.mp4"


def test_save_session_result_preserves_visual_observer_summary(temp_session_store):
    saved = save_session_result(
        {
            "id": "repair-id",
            "visual_observer_repair": {
                "original_coverage": 60.0,
                "repaired_coverage": 80.0,
                "missing_frames": 2,
                "repaired_frames": 2,
                "suspicious_detections": 1,
                "repair_confidence": "Medium",
                "agent_decision": "Short gaps were repaired.",
                "notes": ["2D repair only."],
            },
        }
    )

    assert saved["visual_observer_repair"]["repaired_frames"] == 2
    assert saved["visual_observer_repair"]["suspicious_detections"] == 1


def test_corrupt_json_does_not_crash(temp_session_store):
    temp_session_store.write_text("{not valid json", encoding="utf-8")
    results = load_session_results()
    assert results == []
    assert not temp_session_store.exists()


def test_save_session_result_preserves_calibration_context(temp_session_store):
    saved = save_session_result(
        {
            "id": "calibration-id",
            "calibration_context": {
                "enabled": True,
                "camera_view": "Bowler End",
                "batter_handedness": "right",
                "calibration_score": 0.75,
                "pitch_corridor": {
                    "polygon": [[1, 2], [3, 2], [4, 8], [0, 8]],
                    "bbox": [0, 2, 4, 8],
                    "source": "estimated",
                },
            },
        }
    )

    calibration = saved["calibration_context"]
    assert calibration["enabled"] is True
    assert calibration["camera_view"] == "bowler_end"
    assert calibration["calibration_quality"] in {"Low", "Medium"}
    assert len(calibration["pitch_corridor"]["polygon"]) == 4
