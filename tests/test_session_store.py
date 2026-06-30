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


def test_corrupt_json_does_not_crash(temp_session_store):
    temp_session_store.write_text("{not valid json", encoding="utf-8")
    results = load_session_results()
    assert results == []
    assert not temp_session_store.exists()
