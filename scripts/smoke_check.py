#!/usr/bin/env python3
"""Lightweight smoke checks for CricVision AI (no models, camera, or video files)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _dummy_timeline():
    return [
        {
            "frame_index": 0,
            "ball_detections": [{"center": [100, 100], "confidence": 0.9, "box": [95, 95, 105, 105]}],
            "bat_detections": [{"center": [110, 110], "confidence": 0.8, "bbox": [100, 100, 120, 120]}],
            "stump_detections": [],
        },
        {
            "frame_index": 1,
            "ball_detections": [{"center": [130, 100], "confidence": 0.9, "box": [125, 95, 135, 105]}],
            "bat_detections": [],
            "stump_detections": [],
        },
    ]


def _dummy_repair_timeline():
    timeline = _dummy_timeline()
    timeline.insert(
        1,
        {
            "frame_index": 1,
            "ball_detections": [],
            "bat_detections": [],
            "stump_detections": [],
        },
    )
    timeline[-1]["frame_index"] = 2
    return timeline


def main() -> int:
    importlib.import_module("Backends.src.models.model_registry")
    importlib.import_module("Backends.src.models.model_loader")
    importlib.import_module("Backends.src.storage.session_store")
    importlib.import_module("Backends.src.video_pipeline.video_reader")
    importlib.import_module("Backends.src.video_pipeline.detection_pipeline")
    importlib.import_module("Backends.src.video_pipeline.report_pipeline")
    importlib.import_module("Backends.src.video_pipeline.annotation_writer")
    importlib.import_module("Backends.src.video_pipeline.performance_timer")

    from Backends.src.agents.observer_timeline import build_observer_timeline
    from Backends.src.agents.visual_observer_agent import run_visual_observer_repair
    from Backends.src.agents.vision_agent import run_vision_agent
    from Backends.src.analysis.impact_detection import detect_bat_ball_impact
    from Backends.src.analysis.outcome_prediction import predict_shot_outcome
    from Backends.src.analysis.shot_classification import classify_shot_type
    from Backends.src.analysis.shot_direction import estimate_shot_direction_zone
    from Backends.src.models.model_registry import MODEL_REGISTRY
    from Backends.src.storage.session_store import load_session_results, normalize_session_result
    from Backends.src.video_pipeline.report_pipeline import build_video_reports

    expected_keys = {
        "current_best",
        "cricshot_ball",
        "cricshot_bat",
        "player_type",
        "striker_segmentation",
        "shot_classifier",
    }
    missing = expected_keys - set(MODEL_REGISTRY)
    if missing:
        raise RuntimeError(f"Missing model registry keys: {sorted(missing)}")

    assert isinstance(load_session_results(), list)
    assert normalize_session_result({})["shot_type"] == "Unknown"

    timeline = _dummy_timeline()
    observer = build_observer_timeline(timeline, total_frames=2, fps=25)
    assert observer["processed_frames"] == 2
    repair = run_visual_observer_repair(_dummy_repair_timeline())
    assert repair["repair_report"]["repaired_frames"] > 0

    impact = detect_bat_ball_impact(timeline, fps=25)
    assert "impact_detected" in impact

    shot = classify_shot_type(timeline, impact, batter_handedness="right", fps=25)
    assert shot["shot_type"]

    direction = estimate_shot_direction_zone(timeline, impact, shot_result=shot, batter_handedness="right")
    assert direction["field_zone"]

    outcome = predict_shot_outcome(timeline, impact, shot, fps=25)
    assert outcome["predicted_outcome"]

    agent = run_vision_agent(
        timeline,
        delivery_report={"estimated_line": "Middle"},
        impact_result=impact,
        shot_result=shot,
        direction_result=direction,
        outcome_result=outcome,
        fps=25,
    )
    assert agent["agent_quality"]

    pipeline_reports = build_video_reports(
        timeline,
        fps=25,
        total_frames=2,
        batter_handedness="right",
        delivery_report={"estimated_line": "Middle"},
    )
    expected_report_keys = {
        "visual_observer_repair",
        "observer_timeline",
        "impact_result",
        "shot_result",
        "direction_result",
        "outcome_result",
        "agent_result",
    }
    assert expected_report_keys <= pipeline_reports.keys()
    assert pipeline_reports["observer_timeline"]["processed_frames"] == 2

    print("Smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
