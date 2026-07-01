"""Report orchestration over one shared normalized detection timeline."""

from __future__ import annotations

import time

from Backends.src.agents.observer_timeline import build_observer_timeline
from Backends.src.agents.visual_observer_agent import run_visual_observer_repair
from Backends.src.analysis.delivery_enrichment import run_post_shot_pipeline
from Backends.src.analysis.frame_detection_utils import normalize_frame_detections
from Backends.src.analysis.impact_detection import detect_bat_ball_impact
from Backends.src.analysis.shot_classification import classify_shot_type


def build_video_reports(
    frame_detections,
    *,
    fps=25,
    total_frames=None,
    batter_handedness=None,
    delivery_report=None,
    impact_result=None,
    shot_result=None,
    frame_width=None,
    frame_height=None,
    enable_visual_observer_repair=True,
) -> dict:
    """Build every report from the same timeline without running detection."""
    raw_frames = normalize_frame_detections(frame_detections)
    frame_count = len(raw_frames) if total_frames is None else int(total_frames)

    if enable_visual_observer_repair:
        visual_observer = run_visual_observer_repair(
            frame_detections,
            frame_width=frame_width,
            frame_height=frame_height,
            fps=fps,
        )
        frames = normalize_frame_detections(visual_observer["frame_detections"])
        raw_frame_detections = visual_observer["raw_frame_detections"]
        visual_observer_repair = visual_observer["repair_report"]
    else:
        frames = raw_frames
        raw_frame_detections = raw_frames
        visual_observer_repair = {
            "original_coverage": None,
            "repaired_coverage": None,
            "missing_frames": 0,
            "repaired_frames": 0,
            "removed_or_downgraded_frames": 0,
            "suspicious_detections": 0,
            "false_detection_candidates": 0,
            "repair_confidence": "Unknown",
            "agent_decision": "Visual Observer repair was not run for this analysis.",
            "notes": [],
        }

    observer_started_at = time.perf_counter()
    observer_timeline = build_observer_timeline(
        frames,
        total_frames=frame_count,
        fps=fps,
    )
    observer_timeline_time = time.perf_counter() - observer_started_at

    supplied_impact_result = impact_result
    impact_result = detect_bat_ball_impact(frames, fps=fps)
    _preserve_impact_metadata(impact_result, supplied_impact_result)
    shot_result = shot_result or classify_shot_type(
        frames,
        impact_result,
        batter_handedness=batter_handedness,
        fps=fps,
    )
    direction_result, outcome_result, agent_result, enrichment = run_post_shot_pipeline(
        frames,
        impact_result,
        shot_result,
        batter_handedness,
        fps,
        delivery_report=delivery_report,
    )
    return {
        "frame_detections": frames,
        "raw_frame_detections": raw_frame_detections,
        "visual_observer_repair": visual_observer_repair,
        "observer_timeline": observer_timeline,
        "impact_result": impact_result,
        "shot_result": shot_result,
        "direction_result": direction_result,
        "outcome_result": outcome_result,
        "agent_result": agent_result,
        "enrichment": enrichment,
        "observer_timeline_time_sec": observer_timeline_time,
    }


def timed_video_reports(frame_detections, **kwargs):
    started_at = time.perf_counter()
    result = build_video_reports(frame_detections, **kwargs)
    result["report_generation_time_sec"] = time.perf_counter() - started_at
    return result


def _preserve_impact_metadata(computed_result, supplied_result):
    """Keep UI-only impact metadata while analysis uses the repaired timeline."""
    if not isinstance(supplied_result, dict):
        return
    if supplied_result.get("impact_frame") == computed_result.get("impact_frame"):
        preview_path = supplied_result.get("impact_frame_image_path")
        if preview_path:
            computed_result["impact_frame_image_path"] = preview_path
    supplied_reason = str(
        supplied_result.get("reason")
        or supplied_result.get("impact_reason")
        or ""
    )
    if "unavailable" in supplied_reason.lower():
        computed_result["reason"] = supplied_reason
        computed_result["impact_reason"] = supplied_reason
