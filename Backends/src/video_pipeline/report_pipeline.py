"""Report orchestration over one shared normalized detection timeline."""

from __future__ import annotations

import time

from Backends.src.agents.observer_timeline import build_observer_timeline
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
) -> dict:
    """Build every report from the same timeline without running detection."""
    frames = normalize_frame_detections(frame_detections)
    frame_count = len(frames) if total_frames is None else int(total_frames)

    impact_result = impact_result or detect_bat_ball_impact(frames, fps=fps)
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
    observer_started_at = time.perf_counter()
    observer_timeline = build_observer_timeline(
        frames,
        total_frames=frame_count,
        fps=fps,
    )
    return {
        "frame_detections": frames,
        "observer_timeline": observer_timeline,
        "impact_result": impact_result,
        "shot_result": shot_result,
        "direction_result": direction_result,
        "outcome_result": outcome_result,
        "agent_result": agent_result,
        "enrichment": enrichment,
        "observer_timeline_time_sec": time.perf_counter() - observer_started_at,
    }


def timed_video_reports(frame_detections, **kwargs):
    started_at = time.perf_counter()
    result = build_video_reports(frame_detections, **kwargs)
    result["report_generation_time_sec"] = time.perf_counter() - started_at
    return result
