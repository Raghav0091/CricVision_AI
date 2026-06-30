"""Small performance-profile helpers shared by video analysis pipelines."""

from __future__ import annotations

import time


def create_performance_profile(
    speed_mode: str = "Smart Balanced",
    generate_processed_video: bool = True,
) -> dict:
    return {
        "video_read_time_sec": 0.0,
        "model_inference_time_sec": 0.0,
        "ball_detection_time_sec": 0.0,
        "bat_detection_time_sec": 0.0,
        "stump_detection_time_sec": 0.0,
        "annotation_write_time_sec": 0.0,
        "observer_timeline_time_sec": 0.0,
        "report_generation_time_sec": 0.0,
        "total_analysis_time_sec": 0.0,
        "frames_processed": 0,
        "frames_read": 0,
        "average_ms_per_processed_frame": None,
        "avg_time_per_frame_sec": None,
        "speed_mode": speed_mode,
        "smart_pipeline_used": True,
        "processed_video_generated": bool(generate_processed_video),
    }


def finish_performance_profile(
    profile: dict,
    started_at: float,
    frames_processed: int,
    processed_detection_frames: int = 0,
) -> dict:
    profile["frames_processed"] = frames_processed
    profile["total_analysis_time_sec"] = time.perf_counter() - started_at
    if frames_processed > 0:
        profile["avg_time_per_frame_sec"] = round(
            profile["total_analysis_time_sec"] / frames_processed,
            4,
        )
    if processed_detection_frames > 0:
        profile["average_ms_per_processed_frame"] = round(
            (profile["model_inference_time_sec"] / processed_detection_frames) * 1000,
            2,
        )
    return profile
