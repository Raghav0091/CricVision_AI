"""Batting-focused delivery video processor."""

from __future__ import annotations

import time
from pathlib import Path

from Backends.src.calibration.calibration_context import (
    build_calibration_context,
)
from Backends.src.models.model_loader import get_cached_yolo_model
from Backends.src.models.model_registry import get_model_info
from Backends.src.utils.cv2_loader import cv2
from Backends.src.video_pipeline.annotation_writer import (
    add_impact_marker_to_video,
    draw_ball_detections,
    draw_clean_ball_markers,
    draw_trajectory_lines,
    ensure_frame_writer_size,
)
from Backends.src.video_pipeline.performance_timer import (
    create_performance_profile,
    finish_performance_profile,
)
from Backends.src.video_pipeline.report_pipeline import timed_video_reports


def process_batting_video(
    video_path,
    output_path,
    ball_model_key="current_best",
    bat_model_key="cricshot_bat",
    confidence=0.25,
    speed_mode="Smart Balanced",
    max_frames=None,
    generate_processed_video=True,
    calibration_context=None,
    overlay_detail="Clean",
    progress_callback=None,
):
    """Process a clip with only the models needed for batting intelligence."""
    from Backends.src.analysis.analysis_speed import (
        get_analysis_mode_settings,
        resize_frame_for_inference,
        scale_detections_to_original,
    )
    from Backends.src.analysis.bat_detection import (
        detect_ball_in_frame,
        detect_bat_in_frame,
        draw_bat_detections,
    )
    from Backends.src.analysis.impact_detection import (
        detect_bat_ball_impact,
        save_impact_frame_preview,
    )
    from Backends.src.analysis.smart_pipeline import (
        refine_bat_detections_near_impact,
        should_detect_ball,
        should_detect_bat,
        update_rough_impact_frame,
    )

    ball_model = get_cached_yolo_model(ball_model_key)
    bat_model = get_cached_yolo_model(bat_model_key)
    bat_unavailable_reason = ""
    if ball_model is None:
        return {
            "success": False,
            "error": "The selected ball model is unavailable.",
        }
    if bat_model is None:
        bat_unavailable_reason = (
            "Impact not detected: bat detection unavailable."
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"success": False, "error": "Could not open uploaded video."}
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    speed_settings = get_analysis_mode_settings(speed_mode)
    speed_mode = speed_settings.get("mode", speed_mode)
    resize_width = speed_settings.get("resize_width")
    debug_overlay = str(overlay_detail or "Clean").strip().lower() == "debug"
    generate_processed_video = bool(
        generate_processed_video
        and speed_settings.get("generate_processed_video", True)
    )
    performance = create_performance_profile()
    performance["speed_mode"] = speed_mode
    performance["smart_pipeline_used"] = True
    performance["processed_video_generated"] = generate_processed_video
    analysis_started = time.perf_counter()
    processed_detection_frames = 0
    detection_stats = {"invalid_detection_count": 0}
    if width <= 0 or height <= 0:
        cap.release()
        return {
            "success": False,
            "error": "Could not read video width/height.",
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = None
    if generate_processed_video:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            return {
                "success": False,
                "error": "Could not create output video writer.",
            }

    frame_index = 0
    ball_detected_frames = 0
    bat_detected_frames = 0
    impact_frame_detections = []
    frame_detections = impact_frame_detections
    impact_frame_candidates = {}
    trajectory = []
    rough_impact_state = None
    rough_impact_frame = None
    last_bat_detections = []

    while True:
        read_started = time.perf_counter()
        success, frame = cap.read()
        if not success:
            break
        if max_frames is not None and frame_index >= max_frames:
            break
        performance["video_read_time_sec"] += (
            time.perf_counter() - read_started
        )
        performance["frames_read"] += 1

        annotated_frame = frame.copy()
        ball_detections = []
        bat_detections = []
        inference_frame, detection_scale = resize_frame_for_inference(
            frame,
            resize_width,
        )

        if should_detect_ball(frame_index, speed_settings):
            ball_started = time.perf_counter()
            ball_detections = scale_detections_to_original(
                detect_ball_in_frame(
                    inference_frame,
                    ball_model,
                    confidence,
                ),
                detection_scale,
                stats=detection_stats,
            )
            ball_elapsed = time.perf_counter() - ball_started
            performance["ball_detection_time_sec"] += ball_elapsed
            performance["model_inference_time_sec"] += ball_elapsed
            processed_detection_frames += 1

        if (
            should_detect_bat(
                frame_index,
                speed_settings,
                rough_impact_frame,
            )
            and bat_model
        ):
            bat_started = time.perf_counter()
            bat_detections = scale_detections_to_original(
                detect_bat_in_frame(
                    inference_frame,
                    bat_model,
                    confidence,
                ),
                detection_scale,
                stats=detection_stats,
            )
            performance["bat_detection_time_sec"] += (
                time.perf_counter() - bat_started
            )
            last_bat_detections = bat_detections
        elif last_bat_detections:
            bat_detections = last_bat_detections

        rough_impact_state = update_rough_impact_frame(
            frame_index,
            ball_detections,
            bat_detections,
            rough_impact_state,
        )
        if rough_impact_state is not None:
            rough_impact_frame = rough_impact_state[1]

        if ball_detections:
            ball_detected_frames += 1
            main_ball = max(
                ball_detections,
                key=lambda item: item["confidence"],
            )
            trajectory.append(tuple(main_ball["center"]))
        if bat_detections:
            bat_detected_frames += 1
        impact_frame_detections.append(
            {
                "frame_index": frame_index,
                "ball_detections": ball_detections,
                "bat_detections": bat_detections,
                "stump_detections": [],
            }
        )
        if ball_detections and bat_detections:
            impact_frame_candidates[frame_index] = frame.copy()

        if debug_overlay:
            draw_ball_detections(annotated_frame, ball_detections)
            draw_bat_detections(annotated_frame, bat_detections)
        else:
            draw_clean_ball_markers(annotated_frame, ball_detections)

        if debug_overlay:
            for _ in range(1, len(trajectory[-35:])):
                draw_trajectory_lines(
                    annotated_frame,
                    trajectory[-35:],
                )
        elif len(trajectory) >= 2:
            draw_trajectory_lines(annotated_frame, trajectory[-2:])

        if writer is not None:
            annotation_started = time.perf_counter()
            writer.write(
                ensure_frame_writer_size(
                    annotated_frame,
                    width,
                    height,
                )
            )
            performance["annotation_write_time_sec"] += (
                time.perf_counter() - annotation_started
            )
        frame_index += 1
        _report_progress(progress_callback, frame_index, total_frames)

    cap.release()
    if writer is not None:
        writer.release()
    if frame_index == 0:
        return {
            "success": False,
            "error": "No video frames were processed.",
        }

    preliminary_impact = detect_bat_ball_impact(frame_detections, fps=fps)
    if (
        speed_settings.get("refine_bat_near_impact")
        and preliminary_impact.get("impact_frame") is not None
    ):
        refine_bat_detections_near_impact(
            video_path,
            frame_detections,
            preliminary_impact.get("impact_frame"),
            bat_model,
            confidence,
            resize_width,
            radius=int(speed_settings.get("impact_window_radius", 8)),
            stats=detection_stats,
        )

    impact_info = detect_bat_ball_impact(
        impact_frame_detections,
        fps=fps,
    )
    if bat_unavailable_reason:
        impact_info["reason"] = bat_unavailable_reason
        impact_info["impact_reason"] = bat_unavailable_reason
    impact_frame = impact_info.get("impact_frame")
    if impact_frame is not None:
        preview_path = save_impact_frame_preview(
            impact_frame_candidates.get(impact_frame),
            impact_info,
            prefix=f"batting_impact_{Path(output_path).stem}",
        )
        if preview_path is not None:
            impact_info["impact_frame_image_path"] = str(preview_path)
    if generate_processed_video:
        add_impact_marker_to_video(output_path, impact_info)
    calibration_context = build_calibration_context(
        calibration_context,
        frame_detections=frame_detections,
        frame_width=width,
        frame_height=height,
    )
    report_handedness = calibration_context.get("batter_handedness")
    if report_handedness == "unknown":
        report_handedness = None
    reports = timed_video_reports(
        frame_detections,
        fps=fps,
        total_frames=frame_index,
        batter_handedness=report_handedness,
        impact_result=impact_info,
        frame_width=width,
        frame_height=height,
        calibration_context=calibration_context,
    )
    impact_info = reports["impact_result"]
    shot_info = reports["shot_result"]
    direction_info = reports["direction_result"]
    outcome_info = reports["outcome_result"]
    agent_info = reports["agent_result"]
    enrichment = reports["enrichment"]
    observer_timeline = reports["observer_timeline"]
    visual_observer_repair = reports["visual_observer_repair"]
    performance["report_generation_time_sec"] = reports[
        "report_generation_time_sec"
    ]
    performance["observer_timeline_time_sec"] = reports[
        "observer_timeline_time_sec"
    ]
    finish_performance_profile(
        performance,
        analysis_started,
        frame_index,
        processed_detection_frames,
    )
    performance["invalid_detection_count"] = detection_stats.get(
        "invalid_detection_count",
        0,
    )
    ball_info = get_model_info(ball_model_key) or {}
    bat_info = get_model_info(bat_model_key) or {}
    return {
        "success": True,
        "analysis_mode": "Batting Analysis",
        "output_path": (
            Path(output_path) if generate_processed_video else None
        ),
        "processed_video_generated": generate_processed_video,
        "processed_video_skipped": not generate_processed_video,
        "total_frames": frame_index,
        "ball_detected_frames": ball_detected_frames,
        "bat_detected_frames": bat_detected_frames,
        "ball_detection_rate": (ball_detected_frames / frame_index) * 100,
        "bat_detection_rate": (bat_detected_frames / frame_index) * 100,
        "impact_info": impact_info,
        "frame_detections": reports["frame_detections"],
        "raw_frame_detections": reports["raw_frame_detections"],
        "impact_frame_detections": reports["frame_detections"],
        "calibration_context": reports["calibration_context"],
        "visual_observer_repair": visual_observer_repair,
        "observer_timeline": observer_timeline,
        "performance_profile": performance,
        "speed_mode": speed_mode,
        "overlay_detail": overlay_detail,
        "smart_pipeline_used": True,
        "shot_info": shot_info,
        "direction_info": direction_info,
        "agent_info": agent_info,
        "shot_type": shot_info.get("shot_type", "Unknown"),
        "shot_confidence": shot_info.get(
            "shot_confidence",
            "Unknown",
        ),
        "shot_direction": shot_info.get("shot_direction", "Unknown"),
        "shot_height": shot_info.get("shot_height", "Unknown"),
        "shot_reason": shot_info.get(
            "reason",
            shot_info.get("shot_reason", ""),
        ),
        "outcome_info": outcome_info,
        "predicted_outcome": outcome_info.get(
            "predicted_outcome",
            "Unknown",
        ),
        "outcome_confidence": outcome_info.get(
            "outcome_confidence",
            "Unknown",
        ),
        "run_estimate": outcome_info.get("run_estimate"),
        "dismissal_risk": outcome_info.get(
            "dismissal_risk",
            "Unknown",
        ),
        "boundary_chance": outcome_info.get(
            "boundary_chance",
            "Unknown",
        ),
        "outcome_reason": outcome_info.get(
            "reason",
            outcome_info.get("outcome_reason", ""),
        ),
        "ball_model_used": ball_info.get("name", ball_model_key),
        "bat_model_used": (
            bat_info.get("name", bat_model_key)
            if bat_model
            else "Unavailable"
        ),
        **enrichment,
    }


def _report_progress(callback, frame_index, total_frames):
    if callable(callback):
        callback(frame_index, total_frames)
