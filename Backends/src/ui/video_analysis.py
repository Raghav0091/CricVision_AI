import json
import time
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st

from Backends.src.analysis.cricket_agent import (
    calculate_detection_quality,
    detect_analysis_warnings,
    generate_coaching_feedback,
    generate_delivery_report,
)
from Backends.src.analysis.field_zones import generate_wagon_wheel_data
from Backends.src.analysis.field_zones import (
    find_nearest_fielder,
    normalize_handedness,
    save_field_analysis_history,
    save_field_setup,
    suggest_field_adjustment,
)
from Backends.src.calibration.calibration_context import (
    build_calibration_context,
    normalize_calibration_context,
)
from Backends.src.config.constants import DETECTION_PRESETS
from Backends.src.config.paths import (
    PROCESSED_VIDEO_DIR,
    REPORTS_DIR,
    REVIEW_FRAMES_DIR,
    VIDEO_ANALYSIS_OUTPUT_DIR,
)
from Backends.src.utils.cv2_loader import cv2
from Backends.src.tracking.ball_tracking_utils import (
    BallKalmanTracker,
    calculate_tracking_quality,
    detect_bounce_by_direction_change,
    get_tracking_quality_label,
    interpolate_missing_positions,
    smooth_trajectory,
)
from Backends.src.models.model_loader import get_cached_yolo_model
from Backends.src.models.model_registry import (
    get_model_info,
    get_model_path,
    validate_model_paths,
)
from Backends.src.ui.analysis_helpers import (
    ensure_delivery_report_fields,
    persist_result_to_session as _persist_result_to_session,
)
from Backends.src.video_pipeline import detection_pipeline as shared_detection
from Backends.src.video_pipeline import annotation_writer as shared_annotations
from Backends.src.video_pipeline.performance_timer import (
    create_performance_profile,
    finish_performance_profile,
)
from Backends.src.video_pipeline.report_pipeline import timed_video_reports
from Backends.src.cricket_delivery_observer import (
    extract_ball_candidates_from_frame_detections,
    fit_observer_path,
    select_best_cricket_path,
)
from Backends.src.video_pipeline.video_reader import (
    extract_first_video_frame as read_first_video_frame,
)


OUTPUT_DIR = VIDEO_ANALYSIS_OUTPUT_DIR
MAX_REVIEW_FRAMES_PER_ANALYSIS = 80

# Shared backend implementations used by the established frame loops and UI.
get_model_options = shared_detection.get_model_options
get_available_ensemble_model_names = shared_detection.get_available_ensemble_model_names
map_model_classes = shared_detection.map_model_classes
load_detection_model = shared_detection.load_detection_model
load_ensemble_models = shared_detection.load_ensemble_models
run_pitch_roi_detection = shared_detection.run_pitch_roi_detection
run_local_redetection = shared_detection.run_local_redetection
choose_main_ball = shared_detection.choose_main_ball
estimate_auto_pitch_corners = shared_detection.estimate_auto_pitch_corners
compute_pitch_homography = shared_detection.compute_pitch_homography
transform_point_to_pitch = shared_detection.transform_point_to_pitch
estimate_line_from_pitch_x = shared_detection.estimate_line_from_pitch_x
estimate_length_from_pitch_y = shared_detection.estimate_length_from_pitch_y
estimate_line_from_stumps = shared_detection.estimate_line_from_stumps
estimate_length_from_bounce = shared_detection.estimate_length_from_bounce
get_nearest_stump_detections = shared_detection.get_nearest_stump_detections
has_enough_ball_movement = shared_detection.has_enough_ball_movement
draw_label = shared_annotations.draw_label
draw_pitch_roi = shared_annotations.draw_pitch_roi
draw_search_roi = shared_annotations.draw_search_roi
save_review_frame = shared_annotations.save_review_frame
convert_to_browser_mp4 = shared_annotations.convert_to_browser_mp4
_add_impact_marker_to_video = shared_annotations.add_impact_marker_to_video
_draw_ball_detections = shared_annotations.draw_ball_detections
draw_clean_ball_markers = shared_annotations.draw_clean_ball_markers
draw_clean_stump_markers = shared_annotations.draw_clean_stump_markers
draw_trajectory_lines = shared_annotations.draw_trajectory_lines
draw_safe_trajectory_lines = shared_annotations.draw_safe_trajectory_lines
ensure_frame_writer_size = shared_annotations.ensure_frame_writer_size
extract_first_video_frame = read_first_video_frame


def save_batting_report(result, analysis_mode):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report_path = REPORTS_DIR / f"batting_analysis_{timestamp}.json"
    impact = result.get("impact_info", {})
    report = {
        "analysis_mode": analysis_mode,
        "ball_detected": bool(result.get("ball_detected_frames", 0)),
        "bat_detected": bool(result.get("bat_detected_frames", 0)),
        "impact_detected": impact.get("impact_detected", False),
        "possible_impact_frame": impact.get("impact_frame"),
        "impact_time_sec": impact.get("impact_time_sec"),
        "min_ball_bat_distance_px": impact.get("min_ball_bat_distance_px"),
        "impact_confidence": impact.get("impact_confidence", "Unknown"),
        "impact_reason": impact.get("reason", impact.get("impact_reason", "")),
        "impact_frame_image_path": str(impact.get("impact_frame_image_path") or ""),
        "shot_type": result.get("shot_info", {}).get("shot_type", "Unknown"),
        "shot_confidence": result.get("shot_info", {}).get("shot_confidence", "Unknown"),
        "shot_direction": result.get("shot_info", {}).get("shot_direction", "Unknown"),
        "shot_height": result.get("shot_info", {}).get("shot_height", "Unknown"),
        "shot_reason": result.get("shot_info", {}).get("reason", result.get("shot_info", {}).get("shot_reason", "")),
        "predicted_outcome": result.get("outcome_info", {}).get("predicted_outcome", "Unknown"),
        "outcome_confidence": result.get("outcome_info", {}).get("outcome_confidence", "Unknown"),
        "run_estimate": result.get("outcome_info", {}).get("run_estimate"),
        "dismissal_risk": result.get("outcome_info", {}).get("dismissal_risk", "Unknown"),
        "boundary_chance": result.get("outcome_info", {}).get("boundary_chance", "Unknown"),
        "outcome_reason": result.get("outcome_info", {}).get(
            "reason",
            result.get("outcome_info", {}).get("outcome_reason", ""),
        ),
        "field_zone": result.get("field_zone", "Unknown"),
        "zone_confidence": result.get("zone_confidence", "Unknown"),
        "direction_angle_degrees": result.get("direction_angle_degrees"),
        "direction_reason": result.get("direction_reason", ""),
        "movement_dx": result.get("movement_dx"),
        "movement_dy": result.get("movement_dy"),
        "direction_shot_category": result.get("direction_shot_category", "Unknown"),
        "agent_quality": result.get("agent_quality", "Unknown"),
        "agent_confidence": result.get("agent_confidence", "Unknown"),
        "ball_tracking_coverage": result.get("ball_tracking_coverage"),
        "bat_detection_coverage": result.get("bat_detection_coverage"),
        "stump_detection_coverage": result.get("stump_detection_coverage"),
        "missing_ball_frames": result.get("missing_ball_frames", 0),
        "possible_false_ball_detections": result.get("possible_false_ball_detections", 0),
        "analysis_consistency": result.get("analysis_consistency", "Unknown"),
        "review_flags": list(result.get("review_flags") or []),
        "agent_notes": result.get("agent_notes", ""),
        "minimum_ball_bat_distance": impact.get(
            "min_distance",
            impact.get("min_ball_bat_distance_px"),
        ),
        "ball_model_used": result.get("ball_model_used", "Unknown"),
        "bat_model_used": result.get("bat_model_used", "Unknown"),
        "calibration_context": result.get("calibration_context"),
        "processed_video_path": str(result.get("output_path", "")),
    }
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
    result["report_path"] = report_path
    result["batting_report"] = report
    return report_path


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
        return {"success": False, "error": "The selected ball model is unavailable."}
    if bat_model is None:
        bat_unavailable_reason = "Impact not detected: bat detection unavailable."

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
    light_annotation = bool(speed_settings.get("light_annotation", False))
    debug_overlay = str(overlay_detail or "Clean").strip().lower() == "debug"
    generate_processed_video = bool(
        generate_processed_video and speed_settings.get("generate_processed_video", True)
    )
    performance = _empty_performance_profile()
    performance["speed_mode"] = speed_mode
    performance["smart_pipeline_used"] = True
    performance["processed_video_generated"] = generate_processed_video
    analysis_started = time.perf_counter()
    processed_detection_frames = 0
    detection_stats = {"invalid_detection_count": 0}
    if width <= 0 or height <= 0:
        cap.release()
        return {"success": False, "error": "Could not read video width/height."}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = None
    if generate_processed_video:
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            cap.release()
            return {"success": False, "error": "Could not create output video writer."}

    frame_index = 0
    ball_detected_frames = 0
    bat_detected_frames = 0
    ball_tracks = []
    bat_detections_by_frame = {}
    impact_frame_detections = []
    frame_detections = impact_frame_detections
    impact_frame_candidates = {}
    trajectory = []
    rough_impact_state = None
    rough_impact_frame = None
    last_bat_detections = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    while True:
        read_started = time.perf_counter()
        success, frame = cap.read()
        if not success:
            break
        if max_frames is not None and frame_index >= max_frames:
            break
        performance["video_read_time_sec"] += time.perf_counter() - read_started
        performance["frames_read"] += 1

        annotated_frame = frame.copy()
        ball_detections = []
        bat_detections = []
        inference_frame, detection_scale = resize_frame_for_inference(frame, resize_width)

        if should_detect_ball(frame_index, speed_settings):
            ball_started = time.perf_counter()
            ball_detections = scale_detections_to_original(
                detect_ball_in_frame(inference_frame, ball_model, confidence),
                detection_scale,
                stats=detection_stats,
            )
            ball_elapsed = time.perf_counter() - ball_started
            performance["ball_detection_time_sec"] += ball_elapsed
            performance["model_inference_time_sec"] += ball_elapsed
            processed_detection_frames += 1

        if should_detect_bat(frame_index, speed_settings, rough_impact_frame) and bat_model:
            bat_started = time.perf_counter()
            bat_detections = scale_detections_to_original(
                detect_bat_in_frame(inference_frame, bat_model, confidence),
                detection_scale,
                stats=detection_stats,
            )
            performance["bat_detection_time_sec"] += time.perf_counter() - bat_started
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
            main_ball = max(ball_detections, key=lambda item: item["confidence"])
            ball_tracks.append(main_ball["center"])
            trajectory.append(tuple(main_ball["center"]))
        else:
            ball_tracks.append(None)
        if bat_detections:
            bat_detected_frames += 1
            bat_detections_by_frame[frame_index] = bat_detections
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
            _draw_ball_detections(annotated_frame, ball_detections)
            draw_bat_detections(annotated_frame, bat_detections)
        else:
            draw_clean_ball_markers(annotated_frame, ball_detections)

        if debug_overlay:
            recent = trajectory[-35:]
            draw_safe_trajectory_lines(annotated_frame, recent)
        elif len(trajectory) >= 2:
            draw_safe_trajectory_lines(annotated_frame, trajectory[-2:])

        if writer is not None:
            annotation_started = time.perf_counter()
            writer.write(ensure_frame_writer_size(annotated_frame, width, height))
            performance["annotation_write_time_sec"] += time.perf_counter() - annotation_started
        frame_index += 1
        if total_frames > 0:
            progress_bar.progress(min(frame_index / total_frames, 1.0))
            status_text.text(f"Processing frame {frame_index}/{total_frames}")

    cap.release()
    if writer is not None:
        writer.release()
    progress_bar.empty()
    status_text.empty()
    if frame_index == 0:
        return {"success": False, "error": "No video frames were processed."}

    preliminary_impact = detect_bat_ball_impact(frame_detections, fps=fps)
    if speed_settings.get("refine_bat_near_impact") and preliminary_impact.get("impact_frame") is not None:
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

    impact_info = detect_bat_ball_impact(impact_frame_detections, fps=fps)
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
        _add_impact_marker_to_video(output_path, impact_info)
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
    performance["report_generation_time_sec"] = reports["report_generation_time_sec"]
    performance["observer_timeline_time_sec"] = reports["observer_timeline_time_sec"]
    finish_performance_profile(
        performance,
        analysis_started,
        frame_index,
        processed_detection_frames,
    )
    performance["invalid_detection_count"] = detection_stats.get("invalid_detection_count", 0)
    ball_info = get_model_info(ball_model_key) or {}
    bat_info = get_model_info(bat_model_key) or {}
    return {
        "success": True,
        "analysis_mode": "Batting Analysis",
        "output_path": Path(output_path) if generate_processed_video else None,
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
        "shot_confidence": shot_info.get("shot_confidence", "Unknown"),
        "shot_direction": shot_info.get("shot_direction", "Unknown"),
        "shot_height": shot_info.get("shot_height", "Unknown"),
        "shot_reason": shot_info.get("reason", shot_info.get("shot_reason", "")),
        "outcome_info": outcome_info,
        "predicted_outcome": outcome_info.get("predicted_outcome", "Unknown"),
        "outcome_confidence": outcome_info.get("outcome_confidence", "Unknown"),
        "run_estimate": outcome_info.get("run_estimate"),
        "dismissal_risk": outcome_info.get("dismissal_risk", "Unknown"),
        "boundary_chance": outcome_info.get("boundary_chance", "Unknown"),
        "outcome_reason": outcome_info.get("reason", outcome_info.get("outcome_reason", "")),
        "ball_model_used": ball_info.get("name", ball_model_key),
        "bat_model_used": bat_info.get("name", bat_model_key) if bat_model else "Unavailable",
        **enrichment,
    }


def get_default_pitch_points(frame):
    height, width = frame.shape[:2]
    return {
        "top_left": (int(width * 0.40), int(height * 0.28)),
        "top_right": (int(width * 0.60), int(height * 0.28)),
        "bottom_left": (int(width * 0.22), int(height * 0.95)),
        "bottom_right": (int(width * 0.78), int(height * 0.95)),
    }


def show_manual_pitch_point_inputs(frame):
    defaults = get_default_pitch_points(frame)
    height, width = frame.shape[:2]
    preview = frame.copy()

    for label, point in defaults.items():
        cv2.circle(preview, point, 7, (0, 255, 0), -1)
        draw_label(preview, label.replace("_", " "), point[0], point[1], (40, 160, 40))

    st.image(
        cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
        caption="First frame with default pitch-point guide. Adjust the coordinates below.",
        use_container_width=True,
    )

    points = []
    labels = [
        ("Top-left pitch corner", "top_left"),
        ("Top-right pitch corner", "top_right"),
        ("Bottom-left pitch corner", "bottom_left"),
        ("Bottom-right pitch corner", "bottom_right"),
    ]

    for label, key in labels:
        default_x, default_y = defaults[key]
        col_x, col_y = st.columns(2)

        with col_x:
            point_x = st.number_input(
                f"{label} X",
                min_value=0,
                max_value=max(width - 1, 0),
                value=default_x,
                step=1,
                key=f"manual_pitch_{key}_x",
            )

        with col_y:
            point_y = st.number_input(
                f"{label} Y",
                min_value=0,
                max_value=max(height - 1, 0),
                value=default_y,
                step=1,
                key=f"manual_pitch_{key}_y",
            )

        points.append((int(point_x), int(point_y)))

    return points


def show_cricket_delivery_report(result):
    ensure_delivery_report_fields(result)

    quality = calculate_detection_quality(result)
    report = generate_delivery_report(result)
    feedback_items = generate_coaching_feedback(result)
    warnings = detect_analysis_warnings(result)

    st.subheader("🏏 Delivery Report")
    st.metric(
        "Analysis Quality",
        f"{quality['quality_score']}/100",
        quality["quality_label"],
    )
    st.info(report)

    st.markdown("**Coaching Feedback**")
    for feedback_item in feedback_items:
        st.markdown(f"- {feedback_item}")

    if warnings:
        st.warning("Warnings:\n" + "\n".join(f"- {warning}" for warning in warnings))


DEBUG_PERFORMANCE = False


def _empty_performance_profile():
    return create_performance_profile()


def process_video(
    video_path,
    output_path,
    model_path,
    model_key=None,
    class_names=None,
    confidence=0.25,
    imgsz=640,
    use_ensemble=False,
    show_pitch_roi=False,
    calibration_mode="Auto calibration using detected stumps",
    manual_pitch_points=None,
    shot_trajectory_mode="Use last part of trajectory",
    manual_contact_frame=None,
    field_setup=None,
    bat_model_key=None,
    speed_mode="Smart Balanced",
    max_frames=None,
    generate_processed_video=True,
    calibration_context=None,
    overlay_detail="Clean",
):
    from Backends.src.analysis.analysis_speed import (
        get_analysis_mode_settings,
        normalize_detections,
        resize_frame_for_inference,
        scale_detections_to_original,
    )
    from Backends.src.analysis.smart_pipeline import (
        apply_locked_stump,
        lock_static_stump_detection,
        refine_bat_detections_near_impact,
        should_detect_ball,
        should_detect_bat,
        should_detect_stump,
        update_rough_impact_frame,
    )
    from Backends.src.analysis.bat_detection import (
        detect_bat_in_frame,
        draw_bat_detections,
    )
    from Backends.src.analysis.impact_detection import (
        detect_bat_ball_impact,
        save_impact_frame_preview,
    )

    model = None
    ensemble_models = []
    bat_model = get_cached_yolo_model(bat_model_key) if bat_model_key else None
    bat_unavailable_reason = ""

    if bat_model_key and bat_model is None:
        bat_unavailable_reason = "Impact not detected: bat detection unavailable."
    stump_model = get_cached_yolo_model("current_best")

    if stump_model is None:
        return {
            "success": False,
            "error": "Current Best Ball + Stump Model is unavailable.",
        }

    stump_class_names = map_model_classes(stump_model)

    if use_ensemble:
        ensemble_models = load_ensemble_models()

        if not ensemble_models:
            return {
                "success": False,
                "error": "No ensemble models were found. Add at least one configured model file.",
            }
    else:
        model = load_detection_model(model_key=model_key, model_path=model_path)

        if model is None:
            model_label = (get_model_info(model_key) or {}).get("name") if model_key else None
            return {
                "success": False,
                "error": f"Model not found: {model_label or model_path}",
            }

        class_names = map_model_classes(model)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {
            "success": False,
            "error": "Could not open uploaded video.",
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        cap.release()
        return {
            "success": False,
            "error": "Could not read video width/height.",
        }

    speed_settings = get_analysis_mode_settings(speed_mode)
    speed_mode = speed_settings.get("mode", speed_mode)
    inference_imgsz = int(speed_settings.get("yolo_imgsz", imgsz))
    resize_width = speed_settings.get("resize_width")
    light_annotation = bool(speed_settings.get("light_annotation", False))
    debug_overlay = str(overlay_detail or "Clean").strip().lower() == "debug"
    generate_processed_video = bool(
        generate_processed_video and speed_settings.get("generate_processed_video", True)
    )
    performance = _empty_performance_profile()
    performance["speed_mode"] = speed_mode
    performance["smart_pipeline_used"] = True
    performance["processed_video_generated"] = generate_processed_video
    analysis_started = time.perf_counter()
    processed_detection_frames = 0
    detection_stats = {"invalid_detection_count": 0}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    if generate_processed_video:
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            return {
                "success": False,
                "error": "Could not create output video writer.",
            }

    frame_index = 0

    ball_detected_frames = 0
    stump_detected_frames = 0

    total_ball_detections = 0
    total_stump_detections = 0

    confidence_values = []
    low_confidence_ball_frames = 0
    review_frame_count = 0
    full_frame_detection_time_total = 0
    roi_detection_time_total = 0
    roi_detected_frames = 0
    tracker_recoveries = 0
    kalman_predicted_frames = 0
    last_roi_size = "Full frame"

    trajectory_points = []
    ball_positions = []
    bat_detections_by_frame = {}
    bat_detected_frames = 0
    impact_frame_detections = []
    frame_detections = impact_frame_detections
    impact_frame_candidates = {}
    stump_detections_by_frame = []
    last_raw_frame = None
    previous_roi_box = None
    max_trajectory_points = 35
    pitch_homography = None
    calibration_status = "Not calibrated"
    calibration_source = "None"
    calibration_warning = "Confidence warning: pitch calibration is missing; using image-space fallback."
    pitch_normalized_bounce_point = None

    if calibration_mode.startswith("Manual"):
        pitch_homography = compute_pitch_homography(manual_pitch_points)

        if pitch_homography is not None:
            calibration_status = "Calibrated"
            calibration_source = "Manual"
            calibration_warning = ""

    previous_ball_center = None
    kalman_tracker = BallKalmanTracker(max_missing_frames=10)

    missing_ball_frames = 0
    max_missing_ball_frames = 12

    estimated_bounce_point = None
    estimated_bounce_frame = None

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    estimated_bounce_point = None
    estimated_bounce_frame = None
    estimated_line = "Unknown"
    estimated_length = "Unknown"
    field_setup = field_setup or {}
    batter_handedness = normalize_handedness(field_setup.get("batter_handedness", "right"))
    bowler_arm = field_setup.get("bowler_arm", "Right-arm bowler")
    camera_view = field_setup.get("camera_view", "Behind bowler")
    calibration_context = normalize_calibration_context(calibration_context)
    if calibration_context.get("enabled"):
        if calibration_context.get("batter_handedness") != "unknown":
            batter_handedness = calibration_context["batter_handedness"]
        if calibration_context.get("camera_view") != "unknown":
            camera_view = calibration_context["camera_view"]
    fielders = field_setup.get("fielders", [])
    field_preset = field_setup.get("preset", "Custom")

    min_track_points_for_bounce = 8
    min_movement_distance = 40
    min_ball_confidence_for_tracking = 0.35
    locked_stump = None
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

        performance["video_read_time_sec"] += time.perf_counter() - read_started
        performance["frames_read"] += 1

        last_raw_frame = frame.copy()
        annotated_frame = frame.copy()
        low_confidence_ball_detections = []
        ball_detections = []
        stump_detections = []
        bat_detections = []

        inference_frame, detection_scale = resize_frame_for_inference(frame, resize_width)
        run_stump = should_detect_stump(frame_index, speed_settings, locked_stump)
        run_bat = should_detect_bat(frame_index, speed_settings, rough_impact_frame)

        if should_detect_ball(frame_index, speed_settings):
            ball_started = time.perf_counter()
            detection_result = run_pitch_roi_detection(
                inference_frame,
                stump_model=stump_model,
                stump_class_names=stump_class_names,
                confidence=confidence,
                imgsz=inference_imgsz,
                previous_roi=previous_roi_box,
                ball_model=model,
                ball_class_names=class_names,
                ensemble_models=ensemble_models,
                use_ensemble=use_ensemble,
                ball_confidence=min_ball_confidence_for_tracking,
                speed_settings=speed_settings,
                detect_stump=run_stump,
                locked_stump_detections=[locked_stump] if locked_stump and not run_stump else None,
                use_roi=speed_settings.get("use_roi", True),
            )
            ball_detections = scale_detections_to_original(
                detection_result["ball_detections"],
                detection_scale,
                stats=detection_stats,
            )
            stump_detections = scale_detections_to_original(
                detection_result["stump_detections"],
                detection_scale,
                stats=detection_stats,
            )
            low_confidence_ball_detections = scale_detections_to_original(
                detection_result.get("low_confidence_ball_detections", []),
                detection_scale,
                stats=detection_stats,
            )
            ball_elapsed = time.perf_counter() - ball_started
            performance["ball_detection_time_sec"] += ball_elapsed
            performance["model_inference_time_sec"] += ball_elapsed
            if run_stump:
                performance["stump_detection_time_sec"] += detection_result.get("full_frame_time_ms", 0) / 1000.0
            processed_detection_frames += 1
            full_frame_detection_time_total += detection_result["full_frame_time_ms"]
            roi_detection_time_total += detection_result["roi_time_ms"]

            if detection_result.get("used_roi"):
                previous_roi_box = detection_result["roi_box"]
                roi_detected_frames += 1
                roi_x1, roi_y1, roi_x2, roi_y2 = detection_result["roi_box"]
                last_roi_size = f"{roi_x2 - roi_x1}x{roi_y2 - roi_y1}"

            if debug_overlay and show_pitch_roi:
                draw_pitch_roi(annotated_frame, detection_result.get("roi_box"))

            confidence_values.extend(item["confidence"] for item in ball_detections)

            if low_confidence_ball_detections:
                low_confidence_ball_frames += 1

                if review_frame_count < MAX_REVIEW_FRAMES_PER_ANALYSIS:
                    save_review_frame(
                        frame,
                        timestamp,
                        frame_index,
                        "low_confidence",
                        low_confidence_ball_detections,
                        source="video_analysis",
                    )
                    review_frame_count += 1

            if ball_detections:
                ball_detected_frames += 1
                total_ball_detections += len(ball_detections)
            elif speed_settings.get("enable_local_redetection", True):
                search_center = previous_ball_center or kalman_tracker.last_prediction
                recovery_result = run_local_redetection(
                    inference_frame,
                    search_center,
                    confidence,
                    inference_imgsz,
                    missing_ball_frames + 1,
                    ball_model=model,
                    ball_class_names=class_names,
                    ensemble_models=ensemble_models,
                    use_ensemble=use_ensemble,
                )

                if debug_overlay and show_pitch_roi:
                    draw_search_roi(annotated_frame, recovery_result.get("search_roi"))

                if recovery_result["recovered"]:
                    ball_detections = scale_detections_to_original(
                        recovery_result["ball_detections"],
                        detection_scale,
                        stats=detection_stats,
                    )
                    tracker_recoveries += 1
                    ball_detected_frames += 1
                    total_ball_detections += len(ball_detections)
                    confidence_values.extend(item["confidence"] for item in ball_detections)
                elif review_frame_count < MAX_REVIEW_FRAMES_PER_ANALYSIS:
                    save_review_frame(
                        frame,
                        timestamp,
                        frame_index,
                        "missed_ball",
                        source="video_analysis",
                        note="No ball detection passed the selected confidence threshold.",
                    )
                    review_frame_count += 1
            elif review_frame_count < MAX_REVIEW_FRAMES_PER_ANALYSIS:
                save_review_frame(
                    frame,
                    timestamp,
                    frame_index,
                    "missed_ball",
                    source="video_analysis",
                    note="No ball detection passed the selected confidence threshold.",
                )
                review_frame_count += 1

        if run_bat and bat_model:
            bat_started = time.perf_counter()
            bat_detections = scale_detections_to_original(
                detect_bat_in_frame(inference_frame, bat_model, confidence),
                detection_scale,
                stats=detection_stats,
            )
            performance["bat_detection_time_sec"] += time.perf_counter() - bat_started
            last_bat_detections = bat_detections
        elif last_bat_detections:
            bat_detections = last_bat_detections

        stump_detections = apply_locked_stump(stump_detections, locked_stump)

        if stump_detections:
            stump_detected_frames += 1
            total_stump_detections += len(stump_detections)

            if calibration_mode.startswith("Auto") and pitch_homography is None:
                auto_pitch_points = estimate_auto_pitch_corners(frame.shape, stump_detections)
                pitch_homography = compute_pitch_homography(auto_pitch_points)

                if pitch_homography is not None:
                    calibration_status = "Calibrated"
                    calibration_source = "Auto"
                    calibration_warning = ""

        if bat_detections:
            bat_detected_frames += 1
            bat_detections_by_frame[frame_index] = bat_detections
        if debug_overlay:
            draw_bat_detections(annotated_frame, bat_detections)

        stump_detections_by_frame.append(stump_detections)
        impact_frame_detections.append(
            {
                "frame_index": frame_index,
                "ball_detections": ball_detections,
                "bat_detections": bat_detections,
                "stump_detections": stump_detections,
            }
        )
        if ball_detections and bat_detections:
            impact_frame_candidates[frame_index] = frame.copy()

        initial_frames = speed_settings.get("stump_detect_initial_frames")
        if locked_stump is None and initial_frames:
            if frame_index + 1 >= int(initial_frames):
                locked_stump = lock_static_stump_detection(frame_detections, int(initial_frames))

        rough_impact_state = update_rough_impact_frame(
            frame_index,
            ball_detections,
            bat_detections,
            rough_impact_state,
        )
        if rough_impact_state is not None:
            rough_impact_frame = rough_impact_state[1]

        if debug_overlay:
            for detection in ball_detections:
                x1, y1, x2, y2 = detection["box"]
                conf = detection["confidence"]
                center_x, center_y = detection["center"]

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    2,
                )

                cv2.circle(
                    annotated_frame,
                    (center_x, center_y),
                    5,
                    (0, 255, 255),
                    -1,
                )

                draw_label(
                    annotated_frame,
                    f"ball {conf:.2f}",
                    x1,
                    y1,
                    (0, 180, 180),
                )

            for detection in stump_detections:
                x1, y1, x2, y2 = detection["box"]
                conf = detection["confidence"]

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 100, 0),
                    2,
                )

                draw_label(
                    annotated_frame,
                    f"stump {conf:.2f}",
                    x1,
                    y1,
                    (255, 100, 0),
                )
        else:
            draw_clean_ball_markers(annotated_frame, ball_detections)
            draw_clean_stump_markers(annotated_frame, stump_detections)

        main_ball = choose_main_ball(ball_detections, previous_ball_center)

        if main_ball is not None:
            missing_ball_frames = 0

            previous_ball_center = main_ball["center"]
            kalman_tracker.update(previous_ball_center)
            ball_positions.append(previous_ball_center)

            smoothed_positions = smooth_trajectory(ball_positions)
            display_trajectory_points = []

            for point in reversed(smoothed_positions):
                if point is None:
                    if display_trajectory_points:
                        break
                    continue
                display_trajectory_points.append(point)

            trajectory_points = list(reversed(display_trajectory_points))[-max_trajectory_points:]
            interpolated_positions = interpolate_missing_positions(ball_positions)
            usable_trajectory_points = [
                point for point in interpolated_positions if point is not None
            ]
            bounce_result = None

            if (
                len(usable_trajectory_points) >= min_track_points_for_bounce
                and has_enough_ball_movement(usable_trajectory_points, min_movement_distance)
            ):
                bounce_result = detect_bounce_by_direction_change(ball_positions)

            if bounce_result is not None and estimated_bounce_point is None:
                
                estimated_bounce_point = bounce_result["point"]
                
                estimated_bounce_frame = bounce_result["frame_index"]
                bounce_stump_detections = get_nearest_stump_detections(
                    stump_detections_by_frame,
                    estimated_bounce_frame,
                )

                estimated_line = estimate_line_from_stumps(
                    estimated_bounce_point,
                    bounce_stump_detections,
                    batter_handedness,
                )

                estimated_length = estimate_length_from_bounce(
                    estimated_bounce_point,
                    height 
                )

                pitch_normalized_bounce_point = transform_point_to_pitch(
                    estimated_bounce_point,
                    pitch_homography,
                )

                if pitch_normalized_bounce_point is not None:
                    pitch_x, pitch_y = pitch_normalized_bounce_point
                    estimated_line = estimate_line_from_pitch_x(pitch_x, batter_handedness)
                    estimated_length = estimate_length_from_pitch_y(pitch_y)


        else:
            missing_ball_frames += 1
            predicted_center = kalman_tracker.predict()

            if predicted_center is not None and missing_ball_frames <= 10:
                ball_positions.append(predicted_center)
                previous_ball_center = predicted_center
                kalman_predicted_frames += 1
            else:
                ball_positions.append(None)

            if missing_ball_frames >= max_missing_ball_frames:
                kalman_tracker.reset()
                if review_frame_count < MAX_REVIEW_FRAMES_PER_ANALYSIS:
                    save_review_frame(
                        frame,
                        timestamp,
                        frame_index,
                        "poor_tracking",
                        source="video_analysis",
                        note=f"Ball missing for {missing_ball_frames} consecutive frames.",
                    )
                    review_frame_count += 1
                trajectory_points.clear()
                previous_ball_center = None
            else:
                smoothed_positions = smooth_trajectory(ball_positions)
                display_trajectory_points = []

                for point in reversed(smoothed_positions):
                    if point is None:
                        if display_trajectory_points:
                            break
                        continue
                    display_trajectory_points.append(point)

                trajectory_points = list(reversed(display_trajectory_points))[-max_trajectory_points:]

        # ponytail: protect yellow path drawing only; tracker/report results stay unchanged.
        draw_safe_trajectory_lines(annotated_frame, trajectory_points)

        if debug_overlay and estimated_bounce_point is not None:
            bx, by = estimated_bounce_point

            cv2.circle(
                annotated_frame,
                (bx, by),
                10,
                (0, 0, 255),
                -1,
            )

            cv2.circle(
                annotated_frame,
                (bx, by),
                16,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"Bounce Frame: {estimated_bounce_frame}",
                (bx + 15, by - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

            cv2.rectangle(
                annotated_frame,
                (15, 15),
                (470, 240),
                (0, 0, 0),
                -1,
            )
        
        

        bounce_text = "Not found"
        if estimated_bounce_frame is not None:
            bounce_text = f"Frame {estimated_bounce_frame}"

        if debug_overlay and not light_annotation:
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_index}/{source_total_frames or frame_index}",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"Balls in frame: {len(ball_detections)}",
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"Stumps in frame: {len(stump_detections)}",
                (30, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 160, 0),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"Trajectory: {len(trajectory_points)} | Missing: {missing_ball_frames}",
                (30, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"Bounce: {bounce_text}",
                (30, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"Line: {estimated_line}",
                (30, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"Length: {estimated_length}",
                (30, 225),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

        if writer is not None:
            annotation_started = time.perf_counter()
            writer.write(ensure_frame_writer_size(annotated_frame, width, height))
            performance["annotation_write_time_sec"] += time.perf_counter() - annotation_started

        frame_index += 1

        if source_total_frames > 0:
            progress = min(frame_index / source_total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_index}/{source_total_frames}")
        else:
            status_text.text(f"Processing frame {frame_index}")

    cap.release()
    if writer is not None:
        writer.release()

    progress_bar.empty()
    status_text.empty()

    if frame_index == 0:
        return {
            "success": False,
            "error": "No frames were processed. The uploaded video may be corrupted or unsupported.",
        }

    preliminary_impact = detect_bat_ball_impact(frame_detections, fps=fps)
    if speed_settings.get("refine_bat_near_impact") and preliminary_impact.get("impact_frame") is not None:
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

    impact_info = detect_bat_ball_impact(impact_frame_detections, fps=fps)
    if bat_unavailable_reason:
        impact_info["reason"] = bat_unavailable_reason
        impact_info["impact_reason"] = bat_unavailable_reason
    impact_frame = impact_info.get("impact_frame")
    if impact_frame is not None:
        preview_path = save_impact_frame_preview(
            impact_frame_candidates.get(impact_frame),
            impact_info,
            prefix=f"video_impact_{Path(output_path).stem}",
        )
        if preview_path is not None:
            impact_info["impact_frame_image_path"] = str(preview_path)
        if not speed_settings.get("skip_impact_video_rewrite") and generate_processed_video:
            _add_impact_marker_to_video(output_path, impact_info)

    report_started = time.perf_counter()
    stump_detection_rate = 0

    if frame_index > 0:
        ball_detection_rate = (ball_detected_frames / frame_index) * 100
        stump_detection_rate = (stump_detected_frames / frame_index) * 100

    average_confidence = 0

    if confidence_values:
        average_confidence = sum(confidence_values) / len(confidence_values)

    tracking_quality = calculate_tracking_quality(ball_positions, frame_index)
    overall_tracking_quality = get_tracking_quality_label(
        tracking_quality["tracking_rate"],
        tracking_quality["interpolated_frames"],
        kalman_predicted_frames,
    )
    delivery_report = {
        "estimated_line": estimated_line,
        "estimated_length": estimated_length,
        "ball_detection_rate": ball_detection_rate,
        "overall_tracking_quality": overall_tracking_quality,
    }
    calibration_context = build_calibration_context(
        calibration_context,
        frame_detections=frame_detections,
        frame_width=width,
        frame_height=height,
    )
    reports = timed_video_reports(
        frame_detections,
        fps=fps,
        total_frames=frame_index,
        batter_handedness=batter_handedness,
        delivery_report=delivery_report,
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
    performance["report_generation_time_sec"] += reports["report_generation_time_sec"]
    performance["observer_timeline_time_sec"] = reports["observer_timeline_time_sec"]
    finish_performance_profile(
        performance,
        analysis_started,
        frame_index,
        processed_detection_frames,
    )
    performance["invalid_detection_count"] = detection_stats.get("invalid_detection_count", 0)
    wagon_wheel = generate_wagon_wheel_data(
        ball_positions,
        batter_handedness=batter_handedness,
        mode=shot_trajectory_mode,
        manual_contact_frame=manual_contact_frame,
    )
    wagon_wheel["mode"] = shot_trajectory_mode
    nearest_fielder = find_nearest_fielder(
        wagon_wheel.get("shot_angle"), fielders, batter_handedness
    )
    wagon_wheel["nearest_fielder"] = nearest_fielder
    wagon_wheel["suggested_adjustment"] = suggest_field_adjustment(wagon_wheel, nearest_fielder)

    if field_setup:
        save_field_setup(field_setup)
        save_field_analysis_history(
            {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "source": "video_analysis",
                "batter_handedness": batter_handedness,
                "bowler_arm": bowler_arm,
                "camera_view": camera_view,
                "preset": field_preset,
                "simple_zone": wagon_wheel.get("simple_zone", "Unknown"),
                "detailed_zone": wagon_wheel.get("detailed_zone", "Unknown"),
                "shot_angle": "" if wagon_wheel.get("shot_angle") is None else f"{wagon_wheel['shot_angle']:.2f}",
                "nearest_fielder": "" if nearest_fielder is None else nearest_fielder.get("name", ""),
                "confidence": wagon_wheel.get("confidence", "Low"),
                "corrected_zone": "",
            }
        )
    average_full_frame_detection_time = 0
    average_roi_detection_time = 0

    if frame_index > 0:
        average_full_frame_detection_time = full_frame_detection_time_total / frame_index

    if roi_detected_frames > 0:
        average_roi_detection_time = roi_detection_time_total / roi_detected_frames

    if last_raw_frame is not None and estimated_bounce_point is None:
        if review_frame_count < MAX_REVIEW_FRAMES_PER_ANALYSIS:
            save_review_frame(
                last_raw_frame,
                timestamp,
                max(frame_index - 1, 0),
                "bounce_unknown",
                source="video_analysis",
                note="Analysis finished without a bounce estimate.",
            )
            review_frame_count += 1

    if last_raw_frame is not None and (estimated_line == "Unknown" or estimated_length == "Unknown"):
        if review_frame_count < MAX_REVIEW_FRAMES_PER_ANALYSIS:
            save_review_frame(
                last_raw_frame,
                timestamp,
                max(frame_index - 1, 0),
                "line_length_unknown",
                source="video_analysis",
                note=f"Line={estimated_line}; Length={estimated_length}.",
            )
            review_frame_count += 1

    result_payload = {
        "success": True,
        "output_path": str(output_path) if generate_processed_video else None,
        "processed_video_generated": generate_processed_video,
        "processed_video_skipped": not generate_processed_video,
        "smart_pipeline_used": True,
        "total_frames": frame_index,
        "ball_detected_frames": ball_detected_frames,
        "stump_detected_frames": stump_detected_frames,
        "bat_detected_frames": bat_detected_frames,
        "total_ball_detections": total_ball_detections,
        "low_confidence_ball_frames": low_confidence_ball_frames,
        "total_stump_detections": total_stump_detections,
        "ball_detection_rate": ball_detection_rate,
        "ball_tracking_rate": tracking_quality["tracking_rate"],
        "interpolated_ball_frames": tracking_quality["interpolated_frames"],
        "kalman_predicted_frames": kalman_predicted_frames,
        "tracker_recoveries": tracker_recoveries,
        "overall_tracking_quality": overall_tracking_quality,
        "stump_detection_rate": stump_detection_rate,
        "average_ball_confidence": average_confidence,
        "full_frame_detection_time_ms": average_full_frame_detection_time,
        "roi_detection_time_ms": average_roi_detection_time,
        "roi_detected_frames": roi_detected_frames,
        "last_roi_size": last_roi_size,
        "estimated_bounce_point": estimated_bounce_point,
        "estimated_bounce_frame": estimated_bounce_frame,
        "pitch_normalized_bounce_point": pitch_normalized_bounce_point,
        "calibration_status": calibration_status,
        "calibration_source": calibration_source,
        "calibration_warning": calibration_warning,
        "estimated_line": estimated_line,
        "estimated_length": estimated_length,
        "wagon_wheel": wagon_wheel,
        "field_setup": field_setup,
        "batter_handedness": batter_handedness,
        "bowler_arm": bowler_arm,
        "camera_view": camera_view,
        "review_frame_count": review_frame_count,
        "review_frames_dir": REVIEW_FRAMES_DIR,
        "frame_detections": reports["frame_detections"],
        "raw_frame_detections": reports["raw_frame_detections"],
        "impact_frame_detections": reports["frame_detections"],
        "calibration_context": reports["calibration_context"],
        "visual_observer_repair": visual_observer_repair,
        "observer_timeline": observer_timeline,
        "performance_profile": performance,
        "speed_mode": speed_mode,
        "overlay_detail": overlay_detail,
        "impact_info": impact_info,
        "shot_info": shot_info,
        "direction_info": direction_info,
        "agent_info": agent_info,
        "shot_type": shot_info.get("shot_type", "Unknown"),
        "shot_confidence": shot_info.get("shot_confidence", "Unknown"),
        "shot_direction": shot_info.get("shot_direction", "Unknown"),
        "shot_height": shot_info.get("shot_height", "Unknown"),
        "shot_reason": shot_info.get("reason", shot_info.get("shot_reason", "")),
        "outcome_info": outcome_info,
        "predicted_outcome": outcome_info.get("predicted_outcome", "Unknown"),
        "outcome_confidence": outcome_info.get("outcome_confidence", "Unknown"),
        "run_estimate": outcome_info.get("run_estimate"),
        "dismissal_risk": outcome_info.get("dismissal_risk", "Unknown"),
        "boundary_chance": outcome_info.get("boundary_chance", "Unknown"),
        "outcome_reason": outcome_info.get("reason", outcome_info.get("outcome_reason", "")),
        "ball_model_used": "Current Best Ball + Stump Model",
        "bat_model_used": (
            (get_model_info(bat_model_key) or {}).get("name", bat_model_key)
            if bat_model_key and bat_model is not None
            else ("Unavailable" if bat_model_key else "Not used")
        ),
        **enrichment,
    }

    # ponytail: compute physics fit once here so UI/3D/overlay all agree on one report.
    result_payload["physics_trajectory"] = build_physics_trajectory_report_from_result(result_payload)
    if generate_processed_video and not speed_settings.get("skip_impact_video_rewrite"):
        shared_annotations.add_physics_trajectory_overlay_to_video(
            output_path,
            result_payload["physics_trajectory"],
        )
    return result_payload


def _coerce_trajectory_point(value):
    if value is None:
        return None
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            try:
                return int(value["x"]), int(value["y"])
            except (TypeError, ValueError):
                return None
        return _coerce_trajectory_point(value.get("center"))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    return None


def extract_trajectory_points_from_result(result):
    """Defensively pull image-space trajectory points from an analysis result.

    Prefers points that retain frame_index so pre-impact truncation can work.
    """
    result = result or {}
    from Backends.src.analysis.frame_detection_utils import (
        best_detection_center,
        normalize_frame_detections,
    )

    frames = normalize_frame_detections(
        result.get("frame_detections") or result.get("impact_frame_detections")
    )
    framed_points = []
    for frame in frames:
        center = best_detection_center(frame.get("ball_detections"))
        coerced = _coerce_trajectory_point(center)
        if coerced is None:
            continue
        framed_points.append(
            {
                "x": coerced[0],
                "y": coerced[1],
                "frame_index": frame.get("frame_index"),
                "source": "observed",
            }
        )
    if len(framed_points) >= 3:
        return framed_points

    for key in ("trajectory_points", "smoothed_trajectory", "display_trajectory_points"):
        raw = result.get(key)
        if not isinstance(raw, (list, tuple)):
            continue
        points = []
        for item in raw:
            if isinstance(item, dict) and ("x" in item or "center" in item):
                point = _coerce_trajectory_point(item)
                if point is None:
                    continue
                payload = {"x": point[0], "y": point[1], "source": "observed"}
                if item.get("frame_index") is not None:
                    payload["frame_index"] = item.get("frame_index")
                points.append(payload)
            else:
                point = _coerce_trajectory_point(item)
                if point is not None:
                    points.append({"x": point[0], "y": point[1], "source": "observed"})
        if len(points) >= 3:
            return points

    ball_positions = result.get("ball_positions")
    if isinstance(ball_positions, (list, tuple)) and ball_positions:
        smoothed_positions = smooth_trajectory(ball_positions)
        display_trajectory_points = []
        for point in reversed(smoothed_positions):
            if point is None:
                if display_trajectory_points:
                    break
                continue
            coerced = _coerce_trajectory_point(point)
            if coerced is not None:
                display_trajectory_points.append(coerced)
        points = list(reversed(display_trajectory_points))[-35:]
        if len(points) >= 3:
            return [{"x": x, "y": y, "source": "observed"} for x, y in points]

    return framed_points


def extract_bounce_point_from_result(result):
    result = result or {}
    for key in ("estimated_bounce_point", "bounce_point"):
        point = _coerce_trajectory_point(result.get(key))
        if point is not None:
            return point
    return None


def render_trajectory_replay_section(result, path_validity=None):
    try:
        from Backends.src.detection_health import build_detection_health
        from Backends.src.trajectory_replay import build_trajectory_replay_image
    except ImportError as exc:
        st.warning(f"Trajectory replay unavailable: {exc}")
        return

    if path_validity is None:
        path_validity = prepare_result_path_validity(result)
    trajectory_points = list(path_validity.get("valid_xy") or [])
    if len(trajectory_points) < 3:
        trajectory_points = _observer_points_to_xy(extract_trajectory_points_from_result(result))
    if len(trajectory_points) < 3:
        st.subheader("CricVision Trajectory Replay")
        st.info("Trajectory replay unavailable: not enough tracked ball points.")
        return

    settings = st.session_state.get("video_analysis_settings", {})
    preset_name = settings.get("preset_name") or result.get("active_preset") or "Balanced Mode"
    preset = DETECTION_PRESETS.get(preset_name, {})
    health = build_detection_health(
        result,
        model_name=settings.get("selected_model_name") or result.get("active_model"),
        detection_preset=preset_name,
        speed_mode=settings.get("speed_mode") or result.get("speed_mode"),
        confidence_threshold=preset.get("confidence"),
        imgsz=preset.get("imgsz"),
    )

    replay_image = build_trajectory_replay_image(
        trajectory_points,
        bounce_point=extract_bounce_point_from_result(result),
        health=health,
    )
    if replay_image is None:
        st.subheader("CricVision Trajectory Replay")
        st.info("Trajectory replay unavailable: not enough tracked ball points.")
        return

    st.subheader("CricVision Trajectory Replay")
    labels = path_validity.get("labels") or []
    if labels:
        st.caption(" · ".join(labels))
    st.image(
        cv2.cvtColor(replay_image, cv2.COLOR_BGR2RGB),
        caption=(
            "Approximate trajectory replay from tracked image-space ball points. "
            "Pitch geometry and speed/swing/spin are not calibrated in v1."
        ),
        use_container_width=True,
    )


def extract_frame_size_from_result(result):
    """Defensively read frame dimensions from an analysis result."""
    result = result or {}
    calibration = result.get("calibration_context") or {}
    width = calibration.get("frame_width")
    height = calibration.get("frame_height")
    if width and height:
        return int(width), int(height)
    for key in ("frame_width", "width"):
        if result.get(key):
            width = result.get(key)
            break
    for key in ("frame_height", "height"):
        if result.get(key):
            height = result.get(key)
            break
    if width and height:
        try:
            return int(width), int(height)
        except (TypeError, ValueError):
            pass
    return 1280, 720


def extract_stump_detections_from_result(result):
    """Collect stump detections already produced by analysis."""
    result = result or {}
    from Backends.src.analysis.frame_detection_utils import normalize_frame_detections

    collected = []
    frames = normalize_frame_detections(
        result.get("frame_detections") or result.get("impact_frame_detections")
    )
    for frame in frames:
        collected.extend(frame.get("stump_detections") or [])

    calibration = result.get("calibration_context") or {}
    stumps = calibration.get("stumps") or {}
    batter_end = stumps.get("batter_end")
    if isinstance(batter_end, dict) and batter_end.get("center"):
        collected.append(batter_end)
    return collected


def extract_pitch_roi_from_result(result):
    """Read pitch corridor / ROI geometry from analysis outputs."""
    result = result or {}
    calibration = result.get("calibration_context") or {}
    corridor = calibration.get("pitch_corridor")
    if isinstance(corridor, dict) and (corridor.get("bbox") or corridor.get("polygon")):
        return corridor
    if result.get("last_roi_size"):
        try:
            x1, y1, x2, y2 = result["last_roi_size"]
            return {"bbox": [x1, y1, x2, y2], "source": "analysis_roi"}
        except (TypeError, ValueError):
            pass
    return None


def prepare_result_path_validity(result, trajectory_points=None):
    """Compute cricket-path validity once for UI drawing / 3D / metrics."""
    from Backends.src.cricket_path_validity import prepare_safe_trajectory_for_draw

    result = result or {}
    points = trajectory_points
    if points is None:
        points = extract_trajectory_points_from_result(result)
    frame_width, frame_height = extract_frame_size_from_result(result)
    return prepare_safe_trajectory_for_draw(
        points,
        frame_size={"width": frame_width, "height": frame_height},
        pitch_roi=extract_pitch_roi_from_result(result),
        stump_context=result.get("calibration_context") or {},
        impact_info=result.get("impact_info"),
    )


def render_trajectory_validity_section(result, prepared=None):
    """Show path-validity metrics after Detection Health."""
    if prepared is None:
        prepared = prepare_result_path_validity(result)

    validity = prepared.get("validity") or {}
    summary = prepared.get("ui_summary") or {}
    st.subheader("Trajectory Validity")
    metric_cols = st.columns(5)
    metric_cols[0].metric("Path Validity", summary.get("path_validity") or validity.get("quality") or "Unavailable")
    metric_cols[1].metric("Valid Points", summary.get("valid_points", 0))
    metric_cols[2].metric("Rejected Points", summary.get("rejected_points", 0))
    metric_cols[3].metric("Draw Allowed", "Yes" if summary.get("draw_allowed") else "No")
    metric_cols[4].metric(
        "Main Reason",
        summary.get("main_rejection_reason") or validity.get("main_rejection_reason") or "None",
    )
    labels = summary.get("labels") or prepared.get("labels") or []
    if labels:
        st.caption(" · ".join(labels))
    path_mode = summary.get("path_mode") or "full_path"
    if path_mode == "pre_contact_plus_projection":
        st.caption(
            f"Impact frame {summary.get('impact_frame')}: drawing pre-contact path plus "
            f"{summary.get('projected_points', 0)} projected no-contact points."
        )
    elif path_mode == "pre_contact":
        st.caption(
            f"Impact frame {summary.get('impact_frame')}: post-impact detections excluded from path."
        )
    reasons = summary.get("reason_summary") or validity.get("reason_summary") or {}
    if reasons:
        st.caption("Main reasons: " + ", ".join(f"{key}={count}" for key, count in reasons.items()))
    with st.expander("Trajectory Validity Details", expanded=False):
        st.json(
            {
                "validity": validity,
                "ui_summary": summary,
                "projection_used": prepared.get("projection_used"),
                "projected_points": prepared.get("projected_points") or [],
                "impact_frame": prepared.get("impact_frame"),
            }
        )
    return prepared


def resolve_pitch_roi_for_result(result):
    """Prefer the manual pitch ROI stored on the result over auto-detected geometry."""
    result = result or {}
    manual = result.get("manual_pitch_roi")
    if isinstance(manual, dict) and manual.get("available"):
        return manual
    return extract_pitch_roi_from_result(result)


def build_reliable_track_from_result(result):
    """Run the Ball Candidate Reliability Tracker over the raw frame detections."""
    from Backends.src.ball_candidate_tracker import build_reliable_ball_track

    result = result or {}
    frame_detections = (
        result.get("raw_frame_detections")
        or result.get("frame_detections")
        or result.get("impact_frame_detections")
        or []
    )
    frame_width, frame_height = extract_frame_size_from_result(result)
    return build_reliable_ball_track(
        frame_detections,
        frame_size={"width": frame_width, "height": frame_height},
        pitch_roi=resolve_pitch_roi_for_result(result),
    )


def reliable_track_usable(reliable_track):
    """True when the reliable selected track is safe to feed downstream."""
    track = reliable_track or {}
    return (
        track.get("track_quality") in {"Good", "Partial"}
        and len(track.get("track_points") or []) >= 5
    )


def build_physics_trajectory_report_from_result(result, reliable_track=None):
    """Compute (or reuse) the physics-assisted trajectory report for a result payload."""
    result = result or {}
    manual_roi = result.get("manual_pitch_roi")
    manual_roi_active = isinstance(manual_roi, dict) and manual_roi.get("available")
    cached = result.get("physics_trajectory")
    # ponytail: cached report was computed without the manual ROI; recompute when it is on.
    if not manual_roi_active and isinstance(cached, dict) and cached.get("physics_quality"):
        return cached

    from Backends.src.physics_trajectory import build_physics_trajectory_report

    if reliable_track is None:
        reliable_track = build_reliable_track_from_result(result)
    if reliable_track_usable(reliable_track):
        points = reliable_track.get("track_points") or []
        input_source = "Reliable selected ball track"
    else:
        points = extract_trajectory_points_from_result(result)
        input_source = "Current tracker path"

    frame_width, frame_height = extract_frame_size_from_result(result)
    impact_info = result.get("impact_info") or {}
    report = build_physics_trajectory_report(
        points,
        impact_frame=impact_info,
        impact_point=impact_info.get("ball_center"),
        frame_size={"width": frame_width, "height": frame_height},
        pitch_roi=resolve_pitch_roi_for_result(result),
    )
    report["input_path_source"] = input_source
    return report


def physics_path_source(physics_report, path_validity, reliable_track=None):
    """Decide which path source 3D replay should trust, in priority order."""
    physics_report = physics_report or {}
    fitted = physics_report.get("fitted_delivery_path") or []
    validity_quality = (path_validity or {}).get("quality") or "Unavailable"
    track_ok = reliable_track_usable(reliable_track)
    if (
        physics_report.get("physics_quality") in {"Good", "Partial"}
        and len(fitted) >= 5
        and (track_ok or validity_quality not in {"Poor", "Unavailable"})
    ):
        return "Physics fitted delivery path"
    if track_ok:
        return "Reliable selected ball track"
    if validity_quality not in {"Poor", "Unavailable"}:
        return "Current validated tracker path"
    return "Trajectory uncertain"


def render_manual_pitch_calibration_section(result):
    """Simple manual pitch ROI (numeric rectangle) stored only on the current result."""
    from Backends.src.pitch_calibration import normalize_pitch_roi

    result = result or {}
    st.subheader("Manual Pitch Calibration")
    frame_width, frame_height = extract_frame_size_from_result(result)
    enabled = st.checkbox(
        "Enable Manual Pitch ROI",
        value=False,
        key="manual_pitch_roi_enabled",
        help="Rejects ball candidates far outside this rectangle. Applies only to the current analysis result.",
    )
    manual_roi = None
    if enabled:
        roi_cols = st.columns(4)
        x1 = roi_cols[0].number_input(
            "Left X", min_value=0, max_value=frame_width, value=int(frame_width * 0.25),
            step=10, key="manual_pitch_roi_x1",
        )
        x2 = roi_cols[1].number_input(
            "Right X", min_value=0, max_value=frame_width, value=int(frame_width * 0.75),
            step=10, key="manual_pitch_roi_x2",
        )
        y1 = roi_cols[2].number_input(
            "Top Y", min_value=0, max_value=frame_height, value=int(frame_height * 0.2),
            step=10, key="manual_pitch_roi_y1",
        )
        y2 = roi_cols[3].number_input(
            "Bottom Y", min_value=0, max_value=frame_height, value=int(frame_height * 0.95),
            step=10, key="manual_pitch_roi_y2",
        )
        manual_roi = normalize_pitch_roi(
            {"bbox": [x1, y1, x2, y2]},
            frame_size={"width": frame_width, "height": frame_height},
        )
        if not manual_roi.get("available"):
            st.warning("Manual pitch ROI rectangle is invalid; it will be ignored.")
            manual_roi = None
    result["manual_pitch_roi"] = manual_roi
    if manual_roi is not None:
        st.caption("Pitch ROI available: Yes (manual rectangle).")
    elif extract_pitch_roi_from_result(result) is not None:
        st.caption("Pitch ROI available: Yes (auto-detected pitch corridor).")
    else:
        st.caption("Pitch ROI available: No.")
    return manual_roi


def render_ball_candidate_reliability_section(
    result, reliable_track=None, physics_report=None, path_validity=None
):
    """Show Ball Candidate Reliability Tracker metrics before Physics Trajectory."""
    from Backends.src.ball_candidate_tracker import build_ball_candidate_debug_report

    if reliable_track is None:
        reliable_track = build_reliable_track_from_result(result)
    if path_validity is None:
        path_validity = prepare_result_path_validity(result)
    if physics_report is None:
        physics_report = build_physics_trajectory_report_from_result(
            result, reliable_track=reliable_track
        )
    debug_report = build_ball_candidate_debug_report(reliable_track)

    st.subheader("Ball Candidate Reliability")
    metric_cols = st.columns(6)
    metric_cols[0].metric("Track Quality", debug_report["track_quality"])
    metric_cols[1].metric("Selected Ball Points", debug_report["selected_points"])
    metric_cols[2].metric("Rejected Candidates", debug_report["rejected_candidates"])
    metric_cols[3].metric("Missing Frames", len(debug_report["missing_frames"]))
    metric_cols[4].metric("Frames With Candidates", debug_report["frames_with_candidates"])
    metric_cols[5].metric(
        "Path Source", physics_path_source(physics_report, path_validity, reliable_track)
    )
    st.caption(
        "Cleaner selected path for physics/3D only — raw detections stay visible in "
        "Raw Detection Preview and Detection Health."
    )
    with st.expander("Ball Candidate Debug Details", expanded=False):
        st.json(
            {
                **debug_report,
                "track_points": reliable_track.get("track_points") or [],
                "rejected_candidate_details": reliable_track.get("rejected_candidates") or [],
            }
        )
    return reliable_track


def render_physics_trajectory_section(result, path_validity=None, physics_report=None, reliable_track=None):
    """Show physics-assisted trajectory fitter metrics after Trajectory Validity."""
    if path_validity is None:
        path_validity = prepare_result_path_validity(result)
    if reliable_track is None:
        reliable_track = build_reliable_track_from_result(result)
    if physics_report is None:
        physics_report = build_physics_trajectory_report_from_result(
            result, reliable_track=reliable_track
        )
    bounce = physics_report.get("bounce") or {}
    impact = physics_report.get("impact") or {}

    st.subheader("Physics Trajectory")
    metric_cols = st.columns(7)
    metric_cols[0].metric("Physics Quality", physics_report.get("physics_quality") or "Unavailable")
    metric_cols[1].metric("Pre-impact Points", len(physics_report.get("pre_impact_path") or []))
    metric_cols[2].metric("Fitted Path Points", len(physics_report.get("fitted_delivery_path") or []))
    metric_cols[3].metric("Bounce Detected", "Yes" if bounce.get("bounce_detected") else "No")
    metric_cols[4].metric("Projection Quality", physics_report.get("projection_quality") or "Unavailable")
    metric_cols[5].metric("Impact Detected", "Yes" if impact.get("impact_detected") else "No")
    metric_cols[6].metric("Path Source", physics_path_source(physics_report, path_validity, reliable_track))
    if physics_report.get("input_path_source"):
        st.caption(f"Physics fit input: {physics_report['input_path_source']}")

    with st.expander("Physics Trajectory Details", expanded=False):
        st.json(physics_report)
    return physics_report


def extract_impact_point_from_result(result):
    """Pull image-space impact coordinates when available."""
    result = result or {}
    impact = result.get("impact_info") or {}
    point = _coerce_trajectory_point(impact.get("ball_center"))
    if point is not None:
        return point
    return None


def _observer_points_to_xy(points):
    extracted = []
    for item in points or []:
        if isinstance(item, dict):
            point = _coerce_trajectory_point(item)
            if point is not None:
                extracted.append(point)
        else:
            point = _coerce_trajectory_point(item)
            if point is not None:
                extracted.append(point)
    return extracted


def _simple_path_score(points):
    points = _observer_points_to_xy(points)
    if len(points) < 5:
        return 0.0
    penalties = 0.0
    for idx in range(1, len(points)):
        dx = points[idx][0] - points[idx - 1][0]
        dy = points[idx][1] - points[idx - 1][1]
        if dy < -1:
            penalties += 0.25
        if abs(dx) > abs(dy) * 1.6 and abs(dx) > 4:
            penalties += 0.2
    normalized_penalty = min(1.0, penalties / max(1, len(points) - 1))
    return max(0.0, 1.0 - normalized_penalty)


def build_delivery_observer_payload(result):
    result = result or {}
    frame_detections = (
        result.get("raw_frame_detections")
        or result.get("frame_detections")
        or result.get("impact_frame_detections")
        or []
    )
    raw_candidates = extract_ball_candidates_from_frame_detections(frame_detections)
    frame_width, frame_height = extract_frame_size_from_result(result)
    selected = select_best_cricket_path(
        raw_candidates,
        frame_size={"width": frame_width, "height": frame_height},
        pitch_roi=extract_pitch_roi_from_result(result),
        stump_context=result.get("calibration_context") or {},
    )
    fitted = fit_observer_path(
        selected.get("observer_path"),
        frame_size={"width": frame_width, "height": frame_height},
    )
    tracker_points = extract_trajectory_points_from_result(result)
    tracker_score = _simple_path_score(tracker_points)
    observer_score = float(selected.get("path_score") or 0.0)
    comparison = "same"
    if observer_score > tracker_score + 0.05:
        comparison = "better"
    elif observer_score + 0.05 < tracker_score:
        comparison = "worse"
    return {
        "raw_candidates": raw_candidates,
        "selection": selected,
        "fit": fitted,
        "tracker_points": tracker_points,
        "tracker_score": round(float(tracker_score), 4),
        "observer_score": round(float(observer_score), 4),
        "comparison": comparison,
    }


def render_cricket_delivery_observer_section(result):
    payload = build_delivery_observer_payload(result)
    selection = payload["selection"]
    fit = payload["fit"]
    observer_path = selection.get("observer_path") or []
    fitted_path = fit.get("fitted_path") or []
    rejected = selection.get("rejected_candidates") or []
    reason_summary = selection.get("reason_summary") or {}

    st.subheader("Cricket Delivery Observer")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Raw candidates considered", len(payload["raw_candidates"]))
    metric_cols[1].metric("Observer path points", len(observer_path))
    metric_cols[2].metric("Rejected candidates", len(rejected))
    metric_cols[3].metric("Current tracker selected points", len(payload["tracker_points"]))

    quality_cols = st.columns(3)
    quality_cols[0].metric("Observer path quality", selection.get("path_quality", "Unavailable"))
    quality_cols[1].metric("Fit quality", fit.get("fit_quality", "Unavailable"))
    quality_cols[2].metric("Observer vs tracker", payload["comparison"])

    st.caption(
        f"Observer score: {payload['observer_score']:.3f} | "
        f"Tracker score: {payload['tracker_score']:.3f}"
    )
    st.write("Reason summary:", reason_summary if reason_summary else {"none": 0})

    with st.expander("Observer Details", expanded=False):
        rejection_counts = {}
        for item in rejected:
            reason = item.get("reason", "unknown")
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        st.json(
            {
                "raw_candidates_count": len(payload["raw_candidates"]),
                "observer_path_count": len(observer_path),
                "fitted_path_count": len(fitted_path),
                "rejected_candidates_count": len(rejected),
                "rejection_counts_by_reason": rejection_counts,
                "path_quality": selection.get("path_quality"),
                "fit_quality": fit.get("fit_quality"),
                "comparison": payload["comparison"],
                "observer_score": payload["observer_score"],
                "tracker_score": payload["tracker_score"],
                "reason_summary": reason_summary,
                "notes": {
                    "selection_notes": selection.get("notes") or [],
                    "fit_notes": fit.get("notes") or [],
                },
                "observer_path": observer_path,
                "fitted_path": fitted_path,
            }
        )
    return payload


def render_3d_replay_section(
    result, observer_payload=None, path_validity=None, physics_report=None, reliable_track=None
):
    try:
        from Backends.src.replay3d.replay_renderer import build_3d_replay_figure
        from Backends.src.replay3d.stump_calibration import build_stump_calibration_context
        from Backends.src.replay3d.trajectory_3d import build_estimated_3d_trajectory
    except ImportError as exc:
        st.warning(f"3D trajectory replay unavailable: {exc}")
        return

    st.subheader("CricVision 3D Trajectory Replay")

    tracker_points = extract_trajectory_points_from_result(result)
    if path_validity is None:
        path_validity = prepare_result_path_validity(result, trajectory_points=tracker_points)

    validity = path_validity.get("validity") or {}
    path_quality = validity.get("quality") or path_validity.get("quality") or "Unavailable"
    rejected_count = path_validity.get("rejected_count", len(validity.get("rejected_points") or []))
    main_reason = path_validity.get("main_rejection_reason") or validity.get("main_rejection_reason")

    validity_cols = st.columns(3)
    validity_cols[0].metric("Path Validity", path_quality)
    validity_cols[1].metric("Rejected Points", rejected_count)
    validity_cols[2].metric("Main Rejection Reason", main_reason or "None")

    # Priority: physics fitted path, then reliable selected track, then validated tracker.
    if reliable_track is None:
        reliable_track = build_reliable_track_from_result(result)
    if physics_report is None:
        physics_report = build_physics_trajectory_report_from_result(
            result, reliable_track=reliable_track
        )
    trajectory_points = []
    source_text = None
    preferred_source = physics_path_source(physics_report, path_validity, reliable_track)
    if preferred_source == "Physics fitted delivery path":
        physics_points = _observer_points_to_xy(physics_report.get("fitted_delivery_path"))
        if len(physics_points) >= 5:
            trajectory_points = physics_points
            source_text = "Physics fitted delivery path"
    if source_text is None and reliable_track_usable(reliable_track):
        track_points = _observer_points_to_xy(reliable_track.get("track_points"))
        if len(track_points) >= 5:
            trajectory_points = track_points
            source_text = "Reliable selected ball track"
    if source_text is None and path_quality not in {"Unavailable", "Poor"}:
        tracker_xy = list(path_validity.get("valid_xy") or [])
        if len(tracker_xy) >= 5:
            trajectory_points = tracker_xy
            source_text = "Current validated tracker path"
    if source_text is None:
        st.info("3D Replay unavailable: trajectory uncertain")
        with st.expander("Path validity notes", expanded=False):
            st.write("\n".join(validity.get("notes") or []))
            st.json(validity)
        return

    if source_text == "Current validated tracker path" and observer_payload:
        selection = observer_payload.get("selection") or {}
        fit = observer_payload.get("fit") or {}
        fitted_points = _observer_points_to_xy(fit.get("fitted_path"))
        if (
            selection.get("path_quality") in {"Good", "Partial"}
            and len(fitted_points) >= 5
            and float(selection.get("path_score") or 0.0)
            > float(observer_payload.get("tracker_score") or 0.0) + 0.05
        ):
            observer_prepared = prepare_result_path_validity(result, trajectory_points=fitted_points)
            observer_quality = observer_prepared.get("quality") or "Unavailable"
            if observer_quality in {"Good", "Partial"} and observer_prepared.get("valid_xy"):
                trajectory_points = list(observer_prepared.get("valid_xy") or [])
                source_text = "Validated observer fitted path"
                path_validity = observer_prepared
                validity = observer_prepared.get("validity") or validity
                path_quality = observer_quality

    st.caption(f"3D Replay path source: {source_text}")
    labels = path_validity.get("labels") or []
    if labels:
        st.caption(" · ".join(labels))
    if path_validity.get("projection_used"):
        st.caption(
            "Projected continuation (no bat contact) appended after impact; "
            "post-impact real detections are excluded."
        )
    if path_quality == "Partial":
        st.caption("Partial / estimated — using cricket-validity filtered points only.")

    if len(trajectory_points) < 5:
        st.info("3D Replay unavailable: trajectory uncertain")
        return

    default_view = (result or {}).get("camera_view") or "unknown"
    view_options = ["unknown", "umpire_end", "batter_view", "bowler_end", "side_view"]
    if default_view not in view_options:
        view_options.insert(0, default_view)

    control_cols = st.columns(2)
    with control_cols[0]:
        camera_view = st.selectbox(
            "Camera view (3D replay)",
            options=view_options,
            index=view_options.index(default_view) if default_view in view_options else 0,
            key="replay3d_camera_view",
        )
    with control_cols[1]:
        camera_height_ft = st.number_input(
            "Estimated camera height (ft)",
            min_value=4.0,
            max_value=30.0,
            value=8.0,
            step=0.5,
            key="replay3d_camera_height_ft",
        )

    try:
        frame_width, frame_height = extract_frame_size_from_result(result)
        calibration_context = build_stump_calibration_context(
            frame_size={"width": frame_width, "height": frame_height},
            stump_detections=extract_stump_detections_from_result(result),
            pitch_roi=extract_pitch_roi_from_result(result),
            camera_height_ft=camera_height_ft,
            camera_view=camera_view,
        )
        trajectory_3d = build_estimated_3d_trajectory(
            trajectory_points,
            calibration_context,
            bounce_point=extract_bounce_point_from_result(result),
            impact_point=extract_impact_point_from_result(result),
        )
        render_payload = build_3d_replay_figure(trajectory_3d, calibration_context)
    except Exception as exc:
        st.warning(f"3D trajectory replay failed: {exc}")
        return

    if not trajectory_3d.get("available") or not render_payload.get("available"):
        st.info(
            "Estimated 3D replay unavailable for this clip. "
            "Try a longer tracked delivery or clearer stump/pitch context."
        )
        with st.expander("3D replay notes", expanded=False):
            st.write("\n".join(trajectory_3d.get("notes") or []))
        return

    metrics = trajectory_3d.get("metrics") or {}
    metric_cols = st.columns(4)
    metric_cols[0].metric("Speed", metrics.get("speed_kmh", "Not calibrated"))
    metric_cols[1].metric("Swing", metrics.get("swing", "Unknown"))
    metric_cols[2].metric("Spin", metrics.get("spin", "Unknown"))
    metric_cols[3].metric("LBW", metrics.get("lbw", "Not available"))

    caption = render_payload.get("caption") or (
        "Estimated 3D Replay — based on tracked video points and stump/pitch calibration."
    )
    if path_quality == "Partial":
        caption = f"Partial / estimated — {caption}"
    if render_payload.get("backend") == "plotly" and render_payload.get("figure") is not None:
        st.plotly_chart(render_payload["figure"], use_container_width=True)
    elif render_payload.get("image") is not None:
        st.image(
            cv2.cvtColor(render_payload["image"], cv2.COLOR_BGR2RGB),
            caption=caption,
            use_container_width=True,
        )
    else:
        st.info("Estimated 3D replay unavailable: renderer returned no output.")
        return

    st.caption(caption)
    st.caption(
        f"Trajectory quality: {trajectory_3d.get('trajectory_quality', 'Unknown')} | "
        f"Path validity: {path_quality} | "
        f"Calibration: {calibration_context.get('calibration_quality', 'Unknown')}"
    )
    with st.expander("3D replay notes", expanded=False):
        st.write("\n".join(trajectory_3d.get("notes") or []))


def _format_health_rate(value):
    if value is None:
        return "Unknown"
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return "Unknown"
    if rate <= 1.0:
        rate *= 100.0
    return f"{rate:.1f}%"


def render_detection_health_section(result):
    try:
        from Backends.src.config.constants import DETECTION_PRESETS
        from Backends.src.detection_health import build_detection_health
    except ImportError as exc:
        st.warning(f"Detection health unavailable: {exc}")
        return

    settings = st.session_state.get("video_analysis_settings", {})
    preset_name = settings.get("preset_name") or result.get("active_preset") or "Balanced Mode"
    preset = DETECTION_PRESETS.get(preset_name, {})

    health = build_detection_health(
        result,
        model_name=settings.get("selected_model_name") or result.get("active_model"),
        detection_preset=preset_name,
        speed_mode=settings.get("speed_mode") or result.get("speed_mode"),
        confidence_threshold=preset.get("confidence"),
        imgsz=preset.get("imgsz"),
    )

    st.subheader("Detection Health")
    metric_cols = st.columns(6)
    metric_cols[0].metric("Ball Detection Rate", _format_health_rate(health.get("ball_detection_rate")))
    metric_cols[1].metric("Ball Tracking Rate", _format_health_rate(health.get("ball_tracking_rate")))
    metric_cols[2].metric("Raw Ball Detections", health.get("raw_ball_detections", 0))
    metric_cols[3].metric("Selected Ball Points", health.get("selected_ball_points", 0))
    metric_cols[4].metric(
        "Overall Tracking Quality",
        health.get("overall_tracking_quality") or "Unknown",
    )
    metric_cols[5].metric("Failure Type", health.get("failure_type") or "unknown")

    with st.expander("Detection Health Details", expanded=False):
        st.json(health)


def show_batting_analysis_results(result):
    from Backends.src.ui.components import render_video_analysis_results_layout

    render_detection_health_section(result)
    path_validity = render_trajectory_validity_section(result)
    render_manual_pitch_calibration_section(result)
    reliable_track = build_reliable_track_from_result(result)
    physics_report = build_physics_trajectory_report_from_result(result, reliable_track=reliable_track)
    render_ball_candidate_reliability_section(
        result,
        reliable_track=reliable_track,
        physics_report=physics_report,
        path_validity=path_validity,
    )
    render_physics_trajectory_section(
        result,
        path_validity=path_validity,
        physics_report=physics_report,
        reliable_track=reliable_track,
    )
    render_trajectory_replay_section(result, path_validity=path_validity)
    observer_payload = render_cricket_delivery_observer_section(result)
    render_3d_replay_section(
        result,
        observer_payload=observer_payload,
        path_validity=path_validity,
        physics_report=physics_report,
        reliable_track=reliable_track,
    )
    render_video_analysis_results_layout(result, context_label="Video Analysis")


def show_video_analysis_results(result, selected_model_name, preset_name, show_pitch_roi):
    from Backends.src.ui.components import render_video_analysis_results_layout

    render_detection_health_section(result)
    path_validity = render_trajectory_validity_section(result)
    render_manual_pitch_calibration_section(result)
    reliable_track = build_reliable_track_from_result(result)
    physics_report = build_physics_trajectory_report_from_result(result, reliable_track=reliable_track)
    render_ball_candidate_reliability_section(
        result,
        reliable_track=reliable_track,
        physics_report=physics_report,
        path_validity=path_validity,
    )
    render_physics_trajectory_section(
        result,
        path_validity=path_validity,
        physics_report=physics_report,
        reliable_track=reliable_track,
    )
    render_trajectory_replay_section(result, path_validity=path_validity)
    observer_payload = render_cricket_delivery_observer_section(result)
    render_3d_replay_section(
        result,
        observer_payload=observer_payload,
        path_validity=path_validity,
        physics_report=physics_report,
        reliable_track=reliable_track,
    )
    render_video_analysis_results_layout(
        result,
        context_label="Video Analysis",
        show_status_banner=True,
    )


def show_video_analysis_page():
    from Backends.src.ui.components import (
        clean_upload_box,
        render_calibration_context_card,
    )
    from Backends.src.ui.theme import render_empty_state, render_page_header

    render_page_header(
        "Analyze",
        "Upload a delivery clip. CricVision uses smart defaults and generates a processed video plus professional report.",
    )

    if "video_analysis_result" not in st.session_state:
        st.session_state.video_analysis_result = None
    if "video_analysis_settings" not in st.session_state:
        st.session_state.video_analysis_settings = {}

    model_options = get_model_options()

    from Backends.src.ui.interactive_field_map import render_field_setup_card

    field_setup = render_field_setup_card(key_prefix="video_analysis_field", compact=True, default_preset="Balanced")

    st.subheader("Practice Environment Calibration")
    calibration_enabled = st.checkbox(
        "Enable practice environment calibration",
        value=True,
        key="practice_calibration_enabled",
        help="Adds approximate 2D stump, crease, pitch-corridor, and line references.",
    )
    calibration_cols = st.columns(2)
    camera_view_labels = {
        "Umpire End": "umpire_end",
        "Batter View": "batter_view",
        "Bowler End": "bowler_end",
        "Side View": "side_view",
        "Unknown": "unknown",
    }
    handedness_labels = {
        "Right-handed": "right",
        "Left-handed": "left",
        "Unknown": "unknown",
    }
    with calibration_cols[0]:
        calibration_camera_label = st.selectbox(
            "Camera view",
            list(camera_view_labels),
            index=0,
            key="practice_calibration_camera_view",
            disabled=not calibration_enabled,
        )
    with calibration_cols[1]:
        calibration_handedness_label = st.selectbox(
            "Batter handedness",
            list(handedness_labels),
            index=0,
            key="practice_calibration_handedness",
            disabled=not calibration_enabled,
        )
    calibration_auto_estimate = st.checkbox(
        "Auto-estimate stumps and pitch corridor from analysis detections",
        value=True,
        key="practice_calibration_auto_estimate",
        disabled=not calibration_enabled,
        help="Uses stump detections already produced after Analyze is clicked; it does not run another model.",
    )
    calibration_confirmed = st.checkbox(
        "Confirm calibration for analysis",
        value=True,
        key="practice_calibration_confirmed",
        disabled=not calibration_enabled,
    )
    practice_calibration_context = build_calibration_context(
        {
            "enabled": calibration_enabled and calibration_confirmed,
            "confirmed": calibration_confirmed,
            "auto_estimate": calibration_auto_estimate,
            "camera_view": camera_view_labels[calibration_camera_label],
            "batter_handedness": handedness_labels[
                calibration_handedness_label
            ],
            "notes": (
                [
                    "Provisional geometry will be refined from existing stump "
                    "detections during analysis."
                ]
                if calibration_enabled and calibration_confirmed
                else []
            ),
        }
    )
    render_calibration_context_card(
        practice_calibration_context,
        compact=True,
    )

    clean_upload_box("Upload cricket video")
    uploaded_video = st.file_uploader(
        "Upload delivery video",
        type=["mp4", "mov", "avi", "mkv"],
        key="video_analysis_upload",
        label_visibility="collapsed",
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

    analyze_clicked = st.button(
        "Analyze Delivery",
        type="primary",
        use_container_width=True,
        disabled=uploaded_video is None,
        key="analyze_video_button",
    )

    speed_mode = st.selectbox(
        "Analysis Mode",
        ["Smart Balanced", "Smart Accurate", "Debug Full Frame"],
        index=0,
        key="video_analysis_speed_mode",
        help="Smart Balanced keeps ball detection on every frame but reduces wasted work from other models.",
    )

    generate_processed_video = st.checkbox(
        "Generate processed video preview",
        value=True,
        key="video_analysis_generate_processed_video",
        help="Disable to run analysis and reports only without writing an annotated video.",
    )

    overlay_detail = st.selectbox(
        "Overlay detail",
        ["Clean", "Debug"],
        index=0,
        key="video_analysis_overlay_detail",
        help="Clean keeps ball trail and key markers only. Debug shows ROI, bounce, and labels.",
    )

    with st.expander("Raw Detection Preview", expanded=False):
        st.caption(
            "YOLO-only sampled frames before Kalman, interpolation, or Visual Observer repair. "
            "This does not change Analyze Delivery results or processed video output."
        )
        run_raw_preview_clicked = st.button(
            "Run Raw Detection Preview",
            key="video_analysis_run_raw_preview",
            disabled=uploaded_video is None,
        )

    with st.expander("Advanced Settings", expanded=False):
        analysis_mode = st.selectbox(
            "Analysis mode",
            ["Bowling Analysis", "Batting Analysis", "Full Delivery Analysis"],
            index=2,
            key="video_analysis_mode",
        )
        selected_bat_model_key = None
        selected_ball_model_key = "current_best"
        selected_model_key = "current_best"

        if analysis_mode == "Batting Analysis":
            batting_ball_options = {
                "Current Best Ball + Stump Model": "current_best",
                "CricShot10k Ball Detector": "cricshot_ball",
            }
            selected_model_name = st.selectbox(
                "Ball model",
                list(batting_ball_options),
                key="video_analysis_batting_ball_model",
            )
            selected_ball_model_key = batting_ball_options[selected_model_name]
            selected_model_path = get_model_path(selected_ball_model_key)
            use_ensemble = False
            selected_bat_model_key = "cricshot_bat"
            st.selectbox("Bat model", ["CricShot10k Bat Detector"], key="video_analysis_bat_model")
        else:
            selected_model_name = st.selectbox(
                "Detection model",
                list(model_options.keys()),
                key="video_analysis_model",
            )
            selected_model = model_options[selected_model_name]
            selected_model_path = selected_model["path"]
            selected_model_key = selected_model.get("model_key")
            use_ensemble = selected_model.get("ensemble", False)
            if analysis_mode == "Full Delivery Analysis":
                selected_bat_model_key = "cricshot_bat"
                st.selectbox("Bat model", ["CricShot10k Bat Detector"], key="video_analysis_full_bat_model")

        preset_name = st.selectbox(
            "Detection preset",
            list(DETECTION_PRESETS.keys()),
            index=1,
            key="video_analysis_preset",
        )
        active_preset = DETECTION_PRESETS[preset_name]
        confidence = active_preset["confidence"]
        image_size = active_preset["imgsz"]

        show_pitch_roi = st.checkbox("Show pitch ROI overlay", value=False, key="video_analysis_show_roi")
        calibration_mode = st.radio(
            "Pitch calibration",
            [
                "Auto calibration using detected stumps",
                "Manual calibration using 4 pitch corner points",
            ],
            index=0,
            key="video_analysis_calibration_mode",
        )
        shot_trajectory_mode = st.radio(
            "Shot direction trajectory",
            [
                "Use full trajectory",
                "Use last part of trajectory",
                "Manually mark bat contact frame",
            ],
            index=1,
            key="video_analysis_shot_trajectory_mode",
        )
        manual_contact_frame = None
        if shot_trajectory_mode == "Manually mark bat contact frame":
            manual_contact_frame = st.number_input(
                "Bat contact frame",
                min_value=0,
                value=0,
                step=1,
                key="video_analysis_bat_contact_frame",
            )

        with st.expander("Model Status", expanded=False):
            for status in validate_model_paths().values():
                st.write(f"{status['status']}: {status['name']}")

        with st.expander("Advanced analysis settings", expanded=False):
            limit_frames_enabled = st.checkbox(
                "Limit frames for testing",
                value=False,
                key="video_analysis_limit_frames_enabled",
            )
            frame_limit_choice = st.selectbox(
                "Frame limit",
                [50, 100, 200, "All frames"],
                index=3,
                key="video_analysis_frame_limit_choice",
                disabled=not limit_frames_enabled,
            )
            st.checkbox(
                "Show performance details",
                value=False,
                key="video_analysis_show_performance",
            )

    analysis_mode = st.session_state.get("video_analysis_mode", "Full Delivery Analysis")
    preset_name = st.session_state.get("video_analysis_preset", "Balanced Mode")
    active_preset = DETECTION_PRESETS[preset_name]
    confidence = active_preset["confidence"]
    image_size = active_preset["imgsz"]
    show_pitch_roi = st.session_state.get("video_analysis_show_roi", False)
    calibration_mode = st.session_state.get(
        "video_analysis_calibration_mode",
        "Auto calibration using detected stumps",
    )
    shot_trajectory_mode = st.session_state.get(
        "video_analysis_shot_trajectory_mode",
        "Use last part of trajectory",
    )
    manual_contact_frame = st.session_state.get("video_analysis_bat_contact_frame", 0)
    if shot_trajectory_mode != "Manually mark bat contact frame":
        manual_contact_frame = None

    from Backends.src.analysis.analysis_speed import resolve_frame_limit

    speed_mode = st.session_state.get("video_analysis_speed_mode", "Smart Balanced")
    generate_processed_video = st.session_state.get("video_analysis_generate_processed_video", True)
    overlay_detail = st.session_state.get("video_analysis_overlay_detail", "Clean")
    max_frames = resolve_frame_limit(
        st.session_state.get("video_analysis_limit_frames_enabled", False),
        st.session_state.get("video_analysis_frame_limit_choice", "All frames"),
    )
    show_performance = st.session_state.get("video_analysis_show_performance", False)

    if analysis_mode == "Batting Analysis":
        batting_ball_options = {
            "Current Best Ball + Stump Model": "current_best",
            "CricShot10k Ball Detector": "cricshot_ball",
        }
        selected_model_name = st.session_state.get(
            "video_analysis_batting_ball_model",
            "Current Best Ball + Stump Model",
        )
        selected_ball_model_key = batting_ball_options.get(selected_model_name, "current_best")
        selected_model_path = get_model_path(selected_ball_model_key)
        selected_model_key = selected_ball_model_key
        use_ensemble = False
        selected_bat_model_key = "cricshot_bat"
    else:
        selected_model_name = st.session_state.get(
            "video_analysis_model",
            list(model_options.keys())[0],
        )
        selected_model = model_options.get(
            selected_model_name,
            list(model_options.values())[0],
        )
        selected_model_path = selected_model["path"]
        selected_model_key = selected_model.get("model_key")
        use_ensemble = selected_model.get("ensemble", False)
        selected_bat_model_key = "cricshot_bat" if analysis_mode == "Full Delivery Analysis" else None

    manual_pitch_points = None

    if run_raw_preview_clicked and uploaded_video is not None:
        uploaded_video.seek(0)
        preview_bytes = uploaded_video.read()
        if not preview_bytes:
            st.error("The uploaded video is empty. Choose a non-empty cricket clip.")
        else:
            upload_suffix = Path(uploaded_video.name or "").suffix.lower()
            if upload_suffix not in {".mp4", ".mov", ".avi", ".mkv"}:
                upload_suffix = ".mp4"
            try:
                from Backends.src.detection_health import run_raw_detection_preview
            except ImportError as exc:
                st.warning(f"Raw detection preview unavailable: {exc}")
            else:
                with TemporaryDirectory(prefix="cricvision_raw_preview_") as preview_temp_dir:
                    preview_video_path = Path(preview_temp_dir) / f"preview_upload{upload_suffix}"
                    preview_video_path.write_bytes(preview_bytes)
                    with st.spinner("Running raw detection preview..."):
                        preview_result = run_raw_detection_preview(
                            preview_video_path,
                            model_key=selected_model_key,
                            model_path=selected_model_path,
                            use_ensemble=use_ensemble,
                            confidence=confidence,
                            imgsz=image_size,
                            speed_mode=speed_mode,
                            detection_preset=preset_name,
                        )

                if not preview_result.get("success"):
                    st.error(preview_result.get("error", "Raw detection preview failed."))
                else:
                    preview_cols = st.columns(4)
                    preview_cols[0].metric("Sampled Frames", preview_result.get("sampled_frames", 0))
                    preview_cols[1].metric("Frames With Ball", preview_result.get("frames_with_ball", 0))
                    preview_cols[2].metric(
                        "Raw Ball Detections",
                        preview_result.get("raw_ball_detections", 0),
                    )
                    avg_conf = preview_result.get("average_confidence")
                    preview_cols[3].metric(
                        "Average Confidence",
                        f"{avg_conf:.2f}" if avg_conf is not None else "Unknown",
                    )
                    st.caption(
                        f"Model: {preview_result.get('model_path', 'Unknown')} | "
                        f"Preset: {preview_result.get('detection_preset', 'Unknown')} | "
                        f"Confidence: {preview_result.get('confidence_threshold', 'Unknown')} | "
                        f"imgsz: {preview_result.get('imgsz', 'Unknown')}"
                    )
                    for frame_item in preview_result.get("frames", []):
                        image_bgr = frame_item.get("image_bgr")
                        if image_bgr is None:
                            continue
                        st.image(
                            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                            caption=f"Frame {frame_item.get('frame_index', '?')} — raw YOLO ball detections",
                            use_container_width=True,
                        )

    if analyze_clicked and uploaded_video is not None:
        if calibration_mode.startswith("Manual"):
            first_frame = extract_first_video_frame(uploaded_video)
            if first_frame is None:
                st.warning("Could not read the first frame for manual calibration.")
            else:
                with st.expander("Manual Pitch Calibration", expanded=True):
                    manual_pitch_points = show_manual_pitch_point_inputs(first_frame)

        uploaded_video.seek(0)
        uploaded_bytes = uploaded_video.read()
        if not uploaded_bytes:
            st.error("The uploaded video is empty. Choose a non-empty cricket clip.")
            st.session_state.video_analysis_result = None
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        PROCESSED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        raw_output_path = PROCESSED_VIDEO_DIR / f"raw_cricvision_analysis_{timestamp}.mp4"
        browser_output_path = PROCESSED_VIDEO_DIR / f"cricvision_analysis_{timestamp}.mp4"

        upload_suffix = Path(uploaded_video.name or "").suffix.lower()
        if upload_suffix not in {".mp4", ".mov", ".avi", ".mkv"}:
            upload_suffix = ".mp4"
        try:
            with TemporaryDirectory(prefix="cricvision_upload_") as temp_dir:
                input_video_path = Path(temp_dir) / f"uploaded_video{upload_suffix}"
                input_video_path.write_bytes(uploaded_bytes)

                with st.spinner("Analyzing delivery..."):
                    if analysis_mode == "Batting Analysis":
                        result = process_batting_video(
                            video_path=input_video_path,
                            output_path=raw_output_path,
                            ball_model_key=selected_ball_model_key,
                            bat_model_key=selected_bat_model_key,
                            confidence=confidence,
                            speed_mode=speed_mode,
                            max_frames=max_frames,
                            generate_processed_video=generate_processed_video,
                            calibration_context=practice_calibration_context,
                            overlay_detail=overlay_detail,
                        )
                    else:
                        result = process_video(
                            video_path=input_video_path,
                            output_path=raw_output_path,
                            model_path=selected_model_path,
                            model_key=selected_model_key,
                            confidence=confidence,
                            imgsz=image_size,
                            use_ensemble=use_ensemble,
                            show_pitch_roi=show_pitch_roi,
                            calibration_mode=calibration_mode,
                            manual_pitch_points=manual_pitch_points,
                            shot_trajectory_mode=shot_trajectory_mode,
                            manual_contact_frame=manual_contact_frame,
                            field_setup=field_setup,
                            bat_model_key=selected_bat_model_key,
                            speed_mode=speed_mode,
                            max_frames=max_frames,
                            generate_processed_video=generate_processed_video,
                            calibration_context=practice_calibration_context,
                            overlay_detail=overlay_detail,
                        )
                    result["analysis_mode"] = analysis_mode
                    result["active_preset"] = preset_name
                    result["active_model"] = selected_model_name
                    result["ball_model_used"] = selected_model_name
                    result["show_performance_details"] = show_performance
        except Exception as error:
            print(f"Video analysis failed: {type(error).__name__}: {error}")
            raw_output_path.unlink(missing_ok=True)
            browser_output_path.unlink(missing_ok=True)
            st.error(
                "Video analysis could not complete. Check that the clip is readable "
                "and uses a supported codec, then try again."
            )
            st.session_state.video_analysis_result = None
            return

        if not result.get("success"):
            raw_output_path.unlink(missing_ok=True)
            browser_output_path.unlink(missing_ok=True)
            st.error(result.get("error", "Video analysis did not complete."))
            st.session_state.video_analysis_result = None
        else:
            result["raw_output_path"] = (
                str(raw_output_path)
                if result.get("processed_video_generated") and raw_output_path.exists()
                else None
            )
            if result.get("processed_video_generated") and result.get("output_path"):
                try:
                    final_video_path = convert_to_browser_mp4(
                        input_path=result["output_path"],
                        output_path=browser_output_path,
                    )
                    result["output_path"] = str(final_video_path)
                    result["processed_video_conversion"] = "converted"
                except Exception as conv_error:
                    result["output_path"] = result.get("raw_output_path")
                    result["processed_video_conversion"] = "failed"
                    result["processed_video_conversion_error"] = str(conv_error)
                    st.warning(
                        "Processed video preview conversion failed. "
                        "Analysis results are still available and the raw video can be downloaded."
                    )
            else:
                result["output_path"] = None
            try:
                if analysis_mode in {"Batting Analysis", "Full Delivery Analysis"}:
                    save_batting_report(result, analysis_mode)
                video_name = uploaded_video.name if uploaded_video is not None else None
                _persist_result_to_session(result, "Video Analysis", video_name=video_name)
                st.session_state.video_analysis_result = result
                st.session_state.video_analysis_settings = {
                    "analysis_mode": analysis_mode,
                    "selected_model_name": selected_model_name,
                    "preset_name": preset_name,
                    "show_pitch_roi": show_pitch_roi,
                    "shot_trajectory_mode": shot_trajectory_mode,
                    "speed_mode": speed_mode,
                    "generate_processed_video": generate_processed_video,
                    "overlay_detail": overlay_detail,
                    "show_performance_details": show_performance,
                }
                st.success("Analysis complete.")
            except Exception as error:
                raw_output_path.unlink(missing_ok=True)
                browser_output_path.unlink(missing_ok=True)
                st.error(f"Could not save analysis results: {error}")
                st.session_state.video_analysis_result = None

    result = st.session_state.video_analysis_result
    settings = st.session_state.video_analysis_settings

    if result is None or not result.get("success"):
        render_empty_state(
            "No analysis yet",
            "Upload a clip and click Analyze Delivery to generate a processed video and report.",
            action_label="Smart defaults are applied automatically",
        )
    elif result.get("analysis_mode") == "Batting Analysis":
        show_batting_analysis_results(result)
    else:
        show_video_analysis_results(
            result=result,
            selected_model_name=settings.get("selected_model_name", result.get("active_model", "Unknown")),
            preset_name=settings.get("preset_name", result.get("active_preset", "Balanced Mode")),
            show_pitch_roi=settings.get("show_pitch_roi", False),
        )
