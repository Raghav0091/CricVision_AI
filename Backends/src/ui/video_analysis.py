import tempfile
import json
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from Backends.src.utils.cv2_loader import cv2

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
from Backends.src.video_pipeline import detection_pipeline as shared_detection
from Backends.src.video_pipeline import annotation_writer as shared_annotations
from Backends.src.video_pipeline.performance_timer import (
    create_performance_profile,
    finish_performance_profile,
)
from Backends.src.video_pipeline.report_pipeline import timed_video_reports
from Backends.src.video_pipeline.video_reader import (
    extract_first_video_frame as read_first_video_frame,
)


OUTPUT_DIR = Path("outputs/video_analysis")
PROCESSED_VIDEO_DIR = Path("outputs/processed_videos")
REPORTS_DIR = Path("outputs/reports")
REVIEW_FRAMES_DIR = Path("outputs/review_frames")
MAX_REVIEW_FRAMES_PER_ANALYSIS = 80
DETECTION_PRESETS = {
    "Fast Bowling Mode": {
        "imgsz": 960,
        "confidence": 0.15,
    },
    "Balanced Mode": {
        "imgsz": 768,
        "confidence": 0.25,
    },
    "High Precision Mode": {
        "imgsz": 960,
        "confidence": 0.35,
    },
}

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
extract_first_video_frame = read_first_video_frame


def _persist_result_to_session(result, source_type, video_name=None):
    from Backends.src.ui.analysis_helpers import persist_result_to_session

    return persist_result_to_session(result, source_type, video_name=video_name)


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

        _draw_ball_detections(annotated_frame, ball_detections)
        draw_bat_detections(annotated_frame, bat_detections)
        if not light_annotation:
            for index in range(1, len(trajectory[-35:])):
                recent = trajectory[-35:]
                cv2.line(annotated_frame, recent[index - 1], recent[index], (0, 255, 255), 3)
        elif len(trajectory) >= 2:
            cv2.line(annotated_frame, trajectory[-2], trajectory[-1], (0, 255, 255), 3)

        if writer is not None:
            annotation_started = time.perf_counter()
            writer.write(annotated_frame)
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
    reports = timed_video_reports(
        frame_detections,
        fps=fps,
        total_frames=frame_index,
        batter_handedness=None,
        impact_result=impact_info,
    )
    impact_info = reports["impact_result"]
    shot_info = reports["shot_result"]
    direction_info = reports["direction_result"]
    outcome_info = reports["outcome_result"]
    agent_info = reports["agent_result"]
    enrichment = reports["enrichment"]
    observer_timeline = reports["observer_timeline"]
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
        "frame_detections": frame_detections,
        "impact_frame_detections": frame_detections,
        "observer_timeline": observer_timeline,
        "performance_profile": performance,
        "speed_mode": speed_mode,
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


def ensure_delivery_report_fields(result):
    from Backends.src.ui.analysis_helpers import ensure_delivery_report_fields as apply_defaults

    apply_defaults(result)


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

            if show_pitch_roi:
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

                if show_pitch_roi:
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

        for i in range(1, len(trajectory_points)):
            cv2.line(
                annotated_frame,
                trajectory_points[i - 1],
                trajectory_points[i],
                (0, 255, 255),
                3,
            )

        if estimated_bounce_point is not None:
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

        if not light_annotation:
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
            writer.write(annotated_frame)
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
    reports = timed_video_reports(
        frame_detections,
        fps=fps,
        total_frames=frame_index,
        batter_handedness=batter_handedness,
        delivery_report=delivery_report,
        impact_result=impact_info,
    )
    impact_info = reports["impact_result"]
    shot_info = reports["shot_result"]
    direction_info = reports["direction_result"]
    outcome_info = reports["outcome_result"]
    agent_info = reports["agent_result"]
    enrichment = reports["enrichment"]
    observer_timeline = reports["observer_timeline"]
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

    return {
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
        "frame_detections": frame_detections,
        "impact_frame_detections": frame_detections,
        "observer_timeline": observer_timeline,
        "performance_profile": performance,
        "speed_mode": speed_mode,
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


def show_batting_analysis_results(result):
    from Backends.src.ui.components import (
        render_delivery_report,
        render_impact_frame_preview,
        render_impact_report,
        render_observer_timeline_report,
        render_outcome_prediction,
        render_performance_details,
        render_save_status,
        render_shot_direction_report,
        render_shot_report,
        render_vision_agent_report,
        video_preview_card,
    )

    video_preview_card("Processed Video Preview")
    output_path = result.get("output_path")
    if output_path:
        with open(output_path, "rb") as video_file:
            video_bytes = video_file.read()
        st.video(video_bytes)
        with open(output_path, "rb") as video_file:
            st.download_button(
                "Download Processed Video",
                data=video_file,
                file_name=Path(output_path).name,
                mime="video/mp4",
                use_container_width=True,
                key="download_batting_processed_video",
            )
    elif result.get("processed_video_skipped") or not result.get("processed_video_generated", True):
        st.info("Processed video generation skipped to speed up analysis.")
    else:
        st.warning("Processed video preview is not available for this result.")

    render_observer_timeline_report(result)
    render_delivery_report(result)
    render_impact_report(result)
    render_impact_frame_preview(result)
    render_shot_report(result)
    render_shot_direction_report(result)
    render_outcome_prediction(result)
    render_vision_agent_report(result)
    render_performance_details(result)
    render_save_status(result, "Video Analysis")


def show_video_analysis_results(result, selected_model_name, preset_name, show_pitch_roi):
    from Backends.src.ui.components import (
        render_delivery_report,
        render_impact_frame_preview,
        render_impact_report,
        render_observer_timeline_report,
        render_outcome_prediction,
        render_performance_details,
        render_save_status,
        render_shot_direction_report,
        render_shot_report,
        render_vision_agent_report,
        video_preview_card,
    )
    from Backends.src.ui.theme import render_status_pill

    st.markdown(
        f'<div style="margin:0.75rem 0 1rem 0;">{render_status_pill("Analysis Complete", "success")} '
        f'{render_status_pill(result.get("analysis_mode", "Full Delivery Analysis"), "gold")}</div>',
        unsafe_allow_html=True,
    )

    video_preview_card("Processed Video Preview")
    output_path = result.get("output_path")
    if output_path:
        with open(output_path, "rb") as video_file:
            video_bytes = video_file.read()
        st.video(video_bytes)

        with open(output_path, "rb") as file:
            st.download_button(
                label="Download Processed Video",
                data=file,
                file_name="cricvision_processed_video.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
    elif result.get("processed_video_skipped") or not result.get("processed_video_generated", True):
        st.info("Processed video generation skipped to speed up analysis.")
    else:
        st.warning("Processed video preview is not available for this result.")

    render_observer_timeline_report(result)
    render_delivery_report(result)
    render_impact_report(result)
    render_impact_frame_preview(result)
    render_shot_report(result)
    render_shot_direction_report(result)
    render_outcome_prediction(result)
    render_vision_agent_report(result)
    render_performance_details(result)
    render_save_status(result, "Video Analysis")


def show_video_analysis_page():
    from Backends.src.ui.components import clean_upload_box
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
    if analyze_clicked and uploaded_video is not None:
        if calibration_mode.startswith("Manual"):
            first_frame = extract_first_video_frame(uploaded_video)
            if first_frame is None:
                st.warning("Could not read the first frame for manual calibration.")
            else:
                with st.expander("Manual Pitch Calibration", expanded=True):
                    manual_pitch_points = show_manual_pitch_point_inputs(first_frame)

        uploaded_video.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
            temp_input.write(uploaded_video.read())
            input_video_path = Path(temp_input.name)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        PROCESSED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        raw_output_path = PROCESSED_VIDEO_DIR / f"raw_cricvision_analysis_{timestamp}.mp4"
        browser_output_path = PROCESSED_VIDEO_DIR / f"cricvision_analysis_{timestamp}.mp4"

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
                )
            result["analysis_mode"] = analysis_mode
            result["active_preset"] = preset_name
            result["active_model"] = selected_model_name
            result["ball_model_used"] = selected_model_name
            result["show_performance_details"] = show_performance

        if not result["success"]:
            st.error(result["error"])
            st.session_state.video_analysis_result = None
        else:
            try:
                if result.get("processed_video_generated", True) and result.get("output_path"):
                    final_video_path = convert_to_browser_mp4(
                        input_path=result["output_path"],
                        output_path=browser_output_path,
                    )
                    result["output_path"] = final_video_path
                else:
                    result["output_path"] = None
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
                    "show_performance_details": show_performance,
                }
                st.success("Analysis complete.")
            except Exception as error:
                st.error(f"Video conversion failed: {error}")
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
