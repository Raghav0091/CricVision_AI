"""Full bowling/delivery video processor."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from Backends.src.analysis.field_zones import (
    find_nearest_fielder,
    generate_wagon_wheel_data,
    normalize_handedness,
    save_field_analysis_history,
    save_field_setup,
    suggest_field_adjustment,
)
from Backends.src.calibration.calibration_context import (
    build_calibration_context,
    normalize_calibration_context,
)
from Backends.src.config.paths import (
    REVIEW_FRAMES_DIR,
    VIDEO_ANALYSIS_OUTPUT_DIR,
)
from Backends.src.models.model_loader import get_cached_yolo_model
from Backends.src.models.model_registry import get_model_info
from Backends.src.tracking.ball_tracking_utils import (
    BallKalmanTracker,
    calculate_tracking_quality,
    detect_bounce_by_direction_change,
    get_tracking_quality_label,
    interpolate_missing_positions,
    smooth_trajectory,
)
from Backends.src.utils.cv2_loader import cv2
from Backends.src.video_pipeline.annotation_writer import (
    add_impact_marker_to_video,
    draw_clean_ball_markers,
    draw_clean_stump_markers,
    draw_label,
    draw_pitch_roi,
    draw_search_roi,
    draw_trajectory_lines,
    ensure_frame_writer_size,
    save_review_frame,
)
from Backends.src.video_pipeline.detection_pipeline import (
    choose_main_ball,
    compute_pitch_homography,
    estimate_auto_pitch_corners,
    estimate_length_from_bounce,
    estimate_length_from_pitch_y,
    estimate_line_from_pitch_x,
    estimate_line_from_stumps,
    get_nearest_stump_detections,
    has_enough_ball_movement,
    load_detection_model,
    load_ensemble_models,
    map_model_classes,
    run_local_redetection,
    run_pitch_roi_detection,
    transform_point_to_pitch,
)
from Backends.src.video_pipeline.performance_timer import (
    create_performance_profile,
    finish_performance_profile,
)
from Backends.src.video_pipeline.report_pipeline import timed_video_reports


MAX_REVIEW_FRAMES_PER_ANALYSIS = 80


def process_delivery_video(
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
    progress_callback=None,
):
    """Process one delivery clip with the established full analysis flow."""
    from Backends.src.analysis.analysis_speed import (
        get_analysis_mode_settings,
        resize_frame_for_inference,
        scale_detections_to_original,
    )
    from Backends.src.analysis.bat_detection import (
        detect_bat_in_frame,
        draw_bat_detections,
    )
    from Backends.src.analysis.impact_detection import (
        detect_bat_ball_impact,
        save_impact_frame_preview,
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

    model = None
    ensemble_models = []
    bat_model = (
        get_cached_yolo_model(bat_model_key) if bat_model_key else None
    )
    bat_unavailable_reason = ""

    if bat_model_key and bat_model is None:
        bat_unavailable_reason = (
            "Impact not detected: bat detection unavailable."
        )
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
                "error": (
                    "No ensemble models were found. Add at least one "
                    "configured model file."
                ),
            }
    else:
        model = load_detection_model(
            model_key=model_key,
            model_path=model_path,
        )

        if model is None:
            model_label = (
                (get_model_info(model_key) or {}).get("name")
                if model_key
                else None
            )
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
    light_annotation = bool(
        speed_settings.get("light_annotation", False)
    )
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

    VIDEO_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    if generate_processed_video:
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
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
    calibration_warning = (
        "Confidence warning: pitch calibration is missing; "
        "using image-space fallback."
    )
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
    estimated_line = "Unknown"
    estimated_length = "Unknown"
    field_setup = field_setup or {}
    batter_handedness = normalize_handedness(
        field_setup.get("batter_handedness", "right")
    )
    bowler_arm = field_setup.get(
        "bowler_arm",
        "Right-arm bowler",
    )
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

        performance["video_read_time_sec"] += (
            time.perf_counter() - read_started
        )
        performance["frames_read"] += 1

        last_raw_frame = frame.copy()
        annotated_frame = frame.copy()
        low_confidence_ball_detections = []
        ball_detections = []
        stump_detections = []
        bat_detections = []

        inference_frame, detection_scale = resize_frame_for_inference(
            frame,
            resize_width,
        )
        run_stump = should_detect_stump(
            frame_index,
            speed_settings,
            locked_stump,
        )
        run_bat = should_detect_bat(
            frame_index,
            speed_settings,
            rough_impact_frame,
        )

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
                locked_stump_detections=(
                    [locked_stump]
                    if locked_stump and not run_stump
                    else None
                ),
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
            low_confidence_ball_detections = (
                scale_detections_to_original(
                    detection_result.get(
                        "low_confidence_ball_detections",
                        [],
                    ),
                    detection_scale,
                    stats=detection_stats,
                )
            )
            ball_elapsed = time.perf_counter() - ball_started
            performance["ball_detection_time_sec"] += ball_elapsed
            performance["model_inference_time_sec"] += ball_elapsed
            if run_stump:
                performance["stump_detection_time_sec"] += (
                    detection_result.get("full_frame_time_ms", 0)
                    / 1000.0
                )
            processed_detection_frames += 1
            full_frame_detection_time_total += detection_result[
                "full_frame_time_ms"
            ]
            roi_detection_time_total += detection_result["roi_time_ms"]

            if detection_result.get("used_roi"):
                previous_roi_box = detection_result["roi_box"]
                roi_detected_frames += 1
                roi_x1, roi_y1, roi_x2, roi_y2 = detection_result[
                    "roi_box"
                ]
                last_roi_size = (
                    f"{roi_x2 - roi_x1}x{roi_y2 - roi_y1}"
                )

            if debug_overlay and show_pitch_roi:
                draw_pitch_roi(
                    annotated_frame,
                    detection_result.get("roi_box"),
                )

            confidence_values.extend(
                item["confidence"] for item in ball_detections
            )

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
                search_center = (
                    previous_ball_center
                    or kalman_tracker.last_prediction
                )
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
                    draw_search_roi(
                        annotated_frame,
                        recovery_result.get("search_roi"),
                    )

                if recovery_result["recovered"]:
                    ball_detections = scale_detections_to_original(
                        recovery_result["ball_detections"],
                        detection_scale,
                        stats=detection_stats,
                    )
                    tracker_recoveries += 1
                    ball_detected_frames += 1
                    total_ball_detections += len(ball_detections)
                    confidence_values.extend(
                        item["confidence"]
                        for item in ball_detections
                    )
                elif (
                    review_frame_count
                    < MAX_REVIEW_FRAMES_PER_ANALYSIS
                ):
                    save_review_frame(
                        frame,
                        timestamp,
                        frame_index,
                        "missed_ball",
                        source="video_analysis",
                        note=(
                            "No ball detection passed the selected "
                            "confidence threshold."
                        ),
                    )
                    review_frame_count += 1
            elif review_frame_count < MAX_REVIEW_FRAMES_PER_ANALYSIS:
                save_review_frame(
                    frame,
                    timestamp,
                    frame_index,
                    "missed_ball",
                    source="video_analysis",
                    note=(
                        "No ball detection passed the selected "
                        "confidence threshold."
                    ),
                )
                review_frame_count += 1

        if run_bat and bat_model:
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

        stump_detections = apply_locked_stump(
            stump_detections,
            locked_stump,
        )

        if stump_detections:
            stump_detected_frames += 1
            total_stump_detections += len(stump_detections)

            if (
                calibration_mode.startswith("Auto")
                and pitch_homography is None
            ):
                auto_pitch_points = estimate_auto_pitch_corners(
                    frame.shape,
                    stump_detections,
                )
                pitch_homography = compute_pitch_homography(
                    auto_pitch_points
                )

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

        initial_frames = speed_settings.get(
            "stump_detect_initial_frames"
        )
        if locked_stump is None and initial_frames:
            if frame_index + 1 >= int(initial_frames):
                locked_stump = lock_static_stump_detection(
                    frame_detections,
                    int(initial_frames),
                )

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
            draw_clean_ball_markers(
                annotated_frame,
                ball_detections,
            )
            draw_clean_stump_markers(
                annotated_frame,
                stump_detections,
            )

        main_ball = choose_main_ball(
            ball_detections,
            previous_ball_center,
        )

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

            trajectory_points = list(
                reversed(display_trajectory_points)
            )[-max_trajectory_points:]
            interpolated_positions = interpolate_missing_positions(
                ball_positions
            )
            usable_trajectory_points = [
                point
                for point in interpolated_positions
                if point is not None
            ]
            bounce_result = None

            if (
                len(usable_trajectory_points)
                >= min_track_points_for_bounce
                and has_enough_ball_movement(
                    usable_trajectory_points,
                    min_movement_distance,
                )
            ):
                bounce_result = detect_bounce_by_direction_change(
                    ball_positions
                )

            if (
                bounce_result is not None
                and estimated_bounce_point is None
            ):
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
                    height,
                )
                pitch_normalized_bounce_point = transform_point_to_pitch(
                    estimated_bounce_point,
                    pitch_homography,
                )

                if pitch_normalized_bounce_point is not None:
                    pitch_x, pitch_y = pitch_normalized_bounce_point
                    estimated_line = estimate_line_from_pitch_x(
                        pitch_x,
                        batter_handedness,
                    )
                    estimated_length = estimate_length_from_pitch_y(
                        pitch_y
                    )
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
                        note=(
                            f"Ball missing for {missing_ball_frames} "
                            "consecutive frames."
                        ),
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

                trajectory_points = list(
                    reversed(display_trajectory_points)
                )[-max_trajectory_points:]

        draw_trajectory_lines(annotated_frame, trajectory_points)

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
                (
                    f"Frame: {frame_index}/"
                    f"{source_total_frames or frame_index}"
                ),
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
                (
                    f"Trajectory: {len(trajectory_points)} | "
                    f"Missing: {missing_ball_frames}"
                ),
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
        _report_progress(
            progress_callback,
            frame_index,
            source_total_frames,
        )

    cap.release()
    if writer is not None:
        writer.release()

    if frame_index == 0:
        return {
            "success": False,
            "error": (
                "No frames were processed. The uploaded video may be "
                "corrupted or unsupported."
            ),
        }

    preliminary_impact = detect_bat_ball_impact(
        frame_detections,
        fps=fps,
    )
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
            prefix=f"video_impact_{Path(output_path).stem}",
        )
        if preview_path is not None:
            impact_info["impact_frame_image_path"] = str(preview_path)
        if (
            not speed_settings.get("skip_impact_video_rewrite")
            and generate_processed_video
        ):
            add_impact_marker_to_video(output_path, impact_info)

    stump_detection_rate = 0
    if frame_index > 0:
        ball_detection_rate = (
            ball_detected_frames / frame_index
        ) * 100
        stump_detection_rate = (
            stump_detected_frames / frame_index
        ) * 100

    average_confidence = 0
    if confidence_values:
        average_confidence = sum(confidence_values) / len(
            confidence_values
        )

    tracking_quality = calculate_tracking_quality(
        ball_positions,
        frame_index,
    )
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
    performance["report_generation_time_sec"] += reports[
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
    wagon_wheel = generate_wagon_wheel_data(
        ball_positions,
        batter_handedness=batter_handedness,
        mode=shot_trajectory_mode,
        manual_contact_frame=manual_contact_frame,
    )
    wagon_wheel["mode"] = shot_trajectory_mode
    nearest_fielder = find_nearest_fielder(
        wagon_wheel.get("shot_angle"),
        fielders,
        batter_handedness,
    )
    wagon_wheel["nearest_fielder"] = nearest_fielder
    wagon_wheel["suggested_adjustment"] = suggest_field_adjustment(
        wagon_wheel,
        nearest_fielder,
    )

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
                "simple_zone": wagon_wheel.get(
                    "simple_zone",
                    "Unknown",
                ),
                "detailed_zone": wagon_wheel.get(
                    "detailed_zone",
                    "Unknown",
                ),
                "shot_angle": (
                    ""
                    if wagon_wheel.get("shot_angle") is None
                    else f"{wagon_wheel['shot_angle']:.2f}"
                ),
                "nearest_fielder": (
                    ""
                    if nearest_fielder is None
                    else nearest_fielder.get("name", "")
                ),
                "confidence": wagon_wheel.get(
                    "confidence",
                    "Low",
                ),
                "corrected_zone": "",
            }
        )
    average_full_frame_detection_time = 0
    average_roi_detection_time = 0

    if frame_index > 0:
        average_full_frame_detection_time = (
            full_frame_detection_time_total / frame_index
        )

    if roi_detected_frames > 0:
        average_roi_detection_time = (
            roi_detection_time_total / roi_detected_frames
        )

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

    if last_raw_frame is not None and (
        estimated_line == "Unknown"
        or estimated_length == "Unknown"
    ):
        if review_frame_count < MAX_REVIEW_FRAMES_PER_ANALYSIS:
            save_review_frame(
                last_raw_frame,
                timestamp,
                max(frame_index - 1, 0),
                "line_length_unknown",
                source="video_analysis",
                note=(
                    f"Line={estimated_line}; "
                    f"Length={estimated_length}."
                ),
            )
            review_frame_count += 1

    return {
        "success": True,
        "output_path": (
            str(output_path) if generate_processed_video else None
        ),
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
        "interpolated_ball_frames": tracking_quality[
            "interpolated_frames"
        ],
        "kalman_predicted_frames": kalman_predicted_frames,
        "tracker_recoveries": tracker_recoveries,
        "overall_tracking_quality": overall_tracking_quality,
        "stump_detection_rate": stump_detection_rate,
        "average_ball_confidence": average_confidence,
        "full_frame_detection_time_ms": (
            average_full_frame_detection_time
        ),
        "roi_detection_time_ms": average_roi_detection_time,
        "roi_detected_frames": roi_detected_frames,
        "last_roi_size": last_roi_size,
        "estimated_bounce_point": estimated_bounce_point,
        "estimated_bounce_frame": estimated_bounce_frame,
        "pitch_normalized_bounce_point": (
            pitch_normalized_bounce_point
        ),
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
        "ball_model_used": "Current Best Ball + Stump Model",
        "bat_model_used": (
            (get_model_info(bat_model_key) or {}).get(
                "name",
                bat_model_key,
            )
            if bat_model_key and bat_model is not None
            else ("Unavailable" if bat_model_key else "Not used")
        ),
        **enrichment,
    }


def _report_progress(callback, frame_index, total_frames):
    if callable(callback):
        callback(frame_index, total_frames)
