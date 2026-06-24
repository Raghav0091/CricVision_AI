from datetime import datetime
from pathlib import Path
from threading import Lock
from tempfile import TemporaryDirectory

import streamlit as st

from Backends.src.utils.cv2_loader import cv2
from Backends.src.analysis.cricket_agent import (
    calculate_detection_quality,
    detect_analysis_warnings,
    generate_coaching_feedback as generate_agent_coaching_feedback,
    generate_delivery_report,
)
from Backends.src.analysis.field_zones import (
    generate_wagon_wheel_data,
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
from Backends.src.models.model_registry import get_model_info, get_model_path


CRICKET_OBJECTS_MODEL_PATH = Path("Models/cricket_objects/best.pt")
EXTERNAL_BALL_MODEL_PATH = Path("Models/cricket_objects/best_external.pt")
BALL_MODEL_PATH = Path("Models/ball_detector/best.pt")
REVIEW_FRAMES_DIR = Path("outputs/review_frames")
ENSEMBLE_MODEL_NAME = "Ensemble: All Ball Models + Stumps"

LOW_CONFIDENCE_REVIEW_THRESHOLD = 0.35
MIN_TRAJECTORY_POINTS_FOR_BOUNCE = 8
MIN_MOVEMENT_DISTANCE = 40
SHORT_MISSING_BALL_SMOOTHING_FRAMES = 8
MAX_MISSING_BALL_FRAMES = 12
MAX_TRAJECTORY_POINTS = 35
MAX_RECORDED_FRAMES = 450
DEFAULT_RECORDING_FPS = 25
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


def get_live_model_options():
    return {
        "Ball + Stump Detector": {
            "path": CRICKET_OBJECTS_MODEL_PATH,
            "model_key": "current_best",
            "ensemble": False,
        },
        "Old Ball Detector": {
            "path": BALL_MODEL_PATH,
            "model_key": None,
            "ensemble": False,
        },
        "External Ball Model": {
            "path": EXTERNAL_BALL_MODEL_PATH,
            "model_key": None,
            "ensemble": False,
        },
        ENSEMBLE_MODEL_NAME: {
            "path": None,
            "model_key": None,
            "ensemble": True,
        },
    }


class LiveDeliveryRecordingState:
    def __init__(self):
        self.frames = []
        self.recording = False
        self.lock = Lock()
        self.max_frames = MAX_RECORDED_FRAMES

    def start_recording(self):
        with self.lock:
            self.frames = []
            self.recording = True

    def stop_recording(self):
        with self.lock:
            self.recording = False
            return [frame.copy() for frame in self.frames]

    def clear(self):
        with self.lock:
            self.frames = []
            self.recording = False

    def append_frame(self, frame):
        with self.lock:
            if self.recording and len(self.frames) < self.max_frames:
                self.frames.append(frame.copy())

    def get_frame_count(self):
        with self.lock:
            return len(self.frames)


def create_delivery_recorder_class(recording_state):
    import av
    from streamlit_webrtc import VideoProcessorBase

    class DeliveryRecorder(VideoProcessorBase):
        def __init__(self):
            self.recording_state = recording_state

        def recv(self, frame):
            image = frame.to_ndarray(format="bgr24")
            self.recording_state.append_frame(image)

            return av.VideoFrame.from_ndarray(image, format="bgr24")

    return DeliveryRecorder


def write_video(frames, output_path, fps=DEFAULT_RECORDING_FPS):
    if not frames:
        return False

    height, width = frames[0].shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        return False

    for frame in frames:
        writer.write(frame)

    writer.release()
    return True


def collect_detections(result, class_names, get_box_center, ball_confidence):
    ball_detections = []
    low_confidence_ball_detections = []
    stump_detections = []

    if result.boxes is None or len(result.boxes) == 0:
        return ball_detections, low_confidence_ball_detections, stump_detections

    for box in result.boxes:
        class_id = int(box.cls[0].cpu().numpy())
        confidence = float(box.conf[0].cpu().numpy())
        class_name = class_names.get(class_id)

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        if class_name is None:
            continue

        detection = {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "box": (x1, y1, x2, y2),
            "center": get_box_center(x1, y1, x2, y2),
        }

        if class_name == "ball":
            if confidence >= ball_confidence:
                ball_detections.append(detection)
            if 0.10 <= confidence < LOW_CONFIDENCE_REVIEW_THRESHOLD:
                low_confidence_ball_detections.append(detection)
        elif class_name == "stump":
            stump_detections.append(detection)

    return ball_detections, low_confidence_ball_detections, stump_detections


def get_point_distance(point_a, point_b):
    ax, ay = point_a
    bx, by = point_b
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def predict_next_ball_center(trajectory_points, previous_ball_center):
    if len(trajectory_points) >= 2:
        previous_x, previous_y = trajectory_points[-2]
        last_x, last_y = trajectory_points[-1]
        return last_x + (last_x - previous_x), last_y + (last_y - previous_y)

    return previous_ball_center


def choose_continuous_ball(ball_detections, trajectory_points, previous_ball_center, frame_shape):
    if not ball_detections:
        return None

    if previous_ball_center is None:
        return max(ball_detections, key=lambda item: item["confidence"])

    height, width = frame_shape[:2]
    frame_diagonal = (width**2 + height**2) ** 0.5
    max_reasonable_jump = max(45, frame_diagonal * 0.18)
    predicted_center = predict_next_ball_center(trajectory_points, previous_ball_center)

    def continuity_score(item):
        center = item["center"]
        distance_from_previous = get_point_distance(center, previous_ball_center)
        distance_from_prediction = get_point_distance(center, predicted_center)
        return distance_from_prediction + (distance_from_previous * 0.35) - (item["confidence"] * 80)

    nearest_detection = min(ball_detections, key=continuity_score)
    nearest_distance = get_point_distance(nearest_detection["center"], previous_ball_center)

    if len(trajectory_points) >= 3 and nearest_distance > max_reasonable_jump:
        return None

    return nearest_detection


def save_low_confidence_review_frame(frame, detections, timestamp, frame_index):
    if not detections:
        return

    REVIEW_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    review_frame = frame.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        center_x, center_y = detection["center"]
        confidence = detection["confidence"]
        cv2.rectangle(review_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.circle(review_frame, (center_x, center_y), 4, (0, 165, 255), -1)
        label_y = max(y1, 25)
        cv2.rectangle(
            review_frame,
            (x1, label_y - 24),
            (x1 + 140, label_y),
            (0, 120, 220),
            -1,
        )
        cv2.putText(
            review_frame,
            f"low ball {confidence:.2f}",
            (x1 + 5, label_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

    cv2.imwrite(str(REVIEW_FRAMES_DIR / f"low_conf_ball_{timestamp}_{frame_index:04d}.jpg"), review_frame)


def stop_webrtc_context_if_possible(webrtc_context):
    if webrtc_context is None:
        return

    try:
        stop_method = getattr(webrtc_context, "stop", None)
        if callable(stop_method):
            stop_method()

        state = getattr(webrtc_context, "state", None)
        if state is not None and hasattr(state, "playing"):
            state.playing = False
    except Exception:
        pass


def draw_delivery_dashboard(
    frame,
    frame_index,
    total_frames,
    ball_count,
    stump_count,
    trajectory_points,
    missing_ball_frames,
    estimated_bounce_frame,
    estimated_line,
    estimated_length,
):
    panel_x = 15
    panel_y = 15
    panel_w = 470
    panel_h = 240

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    bounce_text = "Not found"
    if estimated_bounce_frame is not None:
        bounce_text = f"Frame {estimated_bounce_frame}"

    dashboard_lines = [
        (f"Frame: {frame_index}/{total_frames}", (255, 255, 255)),
        (f"Balls in frame: {ball_count}", (0, 255, 255)),
        (f"Stumps in frame: {stump_count}", (255, 160, 0)),
        (
            f"Trajectory: {len(trajectory_points)} | Missing: {missing_ball_frames}",
            (0, 255, 255),
        ),
        (f"Bounce: {bounce_text}", (0, 0, 255)),
        (f"Line: {estimated_line}", (255, 255, 0)),
        (f"Length: {estimated_length}", (0, 255, 0)),
    ]

    for index, (text, color) in enumerate(dashboard_lines):
        cv2.putText(
            frame,
            text,
            (panel_x + 15, panel_y + 30 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )


def process_recorded_delivery(
    frames,
    confidence,
    image_size,
    model_path=CRICKET_OBJECTS_MODEL_PATH,
    model_key=None,
    use_ensemble=False,
    show_pitch_roi=False,
    field_setup=None,
    fps=DEFAULT_RECORDING_FPS,
):
    from Backends.src.analysis.bat_detection import detect_bat_in_frame, draw_bat_detections
    from Backends.src.analysis.impact_detection import (
        detect_bat_ball_impact,
        save_impact_frame_preview,
    )
    from Backends.src.analysis.shot_classification import classify_shot_type
    from Backends.src.ui.video_analysis import (
        convert_to_browser_mp4,
        draw_label,
        draw_pitch_roi,
        draw_search_roi,
        estimate_length_from_bounce,
        estimate_line_from_stumps,
        get_nearest_stump_detections,
        has_enough_ball_movement,
        load_ensemble_models,
        load_detection_model,
        map_model_classes,
        run_local_redetection,
        run_pitch_roi_detection,
        save_review_frame,
    )

    if not frames:
        return {
            "success": False,
            "error": "No frames were recorded. Click Start Delivery Recording before bowling.",
        }

    model = None
    ensemble_models = []
    stump_model = get_cached_yolo_model("current_best")
    bat_model = get_cached_yolo_model("cricshot_bat")
    bat_unavailable_reason = ""
    if bat_model is None:
        bat_unavailable_reason = "Impact not detected: bat detection unavailable."

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

    height, width = frames[0].shape[:2]

    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_temp_path = temp_dir_path / f"raw_delivery_{timestamp}.mp4"
        processed_raw_path = temp_dir_path / f"processed_delivery_{timestamp}_raw.mp4"
        processed_browser_path = temp_dir_path / f"processed_delivery_{timestamp}.mp4"

        if not write_video(frames, raw_temp_path, fps=fps):
            return {
                "success": False,
                "error": "Could not prepare raw delivery clip for download.",
            }

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(processed_raw_path), fourcc, fps, (width, height))

        if not writer.isOpened():
            return {
                "success": False,
                "error": "Could not create processed delivery video.",
            }

        total_frames = len(frames)
        ball_detected_frames = 0
        bat_detected_frames = 0
        stump_detected_frames = 0
        total_ball_detections = 0
        low_confidence_ball_frames = 0
        total_stump_detections = 0
        confidence_values = []
        review_frame_count = 0
        full_frame_detection_time_total = 0
        roi_detection_time_total = 0
        roi_detected_frames = 0
        tracker_recoveries = 0
        kalman_predicted_frames = 0
        last_roi_size = "Full frame"

        trajectory_points = []
        ball_positions = []
        impact_frame_detections = []
        impact_frame_candidates = {}
        stump_detections_by_frame = []
        previous_roi_box = None
        previous_ball_center = None
        kalman_tracker = BallKalmanTracker(max_missing_frames=10)
        missing_ball_frames = 0
        last_stump_detections = []

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

        progress_bar = st.progress(0)
        status_text = st.empty()

        for frame_index, frame in enumerate(frames):
            annotated_frame = frame.copy()
            low_confidence_ball_detections = []
            recovered_this_frame = False
            bat_detections = detect_bat_in_frame(frame, bat_model, confidence) if bat_model else []

            if bat_detections:
                bat_detected_frames += 1
            draw_bat_detections(annotated_frame, bat_detections)

            detection_result = run_pitch_roi_detection(
                frame,
                stump_model=stump_model,
                stump_class_names=stump_class_names,
                confidence=confidence,
                imgsz=image_size,
                previous_roi=previous_roi_box,
                ball_model=model,
                ball_class_names=class_names if not use_ensemble else None,
                ensemble_models=ensemble_models,
                use_ensemble=use_ensemble,
                ball_confidence=confidence,
            )
            ball_detections = detection_result["ball_detections"]
            stump_detections = detection_result["stump_detections"]
            low_confidence_ball_detections = detection_result.get(
                "low_confidence_ball_detections",
                [],
            )
            full_frame_detection_time_total += detection_result["full_frame_time_ms"]
            roi_detection_time_total += detection_result["roi_time_ms"]

            if detection_result.get("used_roi"):
                previous_roi_box = detection_result["roi_box"]
                roi_detected_frames += 1
                roi_x1, roi_y1, roi_x2, roi_y2 = detection_result["roi_box"]
                last_roi_size = f"{roi_x2 - roi_x1}x{roi_y2 - roi_y1}"

            if show_pitch_roi:
                draw_pitch_roi(annotated_frame, detection_result.get("roi_box"))

            if low_confidence_ball_detections:
                low_confidence_ball_frames += 1

                if review_frame_count < 80:
                    save_review_frame(
                        frame,
                        timestamp,
                        frame_index,
                        "low_confidence",
                        low_confidence_ball_detections,
                        source="live_session",
                    )
                    review_frame_count += 1

            if not ball_detections:
                search_center = previous_ball_center or kalman_tracker.last_prediction
                recovery_result = run_local_redetection(
                    frame,
                    search_center,
                    confidence,
                    image_size,
                    missing_ball_frames + 1,
                    ball_model=model,
                    ball_class_names=class_names if not use_ensemble else None,
                    ensemble_models=ensemble_models,
                    use_ensemble=use_ensemble,
                )

                if show_pitch_roi:
                    draw_search_roi(annotated_frame, recovery_result.get("search_roi"))

                if recovery_result["recovered"]:
                    ball_detections = recovery_result["ball_detections"]
                    recovered_this_frame = True
                    tracker_recoveries += 1
                    ball_detected_frames += 1
                    total_ball_detections += len(ball_detections)
                    confidence_values.extend(item["confidence"] for item in ball_detections)
                elif review_frame_count < 80:
                    save_review_frame(
                        frame,
                        timestamp,
                        frame_index,
                        "missed_ball",
                        source="live_session",
                        note="No ball detection passed the selected confidence threshold.",
                    )
                    review_frame_count += 1

            if ball_detections and not recovered_this_frame:
                ball_detected_frames += 1
                total_ball_detections += len(ball_detections)
                confidence_values.extend(item["confidence"] for item in ball_detections)

            if stump_detections:
                stump_detected_frames += 1
                total_stump_detections += len(stump_detections)
                last_stump_detections = stump_detections

            stump_detections_by_frame.append(stump_detections)
            impact_frame_detections.append(
                {
                    "frame_index": frame_index,
                    "ball_detections": ball_detections,
                    "bat_detections": bat_detections,
                }
            )
            if ball_detections and bat_detections:
                impact_frame_candidates[frame_index] = frame.copy()

            for detection in ball_detections:
                x1, y1, x2, y2 = detection["box"]
                center_x, center_y = detection["center"]
                detection_confidence = detection["confidence"]

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 255), -1)
                draw_label(
                    annotated_frame,
                    f"ball {detection_confidence:.2f}",
                    x1,
                    y1,
                    (0, 180, 180),
                )

            for detection in stump_detections:
                x1, y1, x2, y2 = detection["box"]
                detection_confidence = detection["confidence"]

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                draw_label(
                    annotated_frame,
                    f"stump {detection_confidence:.2f}",
                    x1,
                    y1,
                    (255, 100, 0),
                )

            main_ball = choose_continuous_ball(
                ball_detections,
                trajectory_points,
                previous_ball_center,
                frame.shape,
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

                trajectory_points = list(reversed(display_trajectory_points))[-MAX_TRAJECTORY_POINTS:]
                interpolated_positions = interpolate_missing_positions(ball_positions)
                usable_trajectory_points = [
                    point for point in interpolated_positions if point is not None
                ]
                bounce_result = None

                if (
                    estimated_bounce_point is None
                    and len(usable_trajectory_points) >= MIN_TRAJECTORY_POINTS_FOR_BOUNCE
                    and has_enough_ball_movement(usable_trajectory_points, MIN_MOVEMENT_DISTANCE)
                ):
                    bounce_result = detect_bounce_by_direction_change(ball_positions)

                    if bounce_result is not None:
                        estimated_bounce_point = bounce_result["point"]
                        estimated_bounce_frame = bounce_result["frame_index"]
                        bounce_stump_detections = get_nearest_stump_detections(
                            stump_detections_by_frame,
                            estimated_bounce_frame,
                        )
                        estimated_line = estimate_line_from_stumps(
                            estimated_bounce_point,
                            bounce_stump_detections or last_stump_detections,
                            batter_handedness,
                        )
                        estimated_length = estimate_length_from_bounce(
                            estimated_bounce_point,
                            height,
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

                if missing_ball_frames >= MAX_MISSING_BALL_FRAMES:
                    kalman_tracker.reset()
                    if review_frame_count < 80:
                        save_review_frame(
                            frame,
                            timestamp,
                            frame_index,
                            "poor_tracking",
                            source="live_session",
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

                    trajectory_points = list(reversed(display_trajectory_points))[-MAX_TRAJECTORY_POINTS:]

            for point_index in range(1, len(trajectory_points)):
                cv2.line(
                    annotated_frame,
                    trajectory_points[point_index - 1],
                    trajectory_points[point_index],
                    (0, 255, 255),
                    3,
                )

            if estimated_bounce_point is not None:
                bx, by = estimated_bounce_point

                cv2.circle(annotated_frame, (bx, by), 10, (0, 0, 255), -1)
                cv2.circle(annotated_frame, (bx, by), 16, (255, 255, 255), 2)
                cv2.putText(
                    annotated_frame,
                    f"Bounce Frame: {estimated_bounce_frame}",
                    (bx + 15, max(by - 15, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            draw_delivery_dashboard(
                annotated_frame,
                frame_index=frame_index,
                total_frames=total_frames,
                ball_count=len(ball_detections),
                stump_count=len(stump_detections),
                trajectory_points=trajectory_points,
                missing_ball_frames=missing_ball_frames,
                estimated_bounce_frame=estimated_bounce_frame,
                estimated_line=estimated_line,
                estimated_length=estimated_length,
            )

            writer.write(annotated_frame)
            progress_bar.progress(min((frame_index + 1) / total_frames, 1.0))
            status_text.text(f"Analyzing delivery frame {frame_index + 1}/{total_frames}")

        writer.release()
        progress_bar.empty()
        status_text.empty()

        try:
            convert_to_browser_mp4(processed_raw_path, processed_browser_path)
        except Exception as error:
            return {
                "success": False,
                "error": f"Video conversion failed: {error}",
            }

        raw_video_bytes = raw_temp_path.read_bytes()
        processed_video_bytes = processed_browser_path.read_bytes()

    ball_detection_rate = (ball_detected_frames / total_frames) * 100
    stump_detection_rate = (stump_detected_frames / total_frames) * 100
    average_ball_confidence = 0

    if confidence_values:
        average_ball_confidence = sum(confidence_values) / len(confidence_values)

    tracking_quality = calculate_tracking_quality(ball_positions, total_frames)
    overall_tracking_quality = get_tracking_quality_label(
        tracking_quality["tracking_rate"],
        tracking_quality["interpolated_frames"],
        kalman_predicted_frames,
    )
    average_full_frame_detection_time = 0
    average_roi_detection_time = 0

    if total_frames > 0:
        average_full_frame_detection_time = full_frame_detection_time_total / total_frames

    if roi_detected_frames > 0:
        average_roi_detection_time = roi_detection_time_total / roi_detected_frames

    if frames and estimated_bounce_point is None and review_frame_count < 80:
        save_review_frame(
            frames[-1],
            timestamp,
            total_frames - 1,
            "bounce_unknown",
            source="live_session",
            note="Analysis finished without a bounce estimate.",
        )
        review_frame_count += 1

    if frames and (estimated_line == "Unknown" or estimated_length == "Unknown") and review_frame_count < 80:
        save_review_frame(
            frames[-1],
            timestamp,
            total_frames - 1,
            "line_length_unknown",
            source="live_session",
            note=f"Line={estimated_line}; Length={estimated_length}.",
        )
        review_frame_count += 1

    impact_info = detect_bat_ball_impact(impact_frame_detections, fps=fps)
    if bat_unavailable_reason:
        impact_info["reason"] = bat_unavailable_reason
        impact_info["impact_reason"] = bat_unavailable_reason
    impact_frame = impact_info.get("impact_frame")
    if impact_frame is not None:
        preview_path = save_impact_frame_preview(
            impact_frame_candidates.get(impact_frame),
            impact_info,
            prefix=f"live_impact_{timestamp}",
        )
        if preview_path is not None:
            impact_info["impact_frame_image_path"] = str(preview_path)
    shot_info = classify_shot_type(
        impact_frame_detections,
        impact_info,
        batter_handedness=batter_handedness,
        fps=fps,
    )
    from Backends.src.analysis.delivery_enrichment import run_post_shot_pipeline

    delivery_report = {
        "estimated_line": estimated_line,
        "estimated_length": estimated_length,
        "ball_detection_rate": ball_detection_rate,
        "overall_tracking_quality": overall_tracking_quality,
    }
    direction_info, outcome_info, agent_info, enrichment = run_post_shot_pipeline(
        impact_frame_detections,
        impact_info,
        shot_info,
        batter_handedness,
        fps,
        delivery_report=delivery_report,
    )

    wagon_wheel = generate_wagon_wheel_data(
        ball_positions,
        batter_handedness=batter_handedness,
        mode="Use last part of trajectory",
    )
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
                "source": "live_session",
                "batter_handedness": batter_handedness,
                "bowler_arm": bowler_arm,
                "camera_view": camera_view,
                "preset": field_setup.get("preset", "Custom"),
                "simple_zone": wagon_wheel.get("simple_zone", "Unknown"),
                "detailed_zone": wagon_wheel.get("detailed_zone", "Unknown"),
                "shot_angle": "" if wagon_wheel.get("shot_angle") is None else f"{wagon_wheel['shot_angle']:.2f}",
                "nearest_fielder": "" if nearest_fielder is None else nearest_fielder.get("name", ""),
                "confidence": wagon_wheel.get("confidence", "Low"),
                "corrected_zone": "",
            }
        )

    return {
        "success": True,
        "raw_video_bytes": raw_video_bytes,
        "processed_video_bytes": processed_video_bytes,
        "raw_file_name": f"raw_delivery_{timestamp}.mp4",
        "processed_file_name": f"processed_delivery_{timestamp}.mp4",
        "total_frames": total_frames,
        "ball_detected_frames": ball_detected_frames,
        "bat_detected_frames": bat_detected_frames,
        "stump_detected_frames": stump_detected_frames,
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
        "average_ball_confidence": average_ball_confidence,
        "full_frame_detection_time_ms": average_full_frame_detection_time,
        "roi_detection_time_ms": average_roi_detection_time,
        "roi_detected_frames": roi_detected_frames,
        "last_roi_size": last_roi_size,
        "estimated_bounce_point": estimated_bounce_point,
        "estimated_bounce_frame": estimated_bounce_frame,
        "estimated_line": estimated_line,
        "estimated_length": estimated_length,
        "wagon_wheel": wagon_wheel,
        "field_setup": field_setup,
        "batter_handedness": batter_handedness,
        "bowler_arm": bowler_arm,
        "camera_view": camera_view,
        "ball_detection_difficult": ball_detection_rate < 35 or low_confidence_ball_frames > 0,
        "review_frame_count": review_frame_count,
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
        "bat_model_used": "CricShot10k Bat Detector" if bat_model else "Unavailable",
        **enrichment,
    }


def get_quality_label(rate):
    if rate >= 70:
        return "Good"
    if rate >= 35:
        return "Fair"
    if rate > 0:
        return "Low"
    return "Not detected"


def get_coaching_feedback(result):
    feedback = []

    if result.get("ball_detection_difficult"):
        feedback.append(
            "Ball was difficult to detect. Try 60 FPS, better lighting, closer camera, and stable recording."
        )

    if result["ball_detection_rate"] < 35:
        feedback.append(
            "Ball tracking was limited. Use stronger lighting, a stable landscape camera, and keep the full pitch in frame."
        )
    else:
        feedback.append("Ball tracking quality is usable for a first delivery review.")

    if result["stump_detection_rate"] < 35:
        feedback.append(
            "Stumps were not consistently visible. Keep the camera behind the bowler and pointed toward the batter's stumps."
        )

    if result["estimated_bounce_point"] is None:
        feedback.append(
            "Bounce point was not found. Record a slightly longer clip with the ball visible before and after pitching."
        )
    elif result["estimated_length"] == "Yorker":
        feedback.append("Excellent attacking length if intentional. Watch that it does not drift into a full toss.")
    elif result["estimated_length"] == "Full":
        feedback.append("Full length detected. Useful for swing, but keep checking line control.")
    elif result["estimated_length"] == "Good Length":
        feedback.append("Good length detected. This is a strong default area for testing the batter.")
    elif result["estimated_length"] == "Short":
        feedback.append("Short length detected. Use it as a variation and make sure it rises enough to challenge the batter.")

    if result["estimated_line"] != "Unknown":
        feedback.append(f"Line estimate: {result['estimated_line']}. Use this with your intended target line.")

    return feedback


def show_delivery_report(result):
    st.subheader("Delivery Report")

    ball_quality = get_quality_label(result["ball_detection_rate"])
    stump_quality = get_quality_label(result["stump_detection_rate"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Frames", result["total_frames"])
    col2.metric("Ball Quality", ball_quality, f"{result['ball_detection_rate']:.1f}%")
    col3.metric("Stump Quality", stump_quality, f"{result['stump_detection_rate']:.1f}%")
    col4.metric("Avg Ball Confidence", f"{result['average_ball_confidence']:.2f}")

    if result.get("ball_detection_difficult"):
        st.warning(
            "Ball was difficult to detect. Try 60 FPS, better lighting, closer camera, and stable recording."
        )

    st.subheader("Bounce / Pitch Estimate")

    if result["estimated_bounce_point"] is not None:
        bx, by = result["estimated_bounce_point"]
        col5, col6, col7, col8, col9 = st.columns(5)
        col5.metric("Bounce Frame", result["estimated_bounce_frame"])
        col6.metric("Bounce X", bx)
        col7.metric("Bounce Y", by)
        col8.metric("Line", result["estimated_line"])
        col9.metric("Length", result["estimated_length"])
    else:
        st.warning("Bounce/pitch point was not found for this delivery.")

    st.subheader("Coaching Feedback")
    for feedback_item in get_coaching_feedback(result):
        st.write(f"- {feedback_item}")

    st.caption("Clips are prepared in memory. They are not permanently saved unless you download them.")


def ensure_delivery_report_fields(result):
    result.setdefault("ball_tracking_rate", result.get("ball_detection_rate", 0))
    result.setdefault("interpolated_ball_frames", 0)
    result.setdefault("estimated_line", "Unknown")
    result.setdefault("estimated_length", "Unknown")
    result.setdefault("estimated_bounce_point", None)
    result.setdefault("average_ball_confidence", 0)
    result.setdefault("kalman_predicted_frames", 0)
    result.setdefault("tracker_recoveries", 0)
    result.setdefault("overall_tracking_quality", "Poor")


def show_cricket_delivery_report(result):
    ensure_delivery_report_fields(result)

    quality = calculate_detection_quality(result)
    report = generate_delivery_report(result)
    feedback_items = generate_agent_coaching_feedback(result)
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


def show_analysis_output(result):
    from Backends.src.ui.components import (
        delivery_summary_card,
        render_delivery_report,
        render_impact_frame_preview,
        render_impact_report,
        render_outcome_prediction,
        render_save_status,
        render_shot_direction_report,
        render_shot_report,
        render_vision_agent_report,
        video_preview_card,
    )

    if result is None:
        delivery_summary_card(None)
        return

    if not result.get("success"):
        st.error(result.get("error", "Live delivery analysis failed."))
        return

    video_preview_card("Processed Video Preview")
    processed_video_bytes = result.get("processed_video_bytes")
    if processed_video_bytes:
        st.video(processed_video_bytes)
        st.download_button(
            label="Download Processed Delivery Clip",
            data=processed_video_bytes,
            file_name=result.get("processed_file_name", "processed_delivery.mp4"),
            mime="video/mp4",
            use_container_width=True,
        )
    else:
        st.info("Processed video preview is not available for this live result.")

    render_delivery_report(result)
    render_impact_report(result)
    render_impact_frame_preview(result)
    render_shot_report(result)
    if result.get("direction_agent_available", bool(result.get("direction_info") or result.get("field_zone"))):
        render_shot_direction_report(result)
        render_outcome_prediction(result)
        render_vision_agent_report(result)
    else:
        st.info("Shot direction and agent review require frame-level detections.")
        render_outcome_prediction(result)
    render_save_status(result, "Live Session")


def reset_live_delivery_state():
    st.session_state.live_delivery_recording = False
    st.session_state.live_recorded_frames = []
    st.session_state.live_last_result = None
    st.session_state.live_recording_state = LiveDeliveryRecordingState()
    st.session_state.live_camera_active = True
    st.session_state.live_camera_session_ended = False
    st.session_state.live_pending_analysis = False
    st.session_state.live_pending_analysis_settings = None
    st.session_state.live_webrtc_key_suffix += 1
    st.session_state.live_status_message = "Ready for a new delivery."


def initialize_live_session_state():
    defaults = {
        "live_delivery_recording": False,
        "live_recorded_frames": [],
        "live_last_result": None,
        "live_recording_state": LiveDeliveryRecordingState(),
        "live_camera_active": True,
        "live_camera_session_ended": False,
        "live_pending_analysis": False,
        "live_pending_analysis_settings": None,
        "live_webrtc_key_suffix": 0,
        "live_status_message": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def show_live_session_page():
    from Backends.src.ui.interactive_field_map import render_field_setup_card
    from Backends.src.ui.theme import render_page_header, render_status_pill

    initialize_live_session_state()
    recording_state = st.session_state.live_recording_state

    if recording_state.recording:
        status = "Recording"
    elif st.session_state.live_pending_analysis:
        status = "Analyzing"
    elif st.session_state.live_camera_session_ended and st.session_state.live_last_result:
        status = "Review Ready"
    else:
        status = "Camera Ready"

    render_page_header(
        "Live Bowling Session",
        "Record one clean delivery. Overlays appear only after analysis.",
        badge=status,
    )

    model_options = get_live_model_options()
    analysis_complete = (
        st.session_state.live_camera_session_ended
        and st.session_state.live_last_result is not None
    )

    if st.session_state.live_camera_session_ended and st.session_state.live_pending_analysis:
        st.info(
            "Delivery captured. Camera session ended. Refresh or click Start New Delivery to record again."
        )

        recorded_frames = st.session_state.live_recorded_frames
        analysis_settings = st.session_state.live_pending_analysis_settings or {}

        if not recorded_frames:
            st.session_state.live_last_result = {
                "success": False,
                "error": "No frames were recorded. Start recording, wait for the camera preview, then bowl.",
            }
        else:
            with st.spinner("Analyzing recorded delivery..."):
                st.session_state.live_last_result = process_recorded_delivery(
                    frames=recorded_frames,
                    confidence=analysis_settings.get("confidence", 0.25),
                    image_size=analysis_settings.get("image_size", 960),
                    model_path=Path(
                        analysis_settings.get(
                            "model_path",
                            str(CRICKET_OBJECTS_MODEL_PATH),
                        )
                    ),
                    model_key=analysis_settings.get("model_key"),
                    use_ensemble=analysis_settings.get("use_ensemble", False),
                    show_pitch_roi=analysis_settings.get("show_pitch_roi", False),
                    field_setup=analysis_settings.get("field_setup"),
                )
                st.session_state.live_last_result["active_model"] = analysis_settings.get(
                    "model_name",
                    "Unknown",
                )
                st.session_state.live_last_result["active_preset"] = analysis_settings.get(
                    "preset_name",
                    "Unknown",
                )

        st.session_state.live_pending_analysis = False
        st.session_state.live_pending_analysis_settings = None
        st.rerun()

    if analysis_complete:
        st.info(
            "Delivery captured. Camera session ended. Refresh or click Start New Delivery to record again."
        )
        if st.button("Start New Delivery", type="primary"):
            reset_live_delivery_state()
            st.rerun()

        show_analysis_output(st.session_state.live_last_result)
        return

    st.markdown(
        f'<div style="margin-bottom:1rem;">{render_status_pill("Live preview stays clean until analysis", "gold")}</div>',
        unsafe_allow_html=True,
    )

    field_setup = render_field_setup_card(key_prefix="live_session_field", compact=True, default_preset="Balanced")

    with st.expander("Advanced Settings", expanded=False):
        st.selectbox(
            "Detection model",
            list(model_options.keys()),
            key="live_session_model",
        )
        selected_model_name = st.session_state["live_session_model"]
        selected_model = model_options[selected_model_name]
        selected_model_path = selected_model["path"]
        selected_model_key = selected_model.get("model_key")
        status_path = get_model_path(selected_model_key) if selected_model_key else selected_model_path
        if not selected_model.get("ensemble", False) and status_path is not None and not status_path.exists():
            st.warning(f"Model not found: {status_path}")

        st.selectbox(
            "Detection preset",
            list(DETECTION_PRESETS.keys()),
            index=1,
            key="live_session_preset",
        )
        st.checkbox("Show pitch ROI overlay", value=False, key="live_session_show_roi")

    selected_model_name = st.session_state.get(
        "live_session_model",
        list(model_options.keys())[0],
    )
    selected_model = model_options[selected_model_name]
    selected_model_path = selected_model["path"]
    selected_model_key = selected_model.get("model_key")
    use_ensemble = selected_model.get("ensemble", False)
    preset_name = st.session_state.get("live_session_preset", "Balanced Mode")
    active_preset = DETECTION_PRESETS[preset_name]
    confidence = active_preset["confidence"]
    image_size = active_preset["imgsz"]
    show_pitch_roi = st.session_state.get("live_session_show_roi", False)

    if st.session_state.live_status_message:
        st.info(st.session_state.live_status_message)
        st.session_state.live_status_message = None

    try:
        from streamlit_webrtc import RTCConfiguration, webrtc_streamer
    except ImportError:
        st.error("streamlit-webrtc is not installed. Add streamlit-webrtc to requirements.txt.")
        return

    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    webrtc_context = None

    if st.session_state.live_camera_active:
        webrtc_context = webrtc_streamer(
            key=f"cricvision-live-delivery-recorder-{st.session_state.live_webrtc_key_suffix}",
            video_processor_factory=create_delivery_recorder_class(
                st.session_state.live_recording_state
            ),
            rtc_configuration=rtc_config,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280},
                    "height": {"ideal": 720},
                    "facingMode": "environment",
                },
                "audio": False,
            },
            async_processing=True,
        )

    recorder = None
    if webrtc_context is not None:
        recorder = webrtc_context.video_processor

    if not recording_state.recording:
        start_clicked = st.button(
            "Start Delivery Recording",
            type="primary",
            use_container_width=True,
            disabled=recorder is None,
        )
    else:
        start_clicked = False
        done_clicked = st.button(
            "Analyze Delivery",
            type="primary",
            use_container_width=True,
        )
        clear_clicked = st.button("Clear Delivery", use_container_width=True)

    if not recording_state.recording:
        done_clicked = False
        clear_clicked = False
        if recorder is None:
            st.caption("Allow camera access to begin your live session.")
        else:
            st.caption("Preview stays clean. Start recording, bowl one delivery, then analyze.")
    else:
        st.info(f"Recording... {recording_state.get_frame_count()} frames captured.")

    if start_clicked:
        recording_state.start_recording()
        st.session_state.live_delivery_recording = True
        st.session_state.live_recorded_frames = []
        st.session_state.live_last_result = None
        st.session_state.live_status_message = (
            "Recording started. Bowl one delivery, then click Done / Analyze Delivery."
        )
        st.rerun()

    if done_clicked:
        recorded_frames = recording_state.stop_recording()
        st.session_state.live_delivery_recording = False
        st.session_state.live_recorded_frames = recorded_frames
        st.session_state.live_camera_active = False
        st.session_state.live_camera_session_ended = True
        stop_webrtc_context_if_possible(webrtc_context)
        st.session_state.live_pending_analysis = True
        st.session_state.live_pending_analysis_settings = {
            "confidence": confidence,
            "image_size": image_size,
            "model_path": str(selected_model_path),
            "model_key": selected_model_key,
            "use_ensemble": use_ensemble,
            "show_pitch_roi": show_pitch_roi,
            "model_name": selected_model_name,
            "preset_name": preset_name,
            "field_setup": field_setup,
        }
        st.session_state.live_status_message = (
            "Delivery captured. Camera session ended. Refresh or click Start New Delivery to record again."
        )
        st.rerun()

    if clear_clicked:
        reset_live_delivery_state()
        st.session_state.live_status_message = "Recorded delivery cleared."
        st.rerun()

    from Backends.src.ui.theme import render_section_title

    render_section_title("Recent Delivery Summary")
    show_analysis_output(st.session_state.live_last_result)
