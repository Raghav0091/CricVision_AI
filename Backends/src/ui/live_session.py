from datetime import datetime
from pathlib import Path
from threading import Lock
from tempfile import TemporaryDirectory

import streamlit as st

from Backends.src.analysis.cricket_agent import (
    calculate_detection_quality,
    detect_analysis_warnings,
    generate_coaching_feedback as generate_agent_coaching_feedback,
    generate_delivery_report,
)
from Backends.src.analysis.field_zones import (
    SIMPLE_FIELD_ZONES,
    generate_wagon_wheel_data,
    find_nearest_fielder,
    get_active_field_setup,
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
            "ensemble": False,
        },
        "Old Ball Detector": {
            "path": BALL_MODEL_PATH,
            "ensemble": False,
        },
        "External Ball Model": {
            "path": EXTERNAL_BALL_MODEL_PATH,
            "ensemble": False,
        },
        ENSEMBLE_MODEL_NAME: {
            "path": None,
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
    import cv2

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

    import cv2

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
    import cv2

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
    use_ensemble=False,
    show_pitch_roi=False,
    field_setup=None,
    fps=DEFAULT_RECORDING_FPS,
):
    import cv2

    from Backends.src.ui.video_analysis import (
        convert_to_browser_mp4,
        draw_label,
        draw_pitch_roi,
        draw_search_roi,
        estimate_length_from_bounce,
        estimate_line_from_stumps,
        get_box_center,
        get_nearest_stump_detections,
        has_enough_ball_movement,
        load_ensemble_models,
        load_yolo_model,
        map_model_classes,
        run_ensemble_detection,
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
    stump_model = load_yolo_model(str(CRICKET_OBJECTS_MODEL_PATH))

    if stump_model is None:
        return {
            "success": False,
            "error": f"Stump model not found: {CRICKET_OBJECTS_MODEL_PATH}",
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
        model = load_yolo_model(str(model_path))

        if model is None:
            return {
                "success": False,
                "error": f"Model not found: {model_path}",
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

        progress_bar = st.progress(0)
        status_text = st.empty()

        for frame_index, frame in enumerate(frames):
            annotated_frame = frame.copy()
            low_confidence_ball_detections = []
            recovered_this_frame = False

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

    field_setup = field_setup or {}
    batter_handedness = field_setup.get("batter_handedness", "Right-hand batter")
    bowler_arm = field_setup.get("bowler_arm", "Right-arm bowler")
    camera_view = field_setup.get("camera_view", "Behind bowler")
    fielders = field_setup.get("fielders", [])
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
    from Backends.src.ui.ui_components import badge_row, metric_card, section_header, status_badge
    from Backends.src.ui.field_map import draw_field_map

    section_header("Delivery Review")

    if result is None:
        st.info(
            "No delivery has been analyzed yet. The outcome, processed clip, and download buttons appear here after Done / Analyze Delivery."
        )
        return

    if not result["success"]:
        st.error(result["error"])
        return

    tracking_quality = result.get("overall_tracking_quality", "Poor")
    tracking_tone = "green" if tracking_quality in {"Excellent", "Good"} else "amber"

    badge_row(
        [
            status_badge(f"Model: {result.get('active_model', 'Unknown')}", "cyan"),
            status_badge(f"Preset: {result.get('active_preset', 'Unknown')}", "blue"),
            status_badge(f"Tracking: {tracking_quality}", tracking_tone),
        ]
    )

    outcome_col1, outcome_col2, outcome_col3 = st.columns(3)
    outcome_col1.metric("Line", result["estimated_line"])
    outcome_col2.metric("Length", result["estimated_length"])

    bounce_text = "Not found"
    if result["estimated_bounce_point"] is not None:
        bx, by = result["estimated_bounce_point"]
        bounce_text = f"({bx}, {by})"

    outcome_col3.metric("Bounce Point", bounce_text)

    section_header("Processed Delivery Clip")
    st.video(result["processed_video_bytes"])

    section_header("Analysis Stats")
    stats_cols = st.columns(4)
    with stats_cols[0]:
        metric_card("Ball Frames", str(result["ball_detected_frames"]), "Detected ball frames")
    with stats_cols[1]:
        metric_card("Stump Frames", str(result["stump_detected_frames"]), "Detected stump frames")
    with stats_cols[2]:
        metric_card("Ball Detection Rate", f"{result['ball_detection_rate']:.1f}%", "Coverage across clip")
    with stats_cols[3]:
        metric_card("Low-Conf Frames", str(result.get("low_confidence_ball_frames", 0)), "Saved for review")

    stats_cols_2 = st.columns(4)
    with stats_cols_2[0]:
        metric_card("Tracking Rate", f"{result.get('ball_tracking_rate', 0):.1f}%", "Continuous ball path")
    with stats_cols_2[1]:
        metric_card("Interpolated", str(result.get("interpolated_ball_frames", 0)), "Filled trajectory gaps")
    with stats_cols_2[2]:
        metric_card("Kalman Frames", str(result.get("kalman_predicted_frames", 0)), "Predicted positions")
    with stats_cols_2[3]:
        metric_card("Review Frames", str(result.get("review_frame_count", 0)), "Training export candidates")

    with st.expander("Debug Panel", expanded=False):
        st.write(f"Active model: {result.get('active_model', 'Unknown')}")
        st.write(f"Active preset: {result.get('active_preset', 'Unknown')}")
        st.write(f"ROI size: {result.get('last_roi_size', 'Full frame')}")
        st.write(f"Ball detections: {result.get('total_ball_detections', 0)}")
        st.write(f"Tracker recoveries: {result.get('tracker_recoveries', 0)}")
        st.write(f"Average confidence: {result.get('average_ball_confidence', 0):.2f}")
        timing_col1, timing_col2, timing_col3 = st.columns(3)
        timing_col1.metric(
            "Full Frame Detection Time",
            f"{result.get('full_frame_detection_time_ms', 0):.1f} ms",
        )
        timing_col2.metric(
            "ROI Detection Time",
            f"{result.get('roi_detection_time_ms', 0):.1f} ms",
        )
        timing_col3.metric("ROI Frames", result.get("roi_detected_frames", 0))

    section_header("Batting Direction & Field")
    wagon_wheel = result.get("wagon_wheel", {})
    shot_angle = wagon_wheel.get("shot_angle")
    nearest_fielder = wagon_wheel.get("nearest_fielder")
    nearest_fielder_name = "Unknown" if nearest_fielder is None else nearest_fielder.get("name", "Unknown")
    shot_cols = st.columns(4)
    shot_cols[0].metric("Simple Zone", wagon_wheel.get("simple_zone", "Unknown"))
    shot_cols[1].metric("Detailed Zone", wagon_wheel.get("detailed_zone", "Unknown"))
    shot_cols[2].metric("Shot Angle", "Unknown" if shot_angle is None else f"{shot_angle:.1f} deg")
    shot_cols[3].metric("Nearest Fielder", nearest_fielder_name)

    context_cols = st.columns(3)
    context_cols[0].metric("Batter", result.get("batter_handedness", "Unknown"))
    context_cols[1].metric("Bowler Arm", result.get("bowler_arm", "Unknown"))
    context_cols[2].metric("Confidence", wagon_wheel.get("confidence", "Low"))
    st.info(wagon_wheel.get("suggested_adjustment", "No field adjustment suggestion available."))
    st.pyplot(
        draw_field_map(
            shot_angle=shot_angle,
            selected_zone=wagon_wheel.get("detailed_zone", "Unknown"),
            fielders=result.get("field_setup", {}).get("fielders", []),
            batter_handedness=result.get("batter_handedness", "Right-handed"),
            umpires=result.get("field_setup", {}).get("umpires"),
        )
    )
    correction_col1, correction_col2 = st.columns([2, 1])

    with correction_col1:
        corrected_zone = st.selectbox(
            "Manual correction: actual shot zone",
            ["No correction"] + SIMPLE_FIELD_ZONES,
            key="live_session_field_zone_correction",
        )

    with correction_col2:
        save_correction = st.button("Save Zone Correction", key="save_live_field_zone_correction")

    if save_correction:
        if corrected_zone == "No correction":
            st.warning("Choose an actual zone before saving a correction.")
        else:
            save_field_analysis_history(
                {
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "source": "live_session_correction",
                    "batter_handedness": result.get("batter_handedness", ""),
                    "bowler_arm": result.get("bowler_arm", ""),
                    "camera_view": result.get("camera_view", ""),
                    "preset": result.get("field_setup", {}).get("preset", "Custom"),
                    "simple_zone": wagon_wheel.get("simple_zone", "Unknown"),
                    "detailed_zone": wagon_wheel.get("detailed_zone", "Unknown"),
                    "shot_angle": "" if shot_angle is None else f"{shot_angle:.2f}",
                    "nearest_fielder": nearest_fielder_name,
                    "confidence": wagon_wheel.get("confidence", "Low"),
                    "corrected_zone": corrected_zone,
                }
            )
            result["wagon_wheel"]["corrected_zone"] = corrected_zone
            st.session_state.live_last_result = result
            st.success("Field-zone correction saved for future review.")

    show_delivery_report(result)
    show_cricket_delivery_report(result)

    st.download_button(
        label="Download Processed Delivery Clip",
        data=result["processed_video_bytes"],
        file_name=result["processed_file_name"],
        mime="video/mp4",
    )

    if st.button("Export Review Frames for Training", key="export_review_frames_live"):
        from Backends.src.ui.video_analysis import create_review_frames_zip

        zip_path, file_count = create_review_frames_zip()

        if file_count == 0:
            st.warning("No review frames are available yet.")
        else:
            with open(zip_path, "rb") as zip_file:
                st.download_button(
                    label="Download Review Frames ZIP",
                    data=zip_file,
                    file_name=zip_path.name,
                    mime="application/zip",
                    key="download_review_frames_zip_live",
                )


def show_current_field_setup_preview(field_setup, draw_field_map):
    from Backends.src.ui.ui_components import section_header

    section_header("Current Field Setup")
    setup_cols = st.columns(4)
    setup_cols[0].metric("Preset", field_setup.get("preset", "Attacking Test Field"))
    setup_cols[1].metric("Batter", field_setup.get("batter_handedness", "Right-hand batter"))
    setup_cols[2].metric("Bowler Arm", field_setup.get("bowler_arm", "Right-arm bowler"))
    setup_cols[3].metric("Camera View", field_setup.get("camera_view", "Behind bowler"))

    if field_setup.get("is_default_setup"):
        st.info("No saved field setup found. Using default Attacking Test Field.")

    st.info("Go to Field Map page to adjust field setup.")
    st.pyplot(
        draw_field_map(
            shot_angle=None,
            selected_zone="Unknown",
            fielders=field_setup.get("fielders", []),
            batter_handedness=field_setup.get("batter_handedness", "Right-handed"),
            umpires=field_setup.get("umpires"),
        )
    )


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
    from Backends.src.ui.ui_components import (
        badge_row,
        card,
        info_panel,
        page_header,
        section_header,
        status_badge,
        workflow_step,
    )

    page_header(
        "Live Session",
        "Record one clean delivery from the live camera, then analyze it after the ball is bowled.",
    )

    initialize_live_session_state()

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

    section_header("Live Workflow")
    workflow_cols = st.columns(5)
    workflow_labels = [
        "Start Camera",
        "Start Delivery Recording",
        "Bowl",
        "Done / Analyze",
        "Review Processed Delivery",
    ]
    for index, label in enumerate(workflow_labels):
        with workflow_cols[index]:
            workflow_step(index + 1, label)

    info_panel(
        "<strong>Live camera preview stays clean.</strong> Analysis overlays are generated after delivery."
    )

    settings_tab, camera_tab = st.tabs(["Model Settings", "Camera & Controls"])

    with settings_tab:
        from Backends.src.ui.field_map import draw_field_map

        selected_model_name = st.selectbox(
            "Choose detection model",
            list(model_options.keys()),
            key="live_session_model",
        )
        selected_model = model_options[selected_model_name]
        selected_model_path = selected_model["path"]
        use_ensemble = selected_model.get("ensemble", False)

        if not use_ensemble and not selected_model_path.exists():
            st.error(f"Model not found: {selected_model_path}")
            info_panel("Make sure the selected model file exists in the correct Models folder.")
            return

        badge_row([status_badge(f"Model: {selected_model_name}", "cyan")])

        if use_ensemble:
            from Backends.src.ui.video_analysis import get_available_ensemble_model_names

            active_model_names = get_available_ensemble_model_names()

            if active_model_names:
                st.caption("Active ensemble models: " + ", ".join(active_model_names))
            else:
                st.warning("No configured ensemble model files were found.")
        else:
            st.caption(f"Model path: {selected_model_path}")

        preset_name = st.selectbox(
            "Detection preset",
            list(DETECTION_PRESETS.keys()),
            index=1,
            key="live_session_preset",
        )
        active_preset = DETECTION_PRESETS[preset_name]
        confidence = active_preset["confidence"]
        image_size = active_preset["imgsz"]
        badge_row([status_badge(f"Preset: {preset_name}", "blue")])

        st.caption(
            f"Active preset: {preset_name} | imgsz={image_size} | confidence={confidence:.2f}"
        )

        show_pitch_roi = st.checkbox("Show Pitch ROI", value=False, key="live_session_show_roi")
        badge_row(
            [
                status_badge(
                    f"ROI Overlay: {'On' if show_pitch_roi else 'Off'}",
                    "green" if show_pitch_roi else "muted",
                )
            ]
        )

        info_panel(
            "If the selected model does not include stumps, line detection may remain Unknown."
        )

        section_header("Set Field Before Delivery")
        field_setup = get_active_field_setup()
        show_current_field_setup_preview(field_setup, draw_field_map)

    with camera_tab:
        card(
            title="Camera Tips",
            content_html=(
                "Phone users: open this Streamlit app in your phone browser, allow camera access, "
                "use landscape mode, and use the back camera behind the bowler."
            ),
        )

    if st.session_state.live_status_message:
        st.info(st.session_state.live_status_message)
        st.session_state.live_status_message = None

    section_header("Step 1 — Start Camera")

    try:
        from streamlit_webrtc import RTCConfiguration, webrtc_streamer
    except ImportError:
        st.error("streamlit-webrtc is not installed. Add streamlit-webrtc to requirements.txt.")
        return

    rtc_config = RTCConfiguration(
        {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
            ],
        }
    )

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
    recording_state = st.session_state.live_recording_state

    section_header("Steps 2–4 — Record & Analyze")
    button_col1, button_col2, button_col3 = st.columns([1, 1, 1])

    with button_col1:
        start_clicked = st.button(
            "Start Delivery Recording",
            type="primary",
            disabled=recording_state.recording,
        )

    with button_col2:
        done_clicked = st.button(
            "Done / Analyze Delivery",
            disabled=not recording_state.recording,
        )

    with button_col3:
        clear_clicked = st.button(
            "Clear Delivery",
            disabled=recording_state.recording,
        )

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

    recorded_count = recording_state.get_frame_count()

    if recording_state.recording:
        st.info(f"Recording delivery... captured {recorded_count} frames.")
    elif recorder is not None:
        st.caption("Live preview is clean. Click Start Delivery Recording when you are ready to bowl.")
    else:
        st.info("Start the camera stream to enable delivery recording.")

    if st.session_state.live_last_result is None:
        section_header("Step 5 — Review Processed Delivery")
        card(
            title="Waiting for Analysis",
            content_html=(
                "After you click <strong>Done / Analyze Delivery</strong>, the processed clip, "
                "delivery report, stats, and download button will appear here."
            ),
        )
    else:
        show_analysis_output(st.session_state.live_last_result)
