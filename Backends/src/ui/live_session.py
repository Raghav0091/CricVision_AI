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
from Backends.src.tracking.ball_tracking_utils import (
    calculate_tracking_quality,
    detect_bounce_by_direction_change,
    interpolate_missing_positions,
    smooth_trajectory,
)


CRICKET_OBJECTS_MODEL_PATH = Path("Models/cricket_objects/best.pt")
REVIEW_FRAMES_DIR = Path("outputs/review_frames")

CLASS_NAMES = {
    0: "ball",
    1: "stump",
}

LOW_CONFIDENCE_REVIEW_THRESHOLD = 0.35
MIN_TRAJECTORY_POINTS_FOR_BOUNCE = 8
MIN_MOVEMENT_DISTANCE = 40
SHORT_MISSING_BALL_SMOOTHING_FRAMES = 8
MAX_MISSING_BALL_FRAMES = 12
MAX_TRAJECTORY_POINTS = 35
MAX_RECORDED_FRAMES = 450
DEFAULT_RECORDING_FPS = 25


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
        class_name = class_names.get(class_id, f"class_{class_id}")

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

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
    fps=DEFAULT_RECORDING_FPS,
):
    import cv2

    from Backends.src.ui.video_analysis import (
        convert_to_browser_mp4,
        draw_label,
        estimate_length_from_bounce,
        estimate_line_from_stumps,
        get_box_center,
        get_nearest_stump_detections,
        has_enough_ball_movement,
        load_yolo_model,
    )

    if not frames:
        return {
            "success": False,
            "error": "No frames were recorded. Click Start Delivery Recording before bowling.",
        }

    model = load_yolo_model(str(CRICKET_OBJECTS_MODEL_PATH))

    if model is None:
        return {
            "success": False,
            "error": f"Model not found: {CRICKET_OBJECTS_MODEL_PATH}",
        }

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

        trajectory_points = []
        ball_positions = []
        stump_detections_by_frame = []
        previous_ball_center = None
        missing_ball_frames = 0
        last_stump_detections = []

        estimated_bounce_point = None
        estimated_bounce_frame = None
        estimated_line = "Unknown"
        estimated_length = "Unknown"

        progress_bar = st.progress(0)
        status_text = st.empty()

        for frame_index, frame in enumerate(frames):
            results = model.predict(
                source=frame,
                conf=min(confidence, 0.10),
                imgsz=image_size,
                verbose=False,
            )

            result = results[0]
            annotated_frame = frame.copy()
            ball_detections, low_confidence_ball_detections, stump_detections = collect_detections(
                result,
                CLASS_NAMES,
                get_box_center,
                confidence,
            )

            if low_confidence_ball_detections:
                low_confidence_ball_frames += 1
                save_low_confidence_review_frame(
                    frame,
                    low_confidence_ball_detections,
                    timestamp,
                    frame_index,
                )

            if ball_detections:
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
                ball_positions.append(None)
                missing_ball_frames += 1

                if missing_ball_frames >= MAX_MISSING_BALL_FRAMES:
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
        "stump_detection_rate": stump_detection_rate,
        "average_ball_confidence": average_ball_confidence,
        "estimated_bounce_point": estimated_bounce_point,
        "estimated_bounce_frame": estimated_bounce_frame,
        "estimated_line": estimated_line,
        "estimated_length": estimated_length,
        "ball_detection_difficult": ball_detection_rate < 35 or low_confidence_ball_frames > 0,
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
    st.subheader("Analysis Output")

    if result is None:
        st.info(
            "No delivery has been analyzed yet. The outcome, processed clip, and download buttons appear here after Done / Analyze Delivery."
        )
        return

    if not result["success"]:
        st.error(result["error"])
        return

    outcome_col1, outcome_col2, outcome_col3 = st.columns(3)
    outcome_col1.metric("Line", result["estimated_line"])
    outcome_col2.metric("Length", result["estimated_length"])

    bounce_text = "Not found"
    if result["estimated_bounce_point"] is not None:
        bx, by = result["estimated_bounce_point"]
        bounce_text = f"({bx}, {by})"

    outcome_col3.metric("Bounce Point", bounce_text)

    st.subheader("Processed Delivery Clip")
    st.video(result["processed_video_bytes"])

    st.subheader("Analysis Stats")
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    stats_col1.metric("Ball Frames", result["ball_detected_frames"])
    stats_col2.metric("Stump Frames", result["stump_detected_frames"])
    stats_col3.metric("Ball Detection Rate", f"{result['ball_detection_rate']:.1f}%")
    stats_col4.metric("Low-Conf Review Frames", result.get("low_confidence_ball_frames", 0))

    stats_col5, stats_col6 = st.columns(2)
    stats_col5.metric("Ball Tracking Rate", f"{result.get('ball_tracking_rate', 0):.1f}%")
    stats_col6.metric("Interpolated Ball Frames", result.get("interpolated_ball_frames", 0))

    show_delivery_report(result)
    show_cricket_delivery_report(result)

    st.download_button(
        label="Download Processed Delivery Clip",
        data=result["processed_video_bytes"],
        file_name=result["processed_file_name"],
        mime="video/mp4",
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
    st.title("Live Cricket Analysis")
    st.markdown(
        "Record one clean delivery from the live camera, then analyze it after the ball is bowled."
    )

    initialize_live_session_state()

    if not CRICKET_OBJECTS_MODEL_PATH.exists():
        st.error(f"Model not found: {CRICKET_OBJECTS_MODEL_PATH}")
        st.info("Make sure the Ball + Stump Detector file exists in Models/cricket_objects/best.pt.")
        return

    st.success("Model ready: Ball + Stump Detector")
    st.caption(f"Model path: {CRICKET_OBJECTS_MODEL_PATH}")

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

    controls_col, guidance_col = st.columns([1, 1])

    with controls_col:
        confidence = st.slider(
            "Detection confidence",
            min_value=0.10,
            max_value=0.70,
            value=0.25,
            step=0.05,
        )

        image_size = st.selectbox(
            "Image size",
            options=[640, 768, 960],
            index=2,
        )

    with guidance_col:
        st.warning(
            "Live preview stays clean. Detection boxes and trajectory are drawn only after you click Done / Analyze Delivery."
        )
        st.info(
            "Phone users: open this Streamlit app in your phone browser, allow camera access, use landscape mode, and use the back camera behind the bowler."
        )

    if st.session_state.live_status_message:
        st.info(st.session_state.live_status_message)
        st.session_state.live_status_message = None

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

    show_analysis_output(st.session_state.live_last_result)
