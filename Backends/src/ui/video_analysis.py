import tempfile
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import imageio_ffmpeg
import streamlit as st
from ultralytics import YOLO

from Backends.src.analysis.cricket_agent import (
    calculate_detection_quality,
    detect_analysis_warnings,
    generate_coaching_feedback,
    generate_delivery_report,
)
from Backends.src.tracking.ball_tracking_utils import (
    calculate_tracking_quality,
    detect_bounce_by_direction_change,
    interpolate_missing_positions,
    smooth_trajectory,
)


BALL_MODEL_PATH = Path("Models/ball_detector/best.pt")
CRICKET_OBJECTS_MODEL_PATH = Path("Models/cricket_objects/best.pt")
EXTERNAL_BALL_MODEL_PATH = Path("Models/cricket_objects/best_external.pt")
OUTPUT_DIR = Path("outputs/video_analysis")


@st.cache_resource
def load_yolo_model(model_path_str):
    model_path = Path(model_path_str)

    if not model_path.exists():
        return None

    return YOLO(str(model_path))


def get_model_options():
    return {
        "Ball + Stump Detector": {
            "path": CRICKET_OBJECTS_MODEL_PATH,
        },
        "External Ball Model": {
            "path": EXTERNAL_BALL_MODEL_PATH,
        },
        "Old Ball Detector": {
            "path": BALL_MODEL_PATH,
        },
    }


def get_model_names(model):
    names = getattr(model, "names", {})

    if isinstance(names, dict):
        return names

    if isinstance(names, (list, tuple)):
        return {index: name for index, name in enumerate(names)}

    return {}


def map_model_classes(model):
    class_names = {}

    for class_id, raw_name in get_model_names(model).items():
        normalized_name = str(raw_name).lower()

        if "ball" in normalized_name:
            class_names[int(class_id)] = "ball"
        elif (
            "stump" in normalized_name
            or "stumps" in normalized_name
            or "wicket" in normalized_name
        ):
            class_names[int(class_id)] = "stump"

    return class_names


def draw_label(frame, text, x, y, color):
    y = max(y, 25)
    label_width = len(text) * 10 + 10

    cv2.rectangle(
        frame,
        (x, y - 24),
        (x + label_width, y),
        color,
        -1,
    )

    cv2.putText(
        frame,
        text,
        (x + 5, y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )


def get_box_center(x1, y1, x2, y2):
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y


def choose_main_ball(ball_detections, previous_center=None):
    if not ball_detections:
        return None

    if previous_center is None:
        return max(ball_detections, key=lambda item: item["confidence"])

    previous_x, previous_y = previous_center

    def distance_from_previous(item):
        center_x, center_y = item["center"]
        return ((center_x - previous_x) ** 2 + (center_y - previous_y) ** 2) ** 0.5

    return min(ball_detections, key=distance_from_previous)


def estimate_bounce_point(trajectory_points, min_points=6):
    if len([point for point in trajectory_points if point is not None]) < min_points:
        return None

    bounce_result = detect_bounce_by_direction_change(trajectory_points)

    if bounce_result is None:
        return None

    return bounce_result["point"]


def convert_to_browser_mp4(input_path, output_path):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_path),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    return output_path


def get_nearest_stump_detections(stump_detections_by_frame, frame_index):
    if frame_index is None or not stump_detections_by_frame:
        return []

    frame_index = min(frame_index, len(stump_detections_by_frame) - 1)

    for detections in reversed(stump_detections_by_frame[: frame_index + 1]):
        if detections:
            return detections

    for detections in stump_detections_by_frame[frame_index + 1 :]:
        if detections:
            return detections

    return []


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


def process_video(video_path, output_path, model_path, class_names=None, confidence=0.25, imgsz=640):
    model = load_yolo_model(str(model_path))

    if model is None:
        return {
            "success": False,
            "error": f"Model not found: {model_path}",
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        cap.release()
        return {
            "success": False,
            "error": "Could not read video width/height.",
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
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

    trajectory_points = []
    ball_positions = []
    stump_detections_by_frame = []
    max_trajectory_points = 35

    previous_ball_center = None

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
    
    valid_ball_track_started = False
    min_track_points_for_bounce = 8
    min_movement_distance = 40
    min_ball_confidence_for_tracking = 0.35 

    while True:
        success, frame = cap.read()

        if not success:
            break

        results = model.predict(
            source=frame,
            conf=confidence,
            imgsz=imgsz,
            verbose=False,
        )

        result = results[0]
        annotated_frame = frame.copy()

        ball_detections = []
        stump_detections = []

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                class_name = class_names.get(class_id)

                if class_name is None:
                    continue

                center = get_box_center(x1, y1, x2, y2)

                detection = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "box": (x1, y1, x2, y2),
                    "center": center,
                }

                if class_name == "ball":
                    if conf >= min_ball_confidence_for_tracking:
                        ball_detections.append(detection)
                        confidence_values.append(conf)

                elif class_name == "stump":
                    stump_detections.append(detection)

        if ball_detections:
            ball_detected_frames += 1
            total_ball_detections += len(ball_detections)

        if stump_detections:
            stump_detected_frames += 1
            total_stump_detections += len(stump_detections)

        stump_detections_by_frame.append(stump_detections)

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
                    bounce_stump_detections
                )

                estimated_length = estimate_length_from_bounce(
                    estimated_bounce_point,
                    height 
                )


        else:
            ball_positions.append(None)
            missing_ball_frames += 1

            if missing_ball_frames >= max_missing_ball_frames:
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

        cv2.putText(
            annotated_frame,
            f"Frame: {frame_index}/{total_frames}",
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

        writer.write(annotated_frame)

        frame_index += 1

        if total_frames > 0:
            progress = min(frame_index / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_index}/{total_frames}")
        else:
            status_text.text(f"Processing frame {frame_index}")

    cap.release()
    writer.release()

    progress_bar.empty()
    status_text.empty()

    if frame_index == 0:
        return {
            "success": False,
            "error": "No frames were processed. The uploaded video may be corrupted or unsupported.",
        }

    ball_detection_rate = 0
    stump_detection_rate = 0

    if frame_index > 0:
        ball_detection_rate = (ball_detected_frames / frame_index) * 100
        stump_detection_rate = (stump_detected_frames / frame_index) * 100

    average_confidence = 0

    if confidence_values:
        average_confidence = sum(confidence_values) / len(confidence_values)

    tracking_quality = calculate_tracking_quality(ball_positions, frame_index)

    return {
        "success": True,
        "output_path": output_path,
        "total_frames": frame_index,
        "ball_detected_frames": ball_detected_frames,
        "stump_detected_frames": stump_detected_frames,
        "total_ball_detections": total_ball_detections,
        "total_stump_detections": total_stump_detections,
        "ball_detection_rate": ball_detection_rate,
        "ball_tracking_rate": tracking_quality["tracking_rate"],
        "interpolated_ball_frames": tracking_quality["interpolated_frames"],
        "stump_detection_rate": stump_detection_rate,
        "average_ball_confidence": average_confidence,
        "estimated_bounce_point": estimated_bounce_point,
        "estimated_bounce_frame": estimated_bounce_frame,
        "estimated_line": estimated_line,
        "estimated_length": estimated_length,
    }


def show_video_analysis_page():
    st.title("🎥 Cricket Video Analysis")
    st.markdown(
        "Upload a bowling clip. CricVision AI will detect balls, stumps, draw the ball trajectory, and estimate the bounce/pitch point."
    )

    model_options = get_model_options()

    selected_model_name = st.selectbox(
        "Choose detection model",
        list(model_options.keys()),
    )

    selected_model = model_options[selected_model_name]
    selected_model_path = selected_model["path"]

    if not selected_model_path.exists():
        st.error(f"Model not found: {selected_model_path}")
        st.info("Make sure your model file is inside the correct Models folder.")
        st.stop()

    st.success(f"Loaded model: {selected_model_name}")
    st.caption(f"Model path: {selected_model_path}")
    st.info("If selected model does not include stumps, line detection may be Unknown.")

    uploaded_video = st.file_uploader(
        "Upload bowling video from phone or camera",
        type=["mp4", "mov", "avi", "mkv"],
    )

    col1, col2 = st.columns(2)

    with col1:
        confidence = st.slider(
            "Detection confidence",
            min_value=0.05,
            max_value=0.90,
            value=0.25,
            step=0.05,
        )

    with col2:
        image_size = st.selectbox(
            "Image size",
            options=[640, 768, 960],
            index=0,
        )

    st.info(
        "For phone videos, use landscape mode if possible. Good lighting and a stable camera will improve tracking."
    )

    if uploaded_video is not None:
        st.subheader("Original Video")
        st.video(uploaded_video)

        if st.button("Analyze Video", type="primary"):
            uploaded_video.seek(0)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
                temp_input.write(uploaded_video.read())
                input_video_path = Path(temp_input.name)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            raw_output_path = OUTPUT_DIR / f"raw_cricvision_analysis_{timestamp}.mp4"
            browser_output_path = OUTPUT_DIR / f"cricvision_analysis_{timestamp}.mp4"

            with st.spinner("Analyzing video..."):
                result = process_video(
                    video_path=input_video_path,
                    output_path=raw_output_path,
                    model_path=selected_model_path,
                    confidence=confidence,
                    imgsz=image_size,
                )

            if not result["success"]:
                st.error(result["error"])
                return

            try:
                final_video_path = convert_to_browser_mp4(
                    input_path=result["output_path"],
                    output_path=browser_output_path,
                )

                result["output_path"] = final_video_path

            except Exception as e:
                st.error(f"Video conversion failed: {e}")
                st.info(
                    "The analysis worked, but the final video could not be converted for browser playback."
                )
                return

            st.success("Video analysis completed.")

            st.subheader("Processed Video")

            with open(result["output_path"], "rb") as video_file:
                video_bytes = video_file.read()

            st.video(video_bytes)

            st.subheader("Analysis Stats")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Total Frames", result["total_frames"])
            col2.metric("Ball Frames", result["ball_detected_frames"])
            col3.metric("Stump Frames", result["stump_detected_frames"])
            col4.metric("Ball Detection Rate", f"{result['ball_detection_rate']:.1f}%")

            col5, col6, col7 = st.columns(3)

            col5.metric("Total Ball Detections", result["total_ball_detections"])
            col6.metric("Total Stump Detections", result["total_stump_detections"])
            col7.metric("Avg Ball Confidence", f"{result['average_ball_confidence']:.2f}")

            col8, col9 = st.columns(2)
            col8.metric("Ball Tracking Rate", f"{result.get('ball_tracking_rate', 0):.1f}%")
            col9.metric("Interpolated Ball Frames", result.get("interpolated_ball_frames", 0))

            st.subheader("Bounce / Pitch Estimate")

            if result["estimated_bounce_point"] is not None:
                bx, by = result["estimated_bounce_point"]

                col10, col11, col12, col13, col14 = st.columns(5)

                col10.metric("Bounce Frame", result["estimated_bounce_frame"])
                col11.metric("Bounce X", bx)
                col12.metric("Bounce Y", by)
                col13.metric("Estimated Line", result["estimated_line"])
                col14.metric("Estimated Length", result["estimated_length"])
                
                st.success("Estimated bounce/pitch point found.")
            else:
                st.warning("Bounce/pitch point was not found. Try a clearer or longer clip.")

            show_cricket_delivery_report(result)

            st.caption(f"Saved output: {result['output_path']}")

            with open(result["output_path"], "rb") as file:
                st.download_button(
                    label="Download Processed Video",
                    data=file,
                    file_name="cricvision_processed_video.mp4",
                    mime="video/mp4",
                )
                
def estimate_line_from_stumps(bounce_point, stump_detections):
    """
    Estimate cricket line using bounce point and stump position.

    Simple version:
    - left of stumps = off side
    - inside stump width = middle
    - right of stumps = leg side

    This assumes right-handed batter view for now.
    """

    if bounce_point is None or not stump_detections:
        return "Unknown"

    bx, by = bounce_point

    # Use widest stump box if multiple stump boxes are detected
    main_stump = max(
        stump_detections,
        key=lambda item: item["box"][2] - item["box"][0]
    )

    x1, y1, x2, y2 = main_stump["box"]

    stump_width = x2 - x1
    margin = int(stump_width * 0.4)

    if bx < x1 - margin:
        return "Off side"
    elif bx > x2 + margin:
        return "Leg side"
    else:
        return "Middle"
    

def estimate_length_from_bounce(bounce_point, frame_height):
    if bounce_point is None or frame_height <= 0:
        return "Unknown"

    bx, by = bounce_point

    bounce_ratio = by / frame_height

    if bounce_ratio >= 0.82:
        return "Yorker"
    elif bounce_ratio >= 0.68:
        return "Full"
    elif bounce_ratio >= 0.48:
        return "Good Length"
    else:
        return "Short"
    
def has_enough_ball_movement(trajectory_points, min_distance=40):
    if len(trajectory_points) < 2:
        return False

    start_x, start_y = trajectory_points[0]
    end_x, end_y = trajectory_points[-1]

    distance = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5

    return distance >= min_distance
