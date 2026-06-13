from pathlib import Path

import streamlit as st


CRICKET_OBJECTS_MODEL_PATH = Path("Models/cricket_objects/best.pt")
CLASS_NAMES = {
    0: "ball",
    1: "stump",
}

MIN_TRACKING_CONFIDENCE = 0.35
MIN_TRAJECTORY_POINTS_FOR_BOUNCE = 8
MIN_MOVEMENT_DISTANCE = 40
MAX_MISSING_BALL_FRAMES = 12
MAX_TRAJECTORY_POINTS = 35


@st.cache_resource
def load_yolo_model(model_path_str):
    from ultralytics import YOLO

    model_path = Path(model_path_str)

    if not model_path.exists():
        return None

    return YOLO(str(model_path))


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


def estimate_bounce_point(trajectory_points, min_points=MIN_TRAJECTORY_POINTS_FOR_BOUNCE):
    if len(trajectory_points) < min_points:
        return None

    return max(trajectory_points, key=lambda point: point[1])


def estimate_line_from_stumps(bounce_point, stump_detections):
    if bounce_point is None or not stump_detections:
        return "Unknown"

    bx, _ = bounce_point

    main_stump = max(
        stump_detections,
        key=lambda item: item["box"][2] - item["box"][0],
    )

    x1, _, x2, _ = main_stump["box"]
    stump_width = x2 - x1
    margin = int(stump_width * 0.4)

    if bx < x1 - margin:
        return "Off side"
    if bx > x2 + margin:
        return "Leg side"
    return "Middle"


def estimate_length_from_bounce(bounce_point, frame_height):
    if bounce_point is None or frame_height <= 0:
        return "Unknown"

    _, by = bounce_point
    bounce_ratio = by / frame_height

    if bounce_ratio >= 0.82:
        return "Yorker"
    if bounce_ratio >= 0.68:
        return "Full"
    if bounce_ratio >= 0.48:
        return "Good Length"
    return "Short"


def has_enough_ball_movement(trajectory_points, min_distance=MIN_MOVEMENT_DISTANCE):
    if len(trajectory_points) < 2:
        return False

    start_x, start_y = trajectory_points[0]
    end_x, end_y = trajectory_points[-1]
    distance = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5

    return distance >= min_distance


def draw_label(frame, text, x, y, color):
    import cv2

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


def create_live_processor_class(confidence, image_size):
    import av
    import cv2
    from streamlit_webrtc import VideoProcessorBase

    class LiveCricketAnalysisProcessor(VideoProcessorBase):
        def __init__(self):
            self.model = load_yolo_model(str(CRICKET_OBJECTS_MODEL_PATH))
            self.confidence = confidence
            self.image_size = image_size
            self.frame_index = 0
            self.trajectory_points = []
            self.previous_ball_center = None
            self.missing_ball_frames = 0
            self.last_stump_detections = []
            self.estimated_bounce_point = None
            self.estimated_bounce_frame = None
            self.estimated_line = "Unknown"
            self.estimated_length = "Unknown"

        def recv(self, frame):
            image = frame.to_ndarray(format="bgr24")
            frame_height = image.shape[0]

            if self.model is None:
                self._draw_missing_model_message(image)
                return av.VideoFrame.from_ndarray(image, format="bgr24")

            annotated_frame = self._process_frame(image, frame_height)
            self.frame_index += 1

            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

        def _process_frame(self, image, frame_height):
            results = self.model.predict(
                source=image,
                conf=self.confidence,
                imgsz=self.image_size,
                verbose=False,
            )

            result = results[0]
            annotated_frame = image.copy()
            ball_detections, stump_detections = self._collect_detections(result)

            if stump_detections:
                self.last_stump_detections = stump_detections

            self._draw_detections(annotated_frame, ball_detections, stump_detections)
            main_ball = choose_main_ball(ball_detections, self.previous_ball_center)
            self._update_tracking(main_ball, frame_height)
            self._draw_trajectory(annotated_frame)
            self._draw_bounce_point(annotated_frame)
            self._draw_dashboard(
                annotated_frame,
                ball_count=len(ball_detections),
                stump_count=len(stump_detections),
                main_ball=main_ball,
            )

            return annotated_frame

        def _collect_detections(self, result):
            ball_detections = []
            stump_detections = []

            if result.boxes is None or len(result.boxes) == 0:
                return ball_detections, stump_detections

            for box in result.boxes:
                class_id = int(box.cls[0].cpu().numpy())
                box_confidence = float(box.conf[0].cpu().numpy())
                class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                detection = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": box_confidence,
                    "box": (x1, y1, x2, y2),
                    "center": get_box_center(x1, y1, x2, y2),
                }

                if class_name == "ball":
                    if box_confidence >= MIN_TRACKING_CONFIDENCE:
                        ball_detections.append(detection)
                elif class_name == "stump":
                    stump_detections.append(detection)

            return ball_detections, stump_detections

        def _draw_detections(self, frame, ball_detections, stump_detections):
            for detection in ball_detections:
                x1, y1, x2, y2 = detection["box"]
                center_x, center_y = detection["center"]
                box_confidence = detection["confidence"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
                draw_label(frame, f"ball {box_confidence:.2f}", x1, y1, (0, 180, 180))

            for detection in stump_detections:
                x1, y1, x2, y2 = detection["box"]
                box_confidence = detection["confidence"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                draw_label(frame, f"stump {box_confidence:.2f}", x1, y1, (255, 100, 0))

        def _update_tracking(self, main_ball, frame_height):
            if main_ball is None:
                self.missing_ball_frames += 1

                if self.missing_ball_frames >= MAX_MISSING_BALL_FRAMES:
                    self.trajectory_points.clear()
                    self.previous_ball_center = None
                    self.estimated_bounce_point = None
                    self.estimated_bounce_frame = None
                    self.estimated_line = "Unknown"
                    self.estimated_length = "Unknown"

                return

            self.missing_ball_frames = 0
            self.previous_ball_center = main_ball["center"]
            self.trajectory_points.append(main_ball["center"])

            if len(self.trajectory_points) > MAX_TRAJECTORY_POINTS:
                self.trajectory_points.pop(0)

            if self.estimated_bounce_point is not None:
                return

            has_track = len(self.trajectory_points) >= MIN_TRAJECTORY_POINTS_FOR_BOUNCE
            has_movement = has_enough_ball_movement(
                self.trajectory_points,
                MIN_MOVEMENT_DISTANCE,
            )

            if not has_track or not has_movement:
                return

            bounce_point = estimate_bounce_point(
                self.trajectory_points,
                MIN_TRAJECTORY_POINTS_FOR_BOUNCE,
            )

            if bounce_point is None:
                return

            stump_context = self.last_stump_detections
            self.estimated_bounce_point = bounce_point
            self.estimated_bounce_frame = self.frame_index
            self.estimated_line = estimate_line_from_stumps(bounce_point, stump_context)
            self.estimated_length = estimate_length_from_bounce(bounce_point, frame_height)

        def _draw_trajectory(self, frame):
            if len(self.trajectory_points) < 2:
                return

            for index in range(1, len(self.trajectory_points)):
                cv2.line(
                    frame,
                    self.trajectory_points[index - 1],
                    self.trajectory_points[index],
                    (0, 255, 255),
                    3,
                )

        def _draw_bounce_point(self, frame):
            if self.estimated_bounce_point is None:
                return

            bx, by = self.estimated_bounce_point

            cv2.circle(frame, (bx, by), 10, (0, 0, 255), -1)
            cv2.circle(frame, (bx, by), 16, (255, 255, 255), 2)

            cv2.putText(
                frame,
                f"Bounce Frame: {self.estimated_bounce_frame}",
                (bx + 15, max(by - 15, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        def _draw_dashboard(self, frame, ball_count, stump_count, main_ball):
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
            if self.estimated_bounce_frame is not None:
                bounce_text = f"Frame {self.estimated_bounce_frame}"

            confidence_text = "None"
            if main_ball is not None:
                confidence_text = f"{main_ball['confidence']:.2f}"

            dashboard_lines = [
                (f"Frame: {self.frame_index}", (255, 255, 255)),
                (f"Balls in frame: {ball_count}", (0, 255, 255)),
                (f"Stumps in frame: {stump_count}", (255, 160, 0)),
                (
                    f"Trajectory: {len(self.trajectory_points)} | Missing: {self.missing_ball_frames}",
                    (0, 255, 255),
                ),
                (f"Main ball confidence: {confidence_text}", (255, 255, 255)),
                (f"Bounce: {bounce_text}", (0, 0, 255)),
                (f"Line: {self.estimated_line}", (255, 255, 0)),
                (f"Length: {self.estimated_length}", (0, 255, 0)),
            ]

            for index, (text, color) in enumerate(dashboard_lines):
                cv2.putText(
                    frame,
                    text,
                    (panel_x + 15, panel_y + 30 + index * 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    color,
                    2,
                )

        def _draw_missing_model_message(self, frame):
            cv2.putText(
                frame,
                f"Model not found: {CRICKET_OBJECTS_MODEL_PATH}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

    return LiveCricketAnalysisProcessor


def show_live_session_page():
    st.title("Live Cricket Analysis")
    st.markdown(
        "Real-time camera preview with cricket ball, stump, trajectory, bounce, line, and length overlays."
    )

    if not CRICKET_OBJECTS_MODEL_PATH.exists():
        st.error(f"Model not found: {CRICKET_OBJECTS_MODEL_PATH}")
        st.info("Make sure the Ball + Stump Detector file exists in Models/cricket_objects/best.pt.")
        return

    st.success("Model ready: Ball + Stump Detector")
    st.caption(f"Model path: {CRICKET_OBJECTS_MODEL_PATH}")

    controls_col, guidance_col = st.columns([1, 1])

    with controls_col:
        confidence = st.slider(
            "Detection confidence",
            min_value=0.05,
            max_value=0.90,
            value=0.25,
            step=0.05,
        )

        image_size = st.selectbox(
            "Image size",
            options=[640, 768, 960],
            index=0,
        )

    with guidance_col:
        st.warning(
            "Live detection is experimental. Results depend heavily on camera angle, lighting, distance, and network latency."
        )
        st.info(
            "Phone users: open this Streamlit app in your phone browser, allow camera access, use landscape mode, and place the phone behind the bowler facing the batter."
        )

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

    processor_class = create_live_processor_class(confidence, image_size)

    st.caption(
        "Tracking guardrails: ball confidence >= 0.35, at least 8 trajectory points, at least 40px movement, and reset after 12 missed frames."
    )

    webrtc_context = webrtc_streamer(
        key="cricvision-live-cricket-analysis",
        video_processor_factory=processor_class,
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

    if webrtc_context.video_processor:
        webrtc_context.video_processor.confidence = confidence
        webrtc_context.video_processor.image_size = image_size
