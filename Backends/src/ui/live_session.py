import streamlit as st
import cv2
import av
import math
from pathlib import Path
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration


MODEL_PATH = Path("Models/ball_detector/best.pt")


@st.cache_resource
def load_ball_model():
    if not MODEL_PATH.exists():
        return None
    return YOLO(str(MODEL_PATH))


class BallDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_ball_model()

        # Lower confidence helps detect fast/blurry balls.
        # If you get too many false detections, increase this to 0.30 or 0.35.
        self.confidence = 0.25

        # Stores recent ball center points for trajectory.
        self.trail_points = []
        self.max_trail_points = 30

        # Stores last valid ball position for jump filtering.
        self.last_center = None

        # Tracking stats
        self.total_frames = 0
        self.detected_frames = 0
        self.missed_frames = 0

    def is_valid_ball_box(self, x1, y1, x2, y2, frame_width, frame_height):
        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height
        frame_area = frame_width * frame_height

        # Cricket ball should not take too much area in umpire-view camera.
        # This removes false detections on shoes, body, stumps, etc.
        min_area_ratio = 0.00002
        max_area_ratio = 0.04

        area_ratio = box_area / frame_area

        if area_ratio < min_area_ratio:
            return False

        if area_ratio > max_area_ratio:
            return False

        # Ball box should be roughly square-ish.
        aspect_ratio = box_width / max(box_height, 1)

        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            return False

        return True

    def is_reasonable_movement(self, center_x, center_y):
        if self.last_center is None:
            return True

        last_x, last_y = self.last_center

        distance = math.sqrt((center_x - last_x) ** 2 + (center_y - last_y) ** 2)

        # If the detection jumps too far suddenly, it may be a false detection.
        # You can increase this if your ball moves very fast on screen.
        max_jump_distance = 250

        return distance <= max_jump_distance

    def draw_trajectory(self, frame):
        if len(self.trail_points) < 2:
            return frame

        for i in range(1, len(self.trail_points)):
            thickness = max(1, int(i / 8))

            cv2.line(
                frame,
                self.trail_points[i - 1],
                self.trail_points[i],
                (0, 255, 255),
                thickness,
            )

        return frame

    def draw_stats_panel(self, frame, detected, confidence=None, center=None):
        panel_x = 20
        panel_y = 25
        line_gap = 28

        status_text = "Ball Detected: YES" if detected else "Ball Detected: NO"
        status_color = (0, 255, 0) if detected else (0, 0, 255)

        cv2.putText(
            frame,
            status_text,
            (panel_x, panel_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
        )

        if confidence is not None:
            cv2.putText(
                frame,
                f"Confidence: {confidence:.2f}",
                (panel_x, panel_y + line_gap),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

        if center is not None:
            center_x, center_y = center
            cv2.putText(
                frame,
                f"Center: ({center_x}, {center_y})",
                (panel_x, panel_y + line_gap * 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

        cv2.putText(
            frame,
            f"Tracked Frames: {self.detected_frames}",
            (panel_x, panel_y + line_gap * 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        return frame

    def recv(self, frame):
        self.total_frames += 1

        img = frame.to_ndarray(format="bgr24")
        frame_height, frame_width = img.shape[:2]

        if self.model is None:
            cv2.putText(
                img,
                "Model not found: Models/ball_detector/best.pt",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        results = self.model.predict(
            source=img,
            conf=self.confidence,
            imgsz=640,
            verbose=False,
        )

        result = results[0]

        # Start with normal YOLO box drawing
        annotated_frame = result.plot()

        detected = False
        best_confidence = None
        best_center = None

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes

            valid_detections = []

            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())

                x1, y1, x2, y2 = box

                if not self.is_valid_ball_box(x1, y1, x2, y2, frame_width, frame_height):
                    continue

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                if not self.is_reasonable_movement(center_x, center_y):
                    continue

                valid_detections.append(
                    {
                        "box": box,
                        "confidence": conf,
                        "center": (center_x, center_y),
                    }
                )

            if valid_detections:
                # Choose highest-confidence valid detection
                best_detection = max(valid_detections, key=lambda d: d["confidence"])

                best_confidence = best_detection["confidence"]
                best_center = best_detection["center"]

                center_x, center_y = best_center

                self.last_center = best_center
                self.trail_points.append(best_center)

                if len(self.trail_points) > self.max_trail_points:
                    self.trail_points.pop(0)

                self.detected_frames += 1
                detected = True

                # Draw center point
                cv2.circle(
                    annotated_frame,
                    (center_x, center_y),
                    7,
                    (0, 255, 255),
                    -1,
                )

                # Draw outer circle around center
                cv2.circle(
                    annotated_frame,
                    (center_x, center_y),
                    14,
                    (0, 255, 255),
                    2,
                )

            else:
                self.missed_frames += 1
        else:
            self.missed_frames += 1

        # Draw trajectory trail
        annotated_frame = self.draw_trajectory(annotated_frame)

        # Draw live stats
        annotated_frame = self.draw_stats_panel(
            annotated_frame,
            detected=detected,
            confidence=best_confidence,
            center=best_center,
        )

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


def show_live_session_page():
    st.title("🏏 Live Cricket Ball Tracking")
    st.markdown("Umpire-view camera with YOLO cricket ball detection and trajectory tracking.")

    if not MODEL_PATH.exists():
        st.error("Model not found. Put your trained model here: Models/ball_detector/best.pt")
        st.stop()

    st.success("YOLO cricket ball model loaded successfully.")

    st.info(
        "Place the camera behind the bowler, facing the batter, like an umpire-view angle."
    )

    st.markdown(
        """
        **Current features:**
        - Cricket ball detection
        - Center point tracking
        - Trajectory trail
        - False detection filtering
        - Live detection stats
        """
    )

    rtc_config = RTCConfiguration(
        {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        }
    )

    webrtc_streamer(
        key="cricvision-live-ball-tracking",
        video_processor_factory=BallDetectionProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )