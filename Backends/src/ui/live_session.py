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
    generate_wagon_wheel_data,
    find_nearest_fielder,
    normalize_handedness,
    save_field_analysis_history,
    save_field_setup,
    suggest_field_adjustment,
)
from Backends.src.config.constants import (
    DETECTION_PRESETS,
    ENSEMBLE_MODEL_NAME,
    LOW_CONFIDENCE_REVIEW_THRESHOLD,
)
from Backends.src.config.paths import (
    CRICKET_OBJECTS_MODEL_PATH,
    OUTPUTS_DIR,
    REVIEW_FRAMES_DIR,
    VIDEO_ANALYSIS_OUTPUT_DIR,
)
from Backends.src.live_delivery_capture import (
    create_delivery_capture_state,
    save_delivery_clip,
    update_delivery_capture_state,
)
from Backends.src.live_stump_validator import (
    validate_stumps_in_alignment_boxes,
)
from Backends.src.session_calibration import (
    build_premium_stump_alignment_boxes,
    capture_calibration_snapshot,
    solve_stump_calibration_from_snapshot,
)
from Backends.src.virtual_pitch_overlay import (
    draw_alignment_boxes,
    draw_environment_preview_overlay,
    draw_setup_complete_overlay,
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
from Backends.src.models.model_registry import get_model_info, get_model_path
from Backends.src.ui.analysis_helpers import (
    ensure_delivery_report_fields,
    persist_result_to_session as _persist_result_to_session,
)
from Backends.src.video_pipeline.annotation_writer import (
    convert_to_browser_mp4,
    draw_label,
    draw_pitch_roi,
    draw_search_roi,
    save_review_frame,
)
from Backends.src.video_pipeline.detection_pipeline import (
    estimate_length_from_bounce,
    estimate_line_from_stumps,
    get_available_ensemble_model_names,
    get_nearest_stump_detections,
    has_enough_ball_movement,
    load_detection_model,
    load_ensemble_models,
    map_model_classes,
    run_local_redetection,
    run_pitch_roi_detection,
)
from Backends.src.video_pipeline.report_pipeline import build_video_reports
from Backends.src.video_pipeline.video_reader import write_video_frames


MIN_TRAJECTORY_POINTS_FOR_BOUNCE = 8
MIN_MOVEMENT_DISTANCE = 40
SHORT_MISSING_BALL_SMOOTHING_FRAMES = 8
MAX_MISSING_BALL_FRAMES = 12
MAX_TRAJECTORY_POINTS = 35
MAX_RECORDED_FRAMES = 450
DEFAULT_RECORDING_FPS = 25
LIVE_DELIVERIES_DIR = OUTPUTS_DIR / "live_deliveries"
DEFAULT_LIVE_FRAME_WIDTH = 1280
DEFAULT_LIVE_FRAME_HEIGHT = 720
# ponytail: sparse live YOLO only — motion capture stays primary; never every frame.
LIVE_DETECT_EVERY_N = 5
LIVE_STAGES = {
    "setup",
    "align_stumps",
    "calibration_solving",
    "setup_complete",
    "live_capture",
    "session_summary",
}
LIVE_STAGE_ALIASES = {
    "camera_calibration": "align_stumps",
    "calibrated_ready": "setup_complete",
    "calibration_locked": "setup_complete",
    "delivery_capture": "live_capture",
    "session_results": "session_summary",
}


def _normalize_live_stage(stage):
    if stage in LIVE_STAGE_ALIASES:
        return LIVE_STAGE_ALIASES[stage]
    return stage if stage in LIVE_STAGES else "setup"


def _get_live_stage():
    stage = st.session_state.get("live_stage")
    if not stage:
        legacy = st.session_state.get("live_session_stage")
        stage = _normalize_live_stage(legacy or "setup")
        st.session_state.live_stage = stage
    return _normalize_live_stage(stage)


class LiveSessionBridge:
    """Shared state between Streamlit UI and the webrtc video processor thread."""

    def __init__(self):
        self.lock = Lock()
        self.stage = "setup"
        self.box_layout = None
        self.calibration = None
        self.show_alignment_boxes = False
        self.show_calibrated_geometry = False
        self.live_session_active = False
        self.capture_state = create_delivery_capture_state()
        self.last_saved_path = None
        self.saved_clip_paths = []
        self.status_message = "Waiting for delivery..."
        self.frame_size = (DEFAULT_LIVE_FRAME_WIDTH, DEFAULT_LIVE_FRAME_HEIGHT)
        # ponytail: detector warmed on Streamlit thread; webrtc only reuses the handle.
        self.detector_model = None
        self.detector_available = None
        self.ball_confidence = 0.25
        self.detect_frame_index = 0
        self.last_ball_point = None
        self.stump_validation = None
        self.stump_validation_history = None
        self.show_pitch_axis_preview = False
        self.last_frame = None

    def get_last_frame(self):
        with self.lock:
            if self.last_frame is None:
                return None
            return self.last_frame.copy()

    def configure(
        self,
        *,
        stage="setup",
        box_layout=None,
        calibration=None,
        show_alignment_boxes=False,
        show_calibrated_geometry=False,
        live_session_active=False,
        detector_model=None,
        ball_confidence=None,
        detector_available=None,
        show_pitch_axis_preview=None,
    ):
        with self.lock:
            self.stage = _normalize_live_stage(stage or "setup")
            self.box_layout = box_layout
            self.calibration = calibration
            self.show_alignment_boxes = bool(show_alignment_boxes)
            self.show_calibrated_geometry = bool(show_calibrated_geometry)
            self.live_session_active = bool(live_session_active)
            if detector_model is not None:
                self.detector_model = detector_model
            if detector_available is not None:
                self.detector_available = bool(detector_available)
            if ball_confidence is not None:
                try:
                    self.ball_confidence = float(ball_confidence)
                except (TypeError, ValueError):
                    pass
            if show_pitch_axis_preview is not None:
                self.show_pitch_axis_preview = bool(show_pitch_axis_preview)

    def reset_capture(self):
        with self.lock:
            self.capture_state = create_delivery_capture_state()
            self.last_saved_path = None
            self.saved_clip_paths = []
            self.status_message = "Waiting for delivery..."
            self.detect_frame_index = 0
            self.last_ball_point = None

    def record_saved_clip(self, path):
        """Thread-safe note that a delivery clip was written to disk."""
        if not path:
            return
        path_str = str(path)
        with self.lock:
            self.last_saved_path = path_str
            if path_str not in self.saved_clip_paths:
                self.saved_clip_paths.append(path_str)

    def snapshot(self):
        with self.lock:
            return {
                "recording": bool(self.capture_state.get("recording")),
                "delivery_count": int(self.capture_state.get("delivery_count") or 0),
                "last_saved_path": self.last_saved_path,
                "saved_clip_paths": list(self.saved_clip_paths),
                "status_message": self.status_message,
                "live_session_active": self.live_session_active,
                "frame_size": self.frame_size,
                "stage": self.stage,
                "last_ball_point": self.last_ball_point,
                "stump_validation": self.stump_validation,
                "stump_validation_history": self.stump_validation_history,
                "detector_available": self.detector_available,
                "has_frame": self.last_frame is not None,
            }


def get_live_model_options():
    """Return user-facing detection model choices for Live Session."""
    options = {
        "Ball + Stump Detector": {
            "path": CRICKET_OBJECTS_MODEL_PATH,
            "model_key": "current_best",
            "ensemble": False,
        },
    }
    if len(get_available_ensemble_model_names()) >= 2:
        options[ENSEMBLE_MODEL_NAME] = {
            "path": None,
            "model_key": None,
            "ensemble": True,
        }
    return options


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


def create_delivery_recorder_class(recording_state, live_bridge=None):
    import av
    from streamlit_webrtc import VideoProcessorBase

    class DeliveryRecorder(VideoProcessorBase):
        def __init__(self):
            self.recording_state = recording_state
            self.live_bridge = live_bridge

        def recv(self, frame):
            image = frame.to_ndarray(format="bgr24")
            self.recording_state.append_frame(image)

            display = image
            bridge = self.live_bridge
            if bridge is not None:
                with bridge.lock:
                    bridge.last_frame = image.copy()
                    frame_w = int(image.shape[1])
                    frame_h = int(image.shape[0])
                    bridge.frame_size = (frame_w, frame_h)
                    stage = bridge.stage
                    box_layout = bridge.box_layout
                    calibration = bridge.calibration
                    show_boxes = bridge.show_alignment_boxes
                    show_geometry = bridge.show_calibrated_geometry
                    session_active = bridge.live_session_active
                    capture_state = bridge.capture_state
                    detector_model = bridge.detector_model
                    ball_confidence = bridge.ball_confidence
                    detect_frame_index = bridge.detect_frame_index
                    stump_validation = bridge.stump_validation
                    show_pitch_axis = bridge.show_pitch_axis_preview

                # ponytail: stage is the source of truth — flags alone can be stale across webrtc frames.
                show_geometry = bool(show_geometry) and stage in {
                    "setup_complete",
                    "live_capture",
                }
                show_boxes = bool(show_boxes) or stage in {
                    "align_stumps",
                    "setup_complete",
                    "live_capture",
                }

                # Rebuild boxes for the real camera size only while aligning.
                if stage == "align_stumps" and show_boxes:
                    live_layout = build_premium_stump_alignment_boxes((frame_w, frame_h))
                    if live_layout.get("available"):
                        box_layout = live_layout
                        with bridge.lock:
                            bridge.box_layout = live_layout

                    # ponytail: optional status-only detections — Continue is never gated here.
                    run_stump_detect = (
                        detector_model is not None
                        and (detect_frame_index % LIVE_DETECT_EVERY_N) == 0
                    )
                    with bridge.lock:
                        bridge.detect_frame_index = detect_frame_index + 1
                    if run_stump_detect:
                        stump_detections = _sparse_live_stump_detections(
                            detector_model,
                            image,
                            confidence=ball_confidence,
                        )
                        stump_validation = validate_stumps_in_alignment_boxes(
                            stump_detections,
                            box_layout,
                            frame_size=(frame_w, frame_h),
                        )
                        with bridge.lock:
                            bridge.stump_validation = stump_validation

                # ponytail: motion capture primary; optional sparse YOLO hint if model already warm.
                if session_active and stage == "live_capture":
                    detections = None
                    ball_point = None
                    run_detect = (
                        detector_model is not None
                        and (detect_frame_index % LIVE_DETECT_EVERY_N) == 0
                    )
                    with bridge.lock:
                        bridge.detect_frame_index = detect_frame_index + 1
                    if run_detect:
                        detections, ball_point = _sparse_live_ball_detections(
                            detector_model,
                            image,
                            confidence=ball_confidence,
                        )
                        with bridge.lock:
                            bridge.last_ball_point = ball_point
                    else:
                        with bridge.lock:
                            ball_point = bridge.last_ball_point

                    update = update_delivery_capture_state(
                        image,
                        detections=detections,
                        calibration=(
                            calibration
                            if isinstance(calibration, dict) and calibration.get("available")
                            else None
                        ),
                        state=capture_state,
                    )
                    with bridge.lock:
                        bridge.capture_state = update["state"]
                        if update.get("delivery_detected"):
                            bridge.status_message = "Delivery detected"
                        elif update.get("recording"):
                            bridge.status_message = "Recording delivery..."
                        elif int(update["state"].get("cooldown_frames") or 0) > 0:
                            bridge.status_message = "Continue bowling"
                        else:
                            bridge.status_message = "Waiting for delivery..."
                        completed = update.get("completed_clip")
                        if completed:
                            save_result = save_delivery_clip(
                                completed,
                                output_dir=LIVE_DELIVERIES_DIR,
                                fps=DEFAULT_RECORDING_FPS,
                                prefix="delivery",
                            )
                            if save_result.get("saved"):
                                path_str = str(save_result.get("path"))
                                bridge.last_saved_path = path_str
                                if path_str not in bridge.saved_clip_paths:
                                    bridge.saved_clip_paths.append(path_str)
                                bridge.status_message = (
                                    "Clip saved — analysing next..."
                                )
                            else:
                                bridge.status_message = "Continue bowling"

                    if ball_point is not None:
                        if display is image:
                            display = image.copy()
                        try:
                            cv2.circle(
                                display,
                                (int(ball_point[0]), int(ball_point[1])),
                                8,
                                (255, 180, 0),
                                2,
                            )
                        except Exception:
                            pass

                # ponytail: stage picks the drawer — alignment boxes only before solve.
                if stage == "align_stumps" and show_boxes:
                    display = image.copy() if display is image else display
                    draw_alignment_boxes(display, box_layout, validation_result=stump_validation)
                elif stage == "setup_complete" and show_geometry:
                    if display is image:
                        display = image.copy()
                    env_context = None
                    if isinstance(calibration, dict):
                        env_context = calibration.get("environment_context")
                    draw_setup_complete_overlay(
                        display,
                        calibration_result=calibration,
                        environment_context=env_context,
                    )
                elif stage == "live_capture" and show_geometry:
                    if display is image:
                        display = image.copy()
                    env_context = None
                    if isinstance(calibration, dict):
                        env_context = calibration.get("environment_context")
                    show_axis = bool(stage == "setup_complete" and show_pitch_axis)
                    draw_environment_preview_overlay(
                        display,
                        env_context,
                        calibration=calibration if show_geometry else None,
                        show_pitch_axis=show_axis,
                    )

            return av.VideoFrame.from_ndarray(display, format="bgr24")

    return DeliveryRecorder


def _sparse_live_stump_detections(model, frame, confidence=0.25):
    """Sparse stump detections for live calibration validation."""
    from Backends.src.stump_detection import resolve_live_stump_detections

    return resolve_live_stump_detections(
        frame,
        primary_model=model,
        conf=confidence,
    )


def _sparse_live_ball_detections(model, frame, confidence=0.25):
    """Optional light ball hint for capture trigger/overlay. Never invents points."""
    if model is None or frame is None:
        return None, None
    try:
        class_names = map_model_classes(model)
        conf = float(confidence) if confidence is not None else 0.25
        results = model.predict(frame, conf=conf, verbose=False, imgsz=640)
        if not results:
            return None, None
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None, None
        ball_detections = []
        for box in result.boxes:
            class_id = int(box.cls[0].cpu().numpy())
            class_name = class_names.get(class_id)
            if class_name != "ball":
                continue
            score = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            ball_detections.append(
                {
                    "class_name": "ball",
                    "confidence": score,
                    "box": (x1, y1, x2, y2),
                    "center": center,
                }
            )
        if not ball_detections:
            return None, None
        best = max(ball_detections, key=lambda item: item["confidence"])
        return {"ball_detections": ball_detections}, best["center"]
    except Exception:
        return None, None


def warm_live_detector():
    """Load current_best on the Streamlit thread for optional sparse live hints."""
    try:
        return get_cached_yolo_model("current_best")
    except Exception:
        return None


def analyse_saved_live_clip(
    clip_path,
    *,
    model_path=None,
    model_key="current_best",
    confidence=0.25,
    imgsz=640,
    use_ensemble=False,
    show_pitch_roi=False,
    field_setup=None,
    session_calibration=None,
    model_name=None,
    preset_name=None,
):
    """Run the existing Video Analysis pipeline on one saved live clip.

    Defensive: failures return success=False and never raise into the live UI.
    """
    from Backends.src.ui.video_analysis import process_video

    clip = Path(clip_path) if clip_path else None
    if clip is None or not clip.is_file():
        return {
            "success": False,
            "error": f"Saved clip not found: {clip_path}",
            "source_clip_path": str(clip_path) if clip_path else None,
        }

    VIDEO_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_output = VIDEO_ANALYSIS_OUTPUT_DIR / f"live_clip_{stamp}_raw.mp4"
    browser_output = VIDEO_ANALYSIS_OUTPUT_DIR / f"live_clip_{stamp}.mp4"
    calibration = (
        session_calibration
        if isinstance(session_calibration, dict)
        else {}
    )

    try:
        result = process_video(
            video_path=clip,
            output_path=raw_output,
            model_path=Path(model_path) if model_path else CRICKET_OBJECTS_MODEL_PATH,
            model_key=model_key,
            confidence=confidence,
            imgsz=imgsz,
            use_ensemble=use_ensemble,
            show_pitch_roi=show_pitch_roi,
            field_setup=field_setup,
            # ponytail: Smart Balanced keeps live post-clip analysis usable on short clips.
            speed_mode="Smart Balanced",
            generate_processed_video=True,
            overlay_detail="Clean",
        )
    except Exception as error:
        return {
            "success": False,
            "error": f"Live clip analysis failed: {type(error).__name__}: {error}",
            "source_clip_path": str(clip),
            "session_calibration": calibration,
        }

    if not isinstance(result, dict):
        return {
            "success": False,
            "error": "Live clip analysis returned an empty result.",
            "source_clip_path": str(clip),
            "session_calibration": calibration,
        }

    result["source_clip_path"] = str(clip)
    result["session_calibration"] = calibration
    result["active_model"] = model_name or result.get("active_model") or "Ball + Stump Detector"
    result["active_preset"] = preset_name or result.get("active_preset") or "Balanced Mode"
    result["analysis_source"] = "live_delivery_clip"

    if result.get("success") and result.get("processed_video_generated") and raw_output.exists():
        result["raw_output_path"] = str(raw_output)
        try:
            final_path = convert_to_browser_mp4(
                input_path=raw_output,
                output_path=browser_output,
            )
            result["output_path"] = str(final_path)
            result["processed_video_conversion"] = "converted"
        except Exception as conv_error:
            result["output_path"] = str(raw_output)
            result["processed_video_conversion"] = "failed"
            result["processed_video_conversion_error"] = str(conv_error)

    if not result.get("success"):
        result.setdefault("source_clip_path", str(clip))
        result.setdefault("session_calibration", calibration)

    return result


def render_live_delivery_result_panel(result, *, expanded_details=False):
    """Compact last-delivery summary — not the full Video Analysis page."""
    if not result:
        st.caption("No delivery analysis yet.")
        return

    source_clip = result.get("source_clip_path")
    if source_clip:
        st.caption(f"Saved clip: `{source_clip}`")
        if Path(source_clip).is_file():
            try:
                st.video(str(source_clip))
            except Exception:
                pass

    if not result.get("success"):
        st.error(result.get("error") or "Delivery analysis failed. Clip was kept.")
        return

    cols = st.columns(4)
    cols[0].metric("Line", result.get("estimated_line") or "Unknown")
    cols[1].metric("Length", result.get("estimated_length") or "Unknown")
    cols[2].metric(
        "Ball detect",
        f"{float(result.get('ball_detection_rate') or 0):.0f}%",
    )
    cols[3].metric(
        "Tracking",
        result.get("overall_tracking_quality")
        or result.get("tracking_quality")
        or "—",
    )

    path_source = "—"
    try:
        from Backends.src.ui.video_analysis import (
            physics_path_source,
            prepare_result_path_validity,
        )

        physics = result.get("physics_trajectory") or {}
        validity = prepare_result_path_validity(result)
        path_source = physics_path_source(physics, validity) or "—"
    except Exception:
        physics = result.get("physics_trajectory") or {}
        path_source = physics.get("input_path_source") or path_source

    badge_cols = st.columns(3)
    badge_cols[0].caption(f"Path source: {path_source}")
    badge_cols[1].caption(
        f"Calibration: {result.get('calibration_status') or 'Not calibrated'}"
    )
    quality = (result.get("session_calibration") or {}).get("quality")
    if quality:
        badge_cols[2].caption(f"Session cal: {quality}")

    processed = result.get("output_path")
    if processed and Path(processed).is_file():
        st.markdown("**Processed replay**")
        try:
            st.video(str(processed))
        except Exception:
            st.caption(f"Processed video: `{processed}`")
    else:
        st.info("Replay / trajectory available after processing when detections are strong enough.")

    with st.expander("Trajectory / physics details", expanded=expanded_details):
        try:
            from Backends.src.ui.video_analysis import render_trajectory_replay_section

            render_trajectory_replay_section(result)
        except Exception as error:
            st.caption(f"Trajectory replay unavailable: {error}")
        physics = result.get("physics_trajectory") or {}
        if physics:
            st.json(
                {
                    "path_source": path_source,
                    "bounce": physics.get("bounce_point") or result.get("estimated_bounce_point"),
                    "fit_quality": physics.get("fit_quality")
                    or result.get("trajectory_fit_quality"),
                    "notes": physics.get("notes") or [],
                }
            )

    with st.expander("Full delivery report", expanded=False):
        show_analysis_output(result)


def _ensure_alignment_box_layout(frame_width, frame_height):
    """Build or refresh fixed FullTrack-style boxes for the current frame size."""
    layout = st.session_state.get("live_box_layout") or st.session_state.get("live_alignment_box_layout")
    size_key = (int(frame_width), int(frame_height))
    cached_size = st.session_state.get("live_alignment_frame_size")
    if (
        isinstance(layout, dict)
        and layout.get("available")
        and cached_size == size_key
    ):
        return layout

    layout = build_premium_stump_alignment_boxes(size_key)
    st.session_state.live_box_layout = layout
    st.session_state.live_alignment_box_layout = layout
    st.session_state.live_alignment_frame_size = size_key
    return layout


def render_live_camera_setup_guidelines():
    st.markdown(
        """
- Use a tripod
- Keep 6 stumps visible
- Camera behind non-striker stumps
- Not too far back; higher is better
- Do not block the camera
"""
    )


def _set_live_session_stage(stage):
    normalized = _normalize_live_stage(stage)
    st.session_state.live_stage = normalized
    st.session_state.live_session_stage = normalized


def render_live_stage_setup(live_bridge):
    """Stage: setup — hero card + requirement chips + one start button."""
    from Backends.src.ui.theme import render_status_row

    st.markdown(
        """
        <div class="cv-hero">
            <h1>Live Bowling Session</h1>
            <p>Calibrate the stumps, then capture deliveries live.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_status_row(
        [
            ("Tripod", "default"),
            ("6 stumps", "default"),
            ("Behind non-striker", "default"),
            ("Good lighting", "default"),
            ("Don't block camera", "warning"),
        ]
    )
    if st.button("Start Live Delivery Analysis", type="primary", use_container_width=True):
        live_bridge.reset_capture()
        st.session_state.live_camera_active = True
        st.session_state.live_camera_session_ended = False
        st.session_state.live_box_layout = None
        st.session_state.live_alignment_box_layout = None
        st.session_state.live_alignment_report = None
        st.session_state.live_calibration_payload = None
        st.session_state.live_stump_validation = None
        st.session_state.live_stump_validation_history = None
        st.session_state.live_environment_context = None
        st.session_state.live_calibration_result = None
        st.session_state.live_calibration_snapshot = None
        st.session_state.live_calibration_snapshot_frame = None
        st.session_state.live_calibration_locked = False
        st.session_state.live_capture_state = create_delivery_capture_state()
        st.session_state.live_auto_session_active = False
        with live_bridge.lock:
            live_bridge.stump_validation = None
            live_bridge.stump_validation_history = None
        _set_live_session_stage("align_stumps")
        st.rerun()
    with st.expander("Advanced settings", expanded=False):
        st.caption("Field setup and detector options appear after calibration.")


def render_live_calibration_instructions():
    """Streamlit instruction card above the camera (not drawn inside the video)."""
    st.info(
        "**Fit both stump sets inside the boxes, then press Continue.**  \n"
        "Move the camera/tripod or adjust zoom if needed."
    )


def render_live_calibration_status_panel(validation, detector_available):
    """Optional stump search status during align_stumps — does not gate Continue."""
    from Backends.src.ui.theme import render_status_row

    validation = validation if isinstance(validation, dict) else {}
    striker_found = bool((validation.get("striker") or {}).get("found"))
    non_striker_found = bool((validation.get("non_striker") or {}).get("found"))

    if detector_available is False:
        st.caption("Stump detector unavailable — validation runs when you press Continue.")
    elif detector_available is None:
        st.caption("Warming stump detector for live calibration...")

    striker_tone = "success" if striker_found else ("warning" if detector_available else "default")
    non_tone = "success" if non_striker_found else ("warning" if detector_available else "default")
    if striker_found and non_striker_found:
        setup_tone = "success"
        setup_label = "Both Found (preview)"
    elif striker_found or non_striker_found:
        setup_tone = "warning"
        setup_label = "Partial (preview)"
    else:
        setup_tone = "default"
        setup_label = "Searching (preview)"

    render_status_row(
        [
            (f"Striker {'Found' if striker_found else 'Searching'}", striker_tone),
            (f"Non-Striker {'Found' if non_striker_found else 'Searching'}", non_tone),
            (setup_label, setup_tone),
        ]
    )
    return validation


def _apply_calibration_solve_result(solve_result, live_layout, live_w, live_h, validation):
    """Store solved calibration into session state for setup_complete."""
    calibration = solve_result.get("calibration")
    if not isinstance(calibration, dict) or not calibration:
        calibration = {
            "available": bool(solve_result.get("available")),
            "quality": solve_result.get("quality"),
            "striker_stumps_box": solve_result.get("striker_stumps_box"),
            "non_striker_stumps_box": solve_result.get("non_striker_stumps_box"),
            "stump_line": solve_result.get("stump_line") or solve_result.get("pitch_axis"),
            "pitch_corridor": solve_result.get("pitch_corridor"),
            "stumps_validated": bool(solve_result.get("success")),
            "notes": list(solve_result.get("notes") or []),
        }
    if solve_result.get("environment_context"):
        calibration["environment_context"] = solve_result["environment_context"]
    if solve_result.get("virtual_stumps"):
        calibration["virtual_stumps"] = solve_result["virtual_stumps"]
    report = {
        "available": bool(calibration.get("available")),
        "quality": calibration.get("quality"),
        "stump_line_available": bool(calibration.get("stump_line")),
        "pitch_corridor_available": bool(calibration.get("pitch_corridor")),
        "calibration": calibration,
        "notes": list(calibration.get("notes") or []),
    }
    st.session_state.live_box_layout = live_layout
    st.session_state.live_alignment_box_layout = live_layout
    st.session_state.live_alignment_frame_size = (live_w, live_h)
    st.session_state.live_stump_validation = validation
    st.session_state.live_environment_context = solve_result.get("environment_context")
    st.session_state.live_calibration_result = solve_result
    st.session_state.live_calibration_payload = calibration
    st.session_state.live_alignment_report = report
    st.session_state.live_calibration_snapshot = None
    st.session_state.live_calibration_snapshot_frame = None
    st.session_state.live_calibration_locked = True


def render_live_stage_camera_calibration_actions(
    live_bridge,
    frame_width,
    frame_height,
    layout,
    validation=None,
    detector_available=None,
    detector_model=None,
    ball_confidence=0.25,
):
    """Continue / Cancel below the camera during align_stumps."""
    override = bool(st.session_state.get("live_calibration_stump_override", False))
    validation = validation if isinstance(validation, dict) else {}

    cols = st.columns(2)
    if cols[0].button(
        "Continue",
        type="primary",
        use_container_width=True,
    ):
        frame = live_bridge.get_last_frame()
        if frame is None:
            st.warning("Camera frame not ready yet. Allow camera access, then try again.")
            return

        live_w, live_h = int(frame.shape[1]), int(frame.shape[0])
        with live_bridge.lock:
            live_layout = live_bridge.box_layout
        if not isinstance(live_layout, dict) or not live_layout.get("available"):
            live_layout = build_premium_stump_alignment_boxes((live_w, live_h))
        if not live_layout.get("available"):
            live_layout = layout

        snapshot = capture_calibration_snapshot(frame, live_layout)
        if not snapshot.get("available"):
            st.warning("Could not capture calibration snapshot. Try again.")
            return

        st.session_state.live_calibration_snapshot = snapshot
        st.session_state.live_calibration_snapshot_frame = frame
        _set_live_session_stage("calibration_solving")

        model = detector_model
        if model is None:
            model = warm_live_detector()
        detections = []
        if model is not None:
            detections = _sparse_live_stump_detections(
                model,
                frame,
                confidence=ball_confidence,
            )
        elif not override:
            st.warning("Stump detector unavailable. Enable developer override or wait for detector.")
            _set_live_session_stage("align_stumps")
            return

        with st.spinner("Detecting stumps on snapshot..."):
            solve_result = solve_stump_calibration_from_snapshot(
                frame,
                live_layout,
                detections=detections,
                frame_size=(live_w, live_h),
                dev_override=override,
            )
            snapshot_validation = solve_result.get("validation") or validation

        if not solve_result.get("success"):
            st.warning("Stumps not detected in both boxes. Align both stump sets and try again.")
            _set_live_session_stage("align_stumps")
            st.rerun()

        _apply_calibration_solve_result(
            solve_result,
            live_layout,
            live_w,
            live_h,
            snapshot_validation,
        )
        _set_live_session_stage("setup_complete")
        st.rerun()

    if cols[1].button("Cancel / Back", use_container_width=True):
        live_bridge.reset_capture()
        st.session_state.live_auto_session_active = False
        st.session_state.live_alignment_report = None
        st.session_state.live_calibration_payload = None
        st.session_state.live_stump_validation = None
        st.session_state.live_stump_validation_history = None
        st.session_state.live_environment_context = None
        st.session_state.live_calibration_result = None
        st.session_state.live_calibration_snapshot = None
        st.session_state.live_calibration_snapshot_frame = None
        st.session_state.live_calibration_locked = False
        with live_bridge.lock:
            live_bridge.stump_validation = None
            live_bridge.stump_validation_history = None
        _set_live_session_stage("setup")
        st.rerun()


def render_live_stage_setup_complete_panel(live_bridge):
    """Streamlit status below camera after solve — setup_complete stage."""
    from Backends.src.ui.theme import render_status_pill, render_status_row

    report = st.session_state.get("live_alignment_report") or {}
    calibration = (
        st.session_state.get("live_calibration_payload")
        or st.session_state.get("live_calibration_result")
        or report.get("calibration")
        or {}
    )
    if isinstance(calibration, dict) and "calibration" in calibration:
        calibration = calibration.get("calibration") or calibration
    validation = st.session_state.get("live_stump_validation") or {}
    env = st.session_state.get("live_environment_context") or calibration.get("environment_context") or {}

    st.markdown(
        """
        <div class="cv-step-card">
            <p class="cv-step-label">Setup Complete</p>
            <p class="cv-step-title">Press Redetect if the pitch/stumps were not detected correctly.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    quality = calibration.get("quality") or validation.get("quality") or "Unavailable"
    tone = "success" if quality in {"Strong", "Found", "Good"} else "warning"
    if quality == "Manual Override / Low":
        tone = "error"
    st.markdown(render_status_pill(f"Quality: {quality}", tone), unsafe_allow_html=True)

    striker_found = bool(
        env.get("striker_stumps_found")
        or (validation.get("striker") or {}).get("found")
        or calibration.get("stumps_validated")
    )
    non_striker_found = bool(
        env.get("non_striker_stumps_found")
        or (validation.get("non_striker") or {}).get("found")
        or calibration.get("stumps_validated")
    )
    render_status_row(
        [
            (f"Striker {'Found' if striker_found else 'Not Found'}", "success" if striker_found else "error"),
            (f"Non-Striker {'Found' if non_striker_found else 'Not Found'}", "success" if non_striker_found else "error"),
        ]
    )
    if calibration.get("available"):
        st.caption("Estimated pitch context preview — not official LBW/DRS.")
    else:
        st.caption("Calibration geometry limited — use Redetect if stumps look wrong.")

    with st.expander("Developer / Advanced", expanded=False):
        st.checkbox(
            "Show pitch axis preview overlay",
            value=False,
            key="live_show_pitch_axis_preview",
        )

    action_cols = st.columns(3)
    if action_cols[0].button("Start Capture", type="primary", use_container_width=True):
        live_bridge.reset_capture()
        detector = warm_live_detector()
        with live_bridge.lock:
            live_bridge.detector_model = detector
        st.session_state.live_auto_session_active = True
        st.session_state.live_capture_state = create_delivery_capture_state()
        st.session_state.live_delivery_log = []
        st.session_state.live_analysed_clip_paths = []
        st.session_state.live_last_result = None
        st.session_state.live_result_by_clip = {}
        st.session_state.live_pending_clip_path = None
        _set_live_session_stage("live_capture")
        st.session_state.live_status_message = "Waiting for delivery..."
        st.rerun()
    if action_cols[1].button("Redetect", use_container_width=True):
        st.session_state.live_auto_session_active = False
        st.session_state.live_alignment_report = None
        st.session_state.live_calibration_payload = None
        st.session_state.live_stump_validation = None
        st.session_state.live_stump_validation_history = None
        st.session_state.live_environment_context = None
        st.session_state.live_calibration_result = None
        st.session_state.live_calibration_snapshot = None
        st.session_state.live_calibration_snapshot_frame = None
        st.session_state.live_calibration_locked = False
        with live_bridge.lock:
            live_bridge.stump_validation = None
            live_bridge.stump_validation_history = None
            live_bridge.detect_frame_index = 0
        _set_live_session_stage("align_stumps")
        st.rerun()
    if action_cols[2].button("Cancel / Back", use_container_width=True):
        live_bridge.reset_capture()
        st.session_state.live_auto_session_active = False
        st.session_state.live_alignment_report = None
        st.session_state.live_calibration_payload = None
        st.session_state.live_stump_validation = None
        st.session_state.live_stump_validation_history = None
        st.session_state.live_environment_context = None
        st.session_state.live_calibration_result = None
        st.session_state.live_calibration_snapshot = None
        st.session_state.live_calibration_snapshot_frame = None
        st.session_state.live_calibration_locked = False
        with live_bridge.lock:
            live_bridge.stump_validation = None
            live_bridge.stump_validation_history = None
        _set_live_session_stage("setup")
        st.rerun()
    return calibration


def render_live_stage_calibrated_ready_panel(live_bridge):
    """Backward-compat alias for setup_complete panel."""
    return render_live_stage_setup_complete_panel(live_bridge)


def _live_analysis_settings_from_session():
    """Read current Live Session model/preset choices without mutating globals."""
    model_options = get_live_model_options()
    selected_model_name = st.session_state.get(
        "live_session_model",
        list(model_options.keys())[0],
    )
    selected_model = model_options.get(selected_model_name) or next(iter(model_options.values()))
    preset_name = st.session_state.get("live_session_preset", "Balanced Mode")
    active_preset = DETECTION_PRESETS.get(preset_name) or DETECTION_PRESETS.get(
        "Balanced Mode",
        {"confidence": 0.25, "imgsz": 640},
    )
    return {
        "model_path": str(selected_model.get("path") or CRICKET_OBJECTS_MODEL_PATH),
        "model_key": selected_model.get("model_key"),
        "use_ensemble": bool(selected_model.get("ensemble")),
        "confidence": active_preset.get("confidence", 0.25),
        "imgsz": active_preset.get("imgsz", 640),
        "show_pitch_roi": bool(st.session_state.get("live_session_show_roi", False)),
        "model_name": selected_model_name,
        "preset_name": preset_name,
        "field_setup": st.session_state.get("current_field_setup"),
        "session_calibration": st.session_state.get("live_calibration_payload"),
    }


def _next_unanalysed_clip_path(live_bridge):
    snap = live_bridge.snapshot()
    analysed = set(st.session_state.get("live_analysed_clip_paths") or [])
    for path in snap.get("saved_clip_paths") or []:
        if path and path not in analysed:
            return path
    last_path = snap.get("last_saved_path") or st.session_state.get("live_last_saved_delivery")
    if last_path and last_path not in analysed:
        return last_path
    return None


def _maybe_poll_for_saved_clips(live_bridge):
    """If Streamlit supports fragments, lightly poll for webrtc-saved clips."""
    fragment = getattr(st, "fragment", None)
    if fragment is None:
        return
    try:

        @fragment(run_every=2.5)
        def _poll():
            if _next_unanalysed_clip_path(live_bridge):
                st.rerun()

        _poll()
    except Exception:
        # ponytail: older Streamlit or fragment quirks — Refresh status button still works.
        pass


def process_pending_live_clip_analysis(live_bridge):
    """If a new saved clip exists, analyse it once via Video Analysis process_video."""
    pending = st.session_state.get("live_pending_clip_path")
    if not pending:
        pending = _next_unanalysed_clip_path(live_bridge)
    if not pending:
        return False

    st.session_state.live_pending_clip_path = pending
    st.session_state.live_last_saved_delivery = pending
    settings = _live_analysis_settings_from_session()
    with st.spinner("Analysing delivery..."):
        result = analyse_saved_live_clip(pending, **settings)

    analysed = list(st.session_state.get("live_analysed_clip_paths") or [])
    if pending not in analysed:
        analysed.append(pending)
    st.session_state.live_analysed_clip_paths = analysed
    st.session_state.live_last_result = result
    st.session_state.live_pending_clip_path = None

    by_clip = dict(st.session_state.get("live_result_by_clip") or {})
    by_clip[pending] = result
    st.session_state.live_result_by_clip = by_clip

    log = list(st.session_state.get("live_delivery_log") or [])
    log.append(
        {
            "clip_path": pending,
            "status": "done" if result.get("success") else "error",
            "error": None if result.get("success") else result.get("error"),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "line": result.get("estimated_line"),
            "length": result.get("estimated_length"),
            "ball_detection_rate": result.get("ball_detection_rate"),
            "output_path": result.get("output_path"),
        }
    )
    st.session_state.live_delivery_log = log

    if result.get("success"):
        try:
            _persist_result_to_session(
                result,
                "Live Session",
                video_name=Path(pending).name,
            )
        except Exception:
            pass
        live_bridge.status_message = "Analysis ready — continue bowling"
    else:
        live_bridge.status_message = "Analysis failed — clip kept; continue bowling"

    return True


def render_live_stage_delivery_capture(live_bridge):
    """Stage: delivery_capture — compact header above camera; drains pending analysis."""
    from Backends.src.ui.theme import render_step_header, render_status_row

    render_step_header(
        "Stage 3",
        "Live Capture",
        "Bowl deliveries. Clips save automatically, then each clip is analysed.",
    )

    # Drain one pending clip analysis per rerun (keeps UI responsive).
    if process_pending_live_clip_analysis(live_bridge):
        st.rerun()

    snap = live_bridge.snapshot()
    status = snap.get("status_message") or "Waiting for delivery..."
    recording = bool(snap.get("recording"))
    render_status_row(
        [
            ("Recording" if recording else "Waiting", "warning" if recording else "success"),
            (f"Deliveries: {snap['delivery_count']}", "gold"),
        ]
    )
    st.info(status)


def render_live_stage_delivery_capture_panel(live_bridge):
    """Below-camera controls + last delivery result for delivery_capture."""
    snap = live_bridge.snapshot()
    last_path = snap["last_saved_path"] or st.session_state.get("live_last_saved_delivery")

    status_cols = st.columns(3)
    status_cols[0].metric("Delivery count", snap["delivery_count"])
    status_cols[1].metric(
        "Recording status",
        "Recording" if snap.get("recording") else "Waiting",
    )
    status_cols[2].metric(
        "Last saved clip",
        Path(last_path).name if last_path else "None",
    )

    poll_cols = st.columns(2)
    if poll_cols[0].button("Refresh status", use_container_width=True):
        st.rerun()
    poll_cols[1].caption("Click refresh after bowling if the clip status looks stale.")

    # ponytail: optional Streamlit fragment poll so webrtc-saved clips surface without a new dep.
    _maybe_poll_for_saved_clips(live_bridge)

    if last_path:
        st.session_state.live_last_saved_delivery = last_path
        st.caption(f"Last saved delivery clip: `{last_path}`")

    last_result = st.session_state.get("live_last_result")
    if last_result:
        st.markdown("### Last delivery result")
        render_live_delivery_result_panel(last_result)
    elif last_path:
        st.caption("Replay / trajectory appear here after clip processing.")

    cols = st.columns(2)
    if cols[0].button("Stop Session", type="primary", use_container_width=True):
        with live_bridge.lock:
            live_bridge.live_session_active = False
            if live_bridge.capture_state.get("recording") and live_bridge.capture_state.get("frames"):
                save_result = save_delivery_clip(
                    list(live_bridge.capture_state["frames"]),
                    output_dir=LIVE_DELIVERIES_DIR,
                    fps=DEFAULT_RECORDING_FPS,
                    prefix="delivery",
                )
                if save_result.get("saved"):
                    path_str = str(save_result.get("path"))
                    live_bridge.last_saved_path = path_str
                    if path_str not in live_bridge.saved_clip_paths:
                        live_bridge.saved_clip_paths.append(path_str)
                    live_bridge.capture_state["delivery_count"] = (
                        int(live_bridge.capture_state.get("delivery_count") or 0) + 1
                    )
                    live_bridge.capture_state["last_delivery_time"] = datetime.now().isoformat(
                        timespec="seconds"
                    )
            live_bridge.capture_state["recording"] = False
            live_bridge.capture_state["frames"] = []
            live_bridge.status_message = "Session stopped"
        st.session_state.live_auto_session_active = False
        # Analyse any remaining unprocessed clips before results.
        while process_pending_live_clip_analysis(live_bridge):
            pass
        _set_live_session_stage("session_summary")
        st.session_state.live_status_message = "Session stopped."
        st.rerun()
    if cols[1].button("Back to Setup", use_container_width=True):
        live_bridge.reset_capture()
        st.session_state.live_auto_session_active = False
        st.session_state.live_alignment_report = None
        st.session_state.live_calibration_payload = None
        _set_live_session_stage("setup")
        st.rerun()


def render_live_stage_session_results(live_bridge):
    """Stage: session_results — recent deliveries + last analysis."""
    from Backends.src.ui.theme import render_step_header, render_status_row

    # Catch any clip saved just before stop that wasn't analysed yet.
    if process_pending_live_clip_analysis(live_bridge):
        st.rerun()

    render_step_header(
        "Stage 4",
        "Session Results",
        "Recent deliveries with clip + analysis status.",
    )
    snap = live_bridge.snapshot()
    log = list(st.session_state.get("live_delivery_log") or [])
    last_path = snap["last_saved_path"] or st.session_state.get("live_last_saved_delivery")
    last_result = st.session_state.get("live_last_result")
    render_status_row(
        [
            (f"Deliveries: {snap['delivery_count']}", "success"),
            (
                "Analysis: Ready" if last_result and last_result.get("success") else "Analysis: Pending/None",
                "gold" if last_result else "default",
            ),
        ]
    )
    st.metric("Deliveries captured", snap["delivery_count"])

    if log:
        st.markdown("### Recent deliveries")
        for index, entry in enumerate(reversed(log), start=1):
            status = entry.get("status") or "unknown"
            label = Path(entry.get("clip_path") or "").name or f"Delivery {index}"
            with st.expander(
                f"{label} — {status}"
                + (
                    f" | {entry.get('line') or '—'} / {entry.get('length') or '—'}"
                    if status == "done"
                    else ""
                ),
                expanded=(index == 1),
            ):
                st.caption(f"Clip: `{entry.get('clip_path')}`")
                if entry.get("error"):
                    st.error(entry["error"])
                clip = entry.get("clip_path")
                if clip and Path(clip).is_file():
                    try:
                        st.video(str(clip))
                    except Exception:
                        pass
                stored = (st.session_state.get("live_result_by_clip") or {}).get(clip)
                if stored:
                    render_live_delivery_result_panel(stored, expanded_details=(index == 1))
                elif (
                    last_result
                    and last_result.get("source_clip_path") == entry.get("clip_path")
                ):
                    render_live_delivery_result_panel(last_result, expanded_details=True)
                elif entry.get("output_path") and Path(entry["output_path"]).is_file():
                    st.markdown("**Processed replay**")
                    try:
                        st.video(str(entry["output_path"]))
                    except Exception:
                        st.caption(entry["output_path"])
    elif last_path:
        st.caption(f"Last saved clip: `{last_path}`")
        st.info("Replay available after clip processing")
    else:
        st.caption("No delivery clips were saved in this session.")

    if last_result and not log:
        st.markdown("### Last analysis result")
        render_live_delivery_result_panel(last_result, expanded_details=True)

    if st.button("Start New Session", type="primary", use_container_width=True):
        live_bridge.reset_capture()
        st.session_state.live_auto_session_active = False
        st.session_state.live_alignment_report = None
        st.session_state.live_calibration_payload = None
        st.session_state.live_alignment_box_layout = None
        st.session_state.live_delivery_log = []
        st.session_state.live_analysed_clip_paths = []
        st.session_state.live_last_result = None
        st.session_state.live_pending_clip_path = None
        st.session_state.live_result_by_clip = {}
        _set_live_session_stage("setup")
        st.rerun()


def write_video(frames, output_path, fps=DEFAULT_RECORDING_FPS):
    return write_video_frames(frames, output_path, fps=fps)


# LEGACY / NOT ACTIVE: Kept for future Live Session detector compatibility.
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


# LEGACY / NOT ACTIVE: Kept for future manual review export compatibility.
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
    delivery_report = {
        "estimated_line": estimated_line,
        "estimated_length": estimated_length,
        "ball_detection_rate": ball_detection_rate,
        "overall_tracking_quality": overall_tracking_quality,
    }
    reports = build_video_reports(
        impact_frame_detections,
        fps=fps,
        total_frames=total_frames,
        batter_handedness=batter_handedness,
        delivery_report=delivery_report,
        impact_result=impact_info,
        enable_visual_observer_repair=False,
    )
    impact_info = reports["impact_result"]
    shot_info = reports["shot_result"]
    direction_info = reports["direction_result"]
    outcome_info = reports["outcome_result"]
    agent_info = reports["agent_result"]
    enrichment = reports["enrichment"]

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
        "frame_detections": reports["frame_detections"],
        "impact_frame_detections": reports["frame_detections"],
        "observer_timeline": reports["observer_timeline"],
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


# LEGACY / NOT ACTIVE: Kept for compatibility with the earlier live report UI.
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
    st.session_state.live_auto_session_active = False
    st.session_state.live_stage = "setup"
    st.session_state.live_session_stage = "setup"
    st.session_state.live_box_layout = None
    st.session_state.live_alignment_box_layout = None
    st.session_state.live_alignment_frame_size = None
    st.session_state.live_alignment_report = None
    st.session_state.live_calibration_payload = None
    st.session_state.live_stump_validation = None
    st.session_state.live_stump_validation_history = None
    st.session_state.live_environment_context = None
    st.session_state.live_calibration_result = None
    st.session_state.live_calibration_snapshot = None
    st.session_state.live_calibration_snapshot_frame = None
    st.session_state.live_calibration_locked = False
    st.session_state.live_capture_state = None
    st.session_state.live_calibration_stump_override = False
    st.session_state.live_delivery_log = []
    st.session_state.live_analysed_clip_paths = []
    st.session_state.live_pending_clip_path = None
    st.session_state.live_result_by_clip = {}
    if "live_session_bridge" in st.session_state:
        st.session_state.live_session_bridge.reset_capture()


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
        "live_auto_session_active": False,
        "live_last_saved_delivery": None,
        "live_stage": "setup",
        "live_session_stage": "setup",
        "live_box_layout": None,
        "live_alignment_box_layout": None,
        "live_alignment_frame_size": None,
        "live_alignment_report": None,
        "live_calibration_payload": None,
        "live_stump_validation": None,
        "live_stump_validation_history": None,
        "live_environment_context": None,
        "live_calibration_result": None,
        "live_calibration_snapshot": None,
        "live_calibration_snapshot_frame": None,
        "live_calibration_locked": False,
        "live_capture_state": None,
        "live_calibration_stump_override": False,
        "live_session_bridge": LiveSessionBridge(),
        "live_delivery_log": [],
        "live_analysed_clip_paths": [],
        "live_pending_clip_path": None,
        "live_result_by_clip": {},
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if not isinstance(st.session_state.get("live_session_bridge"), LiveSessionBridge):
        st.session_state.live_session_bridge = LiveSessionBridge()
    stage = _get_live_stage()
    if stage not in LIVE_STAGES:
        stage = "setup"
        st.session_state.live_stage = stage
        st.session_state.live_session_stage = stage


def show_live_session_page():
    from Backends.src.ui.interactive_field_map import render_field_setup_card
    from Backends.src.ui.theme import render_page_header, render_status_row

    initialize_live_session_state()
    recording_state = st.session_state.live_recording_state
    live_bridge = st.session_state.live_session_bridge
    stage = _get_live_stage()

    if recording_state.recording:
        status = "Recording"
        status_tone = "warning"
    elif st.session_state.live_pending_analysis:
        status = "Analyzing"
        status_tone = "gold"
    elif st.session_state.live_camera_session_ended and st.session_state.live_last_result:
        status = "Review Ready"
        status_tone = "success"
    elif stage == "live_capture":
        status = "Capturing"
        status_tone = "warning"
    elif stage == "align_stumps":
        status = "Calibrating"
        status_tone = "warning"
    elif stage == "setup_complete":
        status = "Setup Complete"
        status_tone = "success"
    elif stage == "session_summary":
        status = "Results"
        status_tone = "gold"
    else:
        status = "Setup"
        status_tone = "default"

    # ponytail: calibration stages stay compact — skip the heavy page chrome.
    if stage == "setup":
        pass
    elif stage not in {"align_stumps", "setup_complete", "calibration_solving"}:
        render_page_header(
            "Live Bowling Session",
            "Align the camera to fixed stump boxes, capture deliveries live, then review results.",
            badge=status,
        )
        render_status_row(
            [
                (f"Stage: {status}", status_tone),
                ("Live preview stays clean until analysis", "gold"),
            ]
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
                _persist_result_to_session(
                    st.session_state.live_last_result,
                    "Live Session",
                    video_name=st.session_state.live_last_result.get("processed_file_name"),
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

    if st.session_state.live_status_message:
        st.info(st.session_state.live_status_message)
        st.session_state.live_status_message = None

    bridge_snap = live_bridge.snapshot()
    frame_width, frame_height = bridge_snap.get("frame_size") or (
        DEFAULT_LIVE_FRAME_WIDTH,
        DEFAULT_LIVE_FRAME_HEIGHT,
    )
    frame_width, frame_height = int(frame_width), int(frame_height)

    box_layout = None
    calibration_payload = None
    show_alignment_boxes = False
    show_calibrated_geometry = False
    camera_needed = stage in {
        "align_stumps",
        "calibration_solving",
        "setup_complete",
        "live_capture",
    }

    if stage == "setup":
        render_live_stage_setup(live_bridge)
    elif stage == "align_stumps":
        st.markdown("### Live Bowling Session")
        box_layout = _ensure_alignment_box_layout(frame_width, frame_height)
        show_alignment_boxes = True
        calibration_payload = None
        show_calibrated_geometry = False
        if not box_layout.get("available"):
            st.warning("Could not build alignment boxes for this frame size.")
        render_live_calibration_instructions()
        bridge_snap = live_bridge.snapshot()
        validation = bridge_snap.get("stump_validation") or st.session_state.get("live_stump_validation") or {}
        st.session_state.live_stump_validation = validation
        render_live_calibration_status_panel(
            validation,
            bridge_snap.get("detector_available"),
        )
        with st.expander("Developer / Advanced", expanded=False):
            st.checkbox(
                "Manual Override / Low",
                value=False,
                key="live_calibration_stump_override",
                help="Continue without stump detections — low-quality geometry only.",
            )
    elif stage == "calibration_solving":
        st.markdown("### Live Bowling Session")
        st.info("Detecting stumps on snapshot...")
        box_layout = _ensure_alignment_box_layout(frame_width, frame_height)
        show_alignment_boxes = True
    elif stage == "setup_complete":
        st.markdown("### Live Bowling Session")
        box_layout = _ensure_alignment_box_layout(frame_width, frame_height)
        calibration_payload = (
            st.session_state.get("live_calibration_payload")
            or (st.session_state.get("live_alignment_report") or {}).get("calibration")
            or st.session_state.get("live_calibration_result")
        )
        if isinstance(calibration_payload, dict) and "calibration" in calibration_payload:
            calibration_payload = calibration_payload.get("calibration") or calibration_payload
        show_alignment_boxes = False
        show_calibrated_geometry = True
    elif stage == "live_capture":
        box_layout = _ensure_alignment_box_layout(frame_width, frame_height)
        render_live_stage_delivery_capture(live_bridge)
        calibration_payload = st.session_state.get("live_calibration_payload")
        if isinstance(calibration_payload, dict) and "calibration" in calibration_payload:
            calibration_payload = calibration_payload.get("calibration") or calibration_payload
        show_alignment_boxes = False
        show_calibrated_geometry = True
    elif stage == "session_summary":
        render_live_stage_session_results(live_bridge)
        with st.expander("Developer / Advanced", expanded=False):
            st.caption("Session debug snapshot")
            st.json(
                {
                    "stage": stage,
                    "delivery_count": bridge_snap.get("delivery_count"),
                    "last_saved_path": bridge_snap.get("last_saved_path")
                    or st.session_state.get("live_last_saved_delivery"),
                    "analysed_clips": st.session_state.get("live_analysed_clip_paths"),
                    "delivery_log_count": len(st.session_state.get("live_delivery_log") or []),
                }
            )
    else:
        _set_live_session_stage("setup")
        st.rerun()

    if camera_needed:
        field_setup = st.session_state.get("current_field_setup")
        # ponytail: hide field/advanced/manual capture during alignment — camera only.
        show_setup_panels = stage == "live_capture"

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

        try:
            from streamlit_webrtc import RTCConfiguration, webrtc_streamer
        except ImportError:
            st.error("streamlit-webrtc is not installed. Add streamlit-webrtc to requirements.txt.")
            return

        session_active = bool(st.session_state.get("live_auto_session_active", False)) and (
            stage == "live_capture"
        )
        detector_for_bridge = None
        detector_available = None
        if stage in {"align_stumps", "live_capture"}:
            with live_bridge.lock:
                detector_for_bridge = live_bridge.detector_model
            if detector_for_bridge is None:
                detector_for_bridge = warm_live_detector()
            detector_available = detector_for_bridge is not None
        live_bridge.configure(
            stage=stage,
            box_layout=box_layout,
            calibration=calibration_payload,
            show_alignment_boxes=show_alignment_boxes,
            show_calibrated_geometry=show_calibrated_geometry,
            live_session_active=session_active,
            detector_model=detector_for_bridge,
            ball_confidence=confidence,
            detector_available=detector_available,
            show_pitch_axis_preview=bool(
                st.session_state.get("live_show_pitch_axis_preview", False)
            ),
        )

        rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        webrtc_context = None

        if st.session_state.live_camera_active:
            webrtc_context = webrtc_streamer(
                key=f"cricvision-live-delivery-recorder-{st.session_state.live_webrtc_key_suffix}",
                video_processor_factory=create_delivery_recorder_class(
                    st.session_state.live_recording_state,
                    live_bridge=live_bridge,
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

        # Actions / status sit under the camera for mobile-friendly calibration.
        if stage == "align_stumps":
            bridge_snap = live_bridge.snapshot()
            validation = bridge_snap.get("stump_validation") or st.session_state.get("live_stump_validation") or {}
            render_live_stage_camera_calibration_actions(
                live_bridge,
                frame_width,
                frame_height,
                box_layout,
                validation=validation,
                detector_available=bridge_snap.get("detector_available"),
                detector_model=detector_for_bridge,
                ball_confidence=confidence,
            )
        elif stage == "setup_complete":
            calibration_payload = render_live_stage_setup_complete_panel(live_bridge)
            live_bridge.configure(
                stage=stage,
                box_layout=box_layout,
                calibration=calibration_payload,
                show_alignment_boxes=False,
                show_calibrated_geometry=True,
                live_session_active=False,
            )
        elif stage == "live_capture":
            render_live_stage_delivery_capture_panel(live_bridge)

        if show_setup_panels:
            with st.expander("Field Setup", expanded=False):
                field_setup = render_field_setup_card(
                    key_prefix="live_session_field",
                    compact=True,
                    default_preset="Balanced",
                )

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

            with st.expander("Manual single-delivery recording (optional)", expanded=False):
                if not recording_state.recording:
                    start_clicked = st.button(
                        "Start Delivery Recording",
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
                        st.caption("Optional: record one clip manually, then analyze.")
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

            if st.session_state.live_last_result:
                with st.expander("Recent Delivery Summary", expanded=False):
                    show_analysis_output(st.session_state.live_last_result)
    else:
        live_bridge.configure(
            stage=stage,
            box_layout=None,
            calibration=None,
            show_alignment_boxes=False,
            show_calibrated_geometry=False,
            live_session_active=False,
        )
