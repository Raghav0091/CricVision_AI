import json
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
from Backends.src.calibration.calibration_context import (
    build_calibration_context,
)
from Backends.src.config.constants import DETECTION_PRESETS
from Backends.src.config.paths import PROCESSED_VIDEO_DIR, REPORTS_DIR
from Backends.src.engine import EngineOptions, analyze_delivery_clip
from Backends.src.models.model_registry import (
    get_model_path,
    validate_model_paths,
)
from Backends.src.ui.analysis_helpers import (
    ensure_delivery_report_fields,
    persist_result_to_session as _persist_result_to_session,
)
from Backends.src.utils.cv2_loader import cv2
from Backends.src.video_pipeline.annotation_writer import draw_label
from Backends.src.video_pipeline.detection_pipeline import get_model_options
from Backends.src.video_pipeline.video_reader import (
    extract_first_video_frame,
)


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
        "min_ball_bat_distance_px": impact.get(
            "min_ball_bat_distance_px"
        ),
        "impact_confidence": impact.get(
            "impact_confidence",
            "Unknown",
        ),
        "impact_reason": impact.get(
            "reason",
            impact.get("impact_reason", ""),
        ),
        "impact_frame_image_path": str(
            impact.get("impact_frame_image_path") or ""
        ),
        "shot_type": result.get("shot_info", {}).get(
            "shot_type",
            "Unknown",
        ),
        "shot_confidence": result.get("shot_info", {}).get(
            "shot_confidence",
            "Unknown",
        ),
        "shot_direction": result.get("shot_info", {}).get(
            "shot_direction",
            "Unknown",
        ),
        "shot_height": result.get("shot_info", {}).get(
            "shot_height",
            "Unknown",
        ),
        "shot_reason": result.get("shot_info", {}).get(
            "reason",
            result.get("shot_info", {}).get("shot_reason", ""),
        ),
        "predicted_outcome": result.get("outcome_info", {}).get(
            "predicted_outcome",
            "Unknown",
        ),
        "outcome_confidence": result.get("outcome_info", {}).get(
            "outcome_confidence",
            "Unknown",
        ),
        "run_estimate": result.get("outcome_info", {}).get(
            "run_estimate"
        ),
        "dismissal_risk": result.get("outcome_info", {}).get(
            "dismissal_risk",
            "Unknown",
        ),
        "boundary_chance": result.get("outcome_info", {}).get(
            "boundary_chance",
            "Unknown",
        ),
        "outcome_reason": result.get("outcome_info", {}).get(
            "reason",
            result.get("outcome_info", {}).get("outcome_reason", ""),
        ),
        "field_zone": result.get("field_zone", "Unknown"),
        "zone_confidence": result.get("zone_confidence", "Unknown"),
        "direction_angle_degrees": result.get(
            "direction_angle_degrees"
        ),
        "direction_reason": result.get("direction_reason", ""),
        "movement_dx": result.get("movement_dx"),
        "movement_dy": result.get("movement_dy"),
        "direction_shot_category": result.get(
            "direction_shot_category",
            "Unknown",
        ),
        "agent_quality": result.get("agent_quality", "Unknown"),
        "agent_confidence": result.get(
            "agent_confidence",
            "Unknown",
        ),
        "ball_tracking_coverage": result.get(
            "ball_tracking_coverage"
        ),
        "bat_detection_coverage": result.get(
            "bat_detection_coverage"
        ),
        "stump_detection_coverage": result.get(
            "stump_detection_coverage"
        ),
        "missing_ball_frames": result.get("missing_ball_frames", 0),
        "possible_false_ball_detections": result.get(
            "possible_false_ball_detections",
            0,
        ),
        "analysis_consistency": result.get(
            "analysis_consistency",
            "Unknown",
        ),
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
        draw_label(
            preview,
            label.replace("_", " "),
            point[0],
            point[1],
            (40, 160, 40),
        )

    st.image(
        cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
        caption=(
            "First frame with default pitch-point guide. "
            "Adjust the coordinates below."
        ),
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
        st.warning(
            "Warnings:\n"
            + "\n".join(f"- {warning}" for warning in warnings)
        )


def show_batting_analysis_results(result):
    from Backends.src.ui.components import (
        render_video_analysis_results_layout,
    )

    render_video_analysis_results_layout(
        result,
        context_label="Video Analysis",
    )


def show_video_analysis_results(
    result,
    selected_model_name,
    preset_name,
    show_pitch_roi,
):
    from Backends.src.ui.components import (
        render_video_analysis_results_layout,
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
        (
            "Upload a delivery clip. CricVision uses smart defaults and "
            "generates a processed video plus professional report."
        ),
    )

    if "video_analysis_result" not in st.session_state:
        st.session_state.video_analysis_result = None
    if "video_analysis_settings" not in st.session_state:
        st.session_state.video_analysis_settings = {}

    model_options = get_model_options()

    from Backends.src.ui.interactive_field_map import render_field_setup_card

    field_setup = render_field_setup_card(
        key_prefix="video_analysis_field",
        compact=True,
        default_preset="Balanced",
    )

    st.subheader("Practice Environment Calibration")
    calibration_enabled = st.checkbox(
        "Enable practice environment calibration",
        value=True,
        key="practice_calibration_enabled",
        help=(
            "Adds approximate 2D stump, crease, pitch-corridor, "
            "and line references."
        ),
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
        (
            "Auto-estimate stumps and pitch corridor from "
            "analysis detections"
        ),
        value=True,
        key="practice_calibration_auto_estimate",
        disabled=not calibration_enabled,
        help=(
            "Uses stump detections already produced after Analyze is "
            "clicked; it does not run another model."
        ),
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
                    "Provisional geometry will be refined from existing "
                    "stump detections during analysis."
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

    st.selectbox(
        "Ball tracking mode",
        ["Balanced", "Accuracy / Small Ball"],
        index=0,
        key="video_analysis_ball_tracking_mode",
        help=(
            "Balanced keeps current defaults. Accuracy / Small Ball "
            "uses lower confidence and larger image size for distant "
            "or small balls."
        ),
    )

    st.selectbox(
        "Analysis Mode",
        ["Smart Balanced", "Smart Accurate", "Debug Full Frame"],
        index=0,
        key="video_analysis_speed_mode",
        help=(
            "Smart Balanced keeps ball detection on every frame but "
            "reduces wasted work from other models."
        ),
    )

    st.checkbox(
        "Generate processed video preview",
        value=True,
        key="video_analysis_generate_processed_video",
        help=(
            "Disable to run analysis and reports only without writing "
            "an annotated video."
        ),
    )

    st.selectbox(
        "Overlay detail",
        ["Clean", "Debug"],
        index=0,
        key="video_analysis_overlay_detail",
        help=(
            "Clean keeps ball trail and key markers only. Debug shows "
            "ROI, bounce, and labels."
        ),
    )

    with st.expander("Advanced Settings", expanded=False):
        analysis_mode = st.selectbox(
            "Analysis mode",
            [
                "Bowling Analysis",
                "Batting Analysis",
                "Full Delivery Analysis",
            ],
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
            selected_ball_model_key = batting_ball_options[
                selected_model_name
            ]
            selected_model_path = get_model_path(selected_ball_model_key)
            use_ensemble = False
            selected_bat_model_key = "cricshot_bat"
            st.selectbox(
                "Bat model",
                ["CricShot10k Bat Detector"],
                key="video_analysis_bat_model",
            )
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
                st.selectbox(
                    "Bat model",
                    ["CricShot10k Bat Detector"],
                    key="video_analysis_full_bat_model",
                )

        preset_name = st.selectbox(
            "Detection preset",
            list(DETECTION_PRESETS.keys()),
            index=1,
            key="video_analysis_preset",
        )
        active_preset = DETECTION_PRESETS[preset_name]
        confidence = active_preset["confidence"]
        image_size = active_preset["imgsz"]

        show_pitch_roi = st.checkbox(
            "Show pitch ROI overlay",
            value=False,
            key="video_analysis_show_roi",
        )
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
        if shot_trajectory_mode == "Manually mark bat contact frame":
            st.number_input(
                "Bat contact frame",
                min_value=0,
                value=0,
                step=1,
                key="video_analysis_bat_contact_frame",
            )

        with st.expander("Model Status", expanded=False):
            for status in validate_model_paths().values():
                st.write(f"{status['status']}: {status['name']}")

        with st.expander(
            "Advanced analysis settings",
            expanded=False,
        ):
            limit_frames_enabled = st.checkbox(
                "Limit frames for testing",
                value=False,
                key="video_analysis_limit_frames_enabled",
            )
            st.selectbox(
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

    analysis_mode = st.session_state.get(
        "video_analysis_mode",
        "Full Delivery Analysis",
    )
    preset_name = st.session_state.get(
        "video_analysis_preset",
        "Balanced Mode",
    )
    active_preset = DETECTION_PRESETS[preset_name]
    confidence = active_preset["confidence"]
    image_size = active_preset["imgsz"]
    show_pitch_roi = st.session_state.get(
        "video_analysis_show_roi",
        False,
    )
    calibration_mode = st.session_state.get(
        "video_analysis_calibration_mode",
        "Auto calibration using detected stumps",
    )
    shot_trajectory_mode = st.session_state.get(
        "video_analysis_shot_trajectory_mode",
        "Use last part of trajectory",
    )
    manual_contact_frame = st.session_state.get(
        "video_analysis_bat_contact_frame",
        0,
    )
    if shot_trajectory_mode != "Manually mark bat contact frame":
        manual_contact_frame = None

    from Backends.src.analysis.analysis_speed import resolve_frame_limit

    speed_mode = st.session_state.get(
        "video_analysis_speed_mode",
        "Smart Balanced",
    )
    ball_tracking_mode = st.session_state.get(
        "video_analysis_ball_tracking_mode",
        "Balanced",
    )
    generate_processed_video = st.session_state.get(
        "video_analysis_generate_processed_video",
        True,
    )
    overlay_detail = st.session_state.get(
        "video_analysis_overlay_detail",
        "Clean",
    )
    max_frames = resolve_frame_limit(
        st.session_state.get(
            "video_analysis_limit_frames_enabled",
            False,
        ),
        st.session_state.get(
            "video_analysis_frame_limit_choice",
            "All frames",
        ),
    )
    show_performance = st.session_state.get(
        "video_analysis_show_performance",
        False,
    )

    if analysis_mode == "Batting Analysis":
        batting_ball_options = {
            "Current Best Ball + Stump Model": "current_best",
            "CricShot10k Ball Detector": "cricshot_ball",
        }
        selected_model_name = st.session_state.get(
            "video_analysis_batting_ball_model",
            "Current Best Ball + Stump Model",
        )
        selected_ball_model_key = batting_ball_options.get(
            selected_model_name,
            "current_best",
        )
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
        selected_bat_model_key = (
            "cricshot_bat"
            if analysis_mode == "Full Delivery Analysis"
            else None
        )

    manual_pitch_points = None
    if analyze_clicked and uploaded_video is not None:
        if calibration_mode.startswith("Manual"):
            first_frame = extract_first_video_frame(uploaded_video)
            if first_frame is None:
                st.warning(
                    "Could not read the first frame for manual calibration."
                )
            else:
                with st.expander(
                    "Manual Pitch Calibration",
                    expanded=True,
                ):
                    manual_pitch_points = show_manual_pitch_point_inputs(
                        first_frame
                    )

        uploaded_video.seek(0)
        uploaded_bytes = uploaded_video.read()
        if not uploaded_bytes:
            st.error(
                "The uploaded video is empty. "
                "Choose a non-empty cricket clip."
            )
            st.session_state.video_analysis_result = None
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        PROCESSED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        raw_output_path = (
            PROCESSED_VIDEO_DIR
            / f"raw_cricvision_analysis_{timestamp}.mp4"
        )
        browser_output_path = (
            PROCESSED_VIDEO_DIR
            / f"cricvision_analysis_{timestamp}.mp4"
        )

        upload_suffix = Path(
            uploaded_video.name or ""
        ).suffix.lower()
        if upload_suffix not in {".mp4", ".mov", ".avi", ".mkv"}:
            upload_suffix = ".mp4"
        try:
            with TemporaryDirectory(
                prefix="cricvision_upload_"
            ) as temp_dir:
                input_video_path = (
                    Path(temp_dir) / f"uploaded_video{upload_suffix}"
                )
                input_video_path.write_bytes(uploaded_bytes)

                with st.spinner("Analyzing delivery..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(frame_index, total_frames):
                        if total_frames > 0:
                            progress_bar.progress(
                                min(frame_index / total_frames, 1.0)
                            )
                            status_text.text(
                                "Processing frame "
                                f"{frame_index}/{total_frames}"
                            )
                        else:
                            status_text.text(
                                f"Processing frame {frame_index}"
                            )

                    engine_options = EngineOptions(
                        analysis_mode=analysis_mode,
                        smart_mode=speed_mode,
                        ball_tracking_mode=ball_tracking_mode,
                        processed_video_enabled=(
                            generate_processed_video
                        ),
                        overlay_detail=overlay_detail,
                        confidence_threshold=confidence,
                        output_path=raw_output_path,
                        browser_output_path=browser_output_path,
                        model_path=selected_model_path,
                        model_key=selected_model_key,
                        model_name=selected_model_name,
                        ball_model_key=(
                            selected_ball_model_key
                            if analysis_mode == "Batting Analysis"
                            else selected_model_key
                        ),
                        bat_model_key=selected_bat_model_key,
                        image_size=image_size,
                        use_ensemble=use_ensemble,
                        show_pitch_roi=show_pitch_roi,
                        calibration_mode=calibration_mode,
                        manual_pitch_points=manual_pitch_points,
                        shot_trajectory_mode=shot_trajectory_mode,
                        manual_contact_frame=manual_contact_frame,
                        field_setup=field_setup,
                        max_frames=max_frames,
                        active_preset=preset_name,
                        show_performance_details=show_performance,
                        progress_callback=update_progress,
                    )
                    try:
                        result = analyze_delivery_clip(
                            input_video_path,
                            calibration_context=(
                                practice_calibration_context
                            ),
                            options=engine_options,
                        )
                    finally:
                        progress_bar.empty()
                        status_text.empty()
        except Exception as error:
            print(
                "Video analysis failed: "
                f"{type(error).__name__}: {error}"
            )
            raw_output_path.unlink(missing_ok=True)
            browser_output_path.unlink(missing_ok=True)
            st.error(
                "Video analysis could not complete. Check that the clip "
                "is readable and uses a supported codec, then try again."
            )
            st.session_state.video_analysis_result = None
            return

        if not result.get("success"):
            raw_output_path.unlink(missing_ok=True)
            browser_output_path.unlink(missing_ok=True)
            st.error(
                result.get(
                    "error",
                    "Video analysis did not complete.",
                )
            )
            st.session_state.video_analysis_result = None
        else:
            if result.get("processed_video_conversion") == "failed":
                st.warning(
                    "Processed video preview conversion failed. "
                    "Analysis results are still available and the raw "
                    "video can be downloaded."
                )
            try:
                if analysis_mode in {
                    "Batting Analysis",
                    "Full Delivery Analysis",
                }:
                    save_batting_report(result, analysis_mode)
                video_name = (
                    uploaded_video.name
                    if uploaded_video is not None
                    else None
                )
                _persist_result_to_session(
                    result,
                    "Video Analysis",
                    video_name=video_name,
                )
                st.session_state.video_analysis_result = result
                st.session_state.video_analysis_settings = {
                    "analysis_mode": analysis_mode,
                    "ball_tracking_mode": ball_tracking_mode,
                    "selected_model_name": selected_model_name,
                    "preset_name": preset_name,
                    "show_pitch_roi": show_pitch_roi,
                    "shot_trajectory_mode": shot_trajectory_mode,
                    "speed_mode": speed_mode,
                    "generate_processed_video": (
                        generate_processed_video
                    ),
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
            (
                "Upload a clip and click Analyze Delivery to generate "
                "a processed video and report."
            ),
            action_label="Smart defaults are applied automatically",
        )
    elif result.get("analysis_mode") == "Batting Analysis":
        show_batting_analysis_results(result)
    else:
        show_video_analysis_results(
            result=result,
            selected_model_name=settings.get(
                "selected_model_name",
                result.get("active_model", "Unknown"),
            ),
            preset_name=settings.get(
                "preset_name",
                result.get("active_preset", "Balanced Mode"),
            ),
            show_pitch_roi=settings.get("show_pitch_roi", False),
        )
