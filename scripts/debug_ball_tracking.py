#!/usr/bin/env python3
"""Generate ball-tracking diagnostics for one local cricket video.

The script is intentionally separate from Streamlit and normal engine output:
it records what the current pipeline is doing without tuning thresholds or
changing production analysis behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime
from math import hypot
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Backends.src.analysis.analysis_speed import (  # noqa: E402
    get_analysis_mode_settings,
    resize_frame_for_inference,
    scale_detections_to_original,
)
from Backends.src.analysis.smart_pipeline import (  # noqa: E402
    apply_locked_stump,
    lock_static_stump_detection,
    should_detect_ball,
    should_detect_stump,
)
from Backends.src.calibration.calibration_context import (  # noqa: E402
    build_calibration_context,
    normalize_calibration_context,
)
from Backends.src.config.paths import (  # noqa: E402
    CRICKET_OBJECTS_MODEL_PATH,
    PROJECT_ROOT,
)
from Backends.src.models.model_registry import (  # noqa: E402
    get_model_info,
    get_model_path,
)
from Backends.src.tracking.ball_tracking_utils import (  # noqa: E402
    BallKalmanTracker,
    calculate_tracking_quality,
    get_tracking_quality_label,
    smooth_trajectory,
)
from Backends.src.tracking.trajectory_scorer import (  # noqa: E402
    TrajectoryBallSelector,
    build_ball_positions_from_tracklet,
    count_after_track_terminated_from_frame,
    resolve_delivery_tracking_quality,
    select_online_best_tracklet,
    should_enable_online_best_tracklet,
)
from Backends.src.tracking.trajectory_fit import fit_delivery_trajectory  # noqa: E402
from Backends.src.utils.cv2_loader import cv2  # noqa: E402
from Backends.src.video_pipeline.annotation_writer import (  # noqa: E402
    draw_fitted_trajectory_overlay,
    draw_label,
    draw_pitch_roi,
    draw_search_roi,
    draw_trajectory_lines,
    ensure_frame_writer_size,
)
from Backends.src.video_pipeline.detection_pipeline import (  # noqa: E402
    get_model_names,
    load_detection_model,
    map_model_classes,
    run_local_redetection,
    run_pitch_roi_detection,
)


CSV_COLUMNS = [
    "frame_index",
    "timestamp_sec",
    "frame_width",
    "frame_height",
    "model_path",
    "class_names",
    "detection_source",
    "detection_ran",
    "used_roi",
    "roi_x1",
    "roi_y1",
    "roi_x2",
    "roi_y2",
    "raw_ball_candidate_count",
    "raw_stump_candidate_count",
    "candidate_id",
    "class_name",
    "confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "center_x",
    "center_y",
    "candidate_selected",
    "candidate_rejected",
    "rejection_reason",
    "selection_score",
    "reference_x",
    "reference_y",
    "predicted_x",
    "predicted_y",
    "distance_from_previous",
    "distance_from_prediction",
    "inside_roi_or_corridor_if_available",
    "accepted_track_count_so_far",
    "tracking_quality_so_far",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Record raw ball candidates, trajectory selection decisions, "
            "ROI/calibration status, and timing for one video."
        )
    )
    parser.add_argument("video_path", help="Local video file to analyse")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "debug"),
        help="Directory for CSV/JSON/optional overlay outputs",
    )
    parser.add_argument(
        "--model-key",
        default="current_best",
        help="Registered YOLO model key for ball detection",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional explicit YOLO model path; overrides --model-key loading",
    )
    parser.add_argument(
        "--conf",
        "--confidence",
        dest="conf",
        type=float,
        default=0.25,
        help="Model confidence threshold used by the current pipeline",
    )
    parser.add_argument(
        "--ball-confidence",
        type=float,
        default=None,
        help=(
            "Optional ball candidate threshold. Defaults to --conf so small/far "
            "ball sensitivity tests actually lower the accepted candidate floor."
        ),
    )
    parser.add_argument(
        "--speed-mode",
        default="Smart Balanced",
        choices=["Smart Balanced", "Smart Accurate", "Debug Full Frame"],
        help="Reuse the existing smart-pipeline preset",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Override YOLO image size; defaults to the selected speed mode",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame cap for quick diagnostics",
    )
    parser.add_argument(
        "--every-nth-frame",
        type=int,
        default=1,
        help="Analyse every Nth source frame; defaults to every frame",
    )
    parser.add_argument(
        "--no-roi",
        action="store_true",
        help="Disable pitch ROI detection for this diagnostic run",
    )
    parser.add_argument(
        "--calibration-context",
        default=None,
        help="Optional JSON file containing a calibration_context object",
    )
    parser.add_argument(
        "--write-overlay",
        action="store_true",
        help="Also write a debug overlay MP4 with raw/selected/rejected candidates",
    )
    parser.add_argument(
        "--label-rejections",
        action="store_true",
        help="Draw small rejection-reason labels on the optional overlay",
    )
    return parser.parse_args(argv)


def apply_best_tracklet_post_pass(
    *,
    selector: TrajectoryBallSelector,
    raw_frame_ball_candidates: dict[int, list[dict]],
    width: int,
    height: int,
    fps: float,
    speed_mode: str,
    ball_positions: list,
    calibration_context,
    ball_tracking_mode: str = "Balanced",
) -> dict:
    """Apply shared offline best-tracklet ranking after the debug detection loop."""
    enabled = should_enable_online_best_tracklet(
        ball_tracking_mode=ball_tracking_mode,
        speed_mode=speed_mode,
    )
    result: dict = {
        "online_best_tracklet_enabled": enabled,
        "best_tracklet_applied": False,
        "fallback_reason": None,
        "online_selector_original_start_frame": selector.selected_track_start_frame(),
        "online_selector_original_end_frame": selector.selected_track_end_frame(),
        "online_selector_after_track_terminated_count": selector.rejection_reasons.get(
            "after_track_terminated",
            0,
        ),
        "candidate_segment_count": 0,
        "rejected_static_segment_count": 0,
        "best_segment_start_frame": None,
        "best_segment_end_frame": None,
        "best_segment_point_count": 0,
        "best_segment_duration_sec": None,
        "best_segment_score": None,
        "best_segment_reason": None,
        "ball_positions": ball_positions,
        "selected_ball_points": selector.accepted_point_count,
        "fit_result": None,
    }
    if not enabled:
        result["fallback_reason"] = "disabled"
        return result
    if not raw_frame_ball_candidates:
        result["fallback_reason"] = "no_raw_ball_candidates"
        return result

    best_tracklet = select_online_best_tracklet(
        raw_frame_ball_candidates,
        width,
        height,
        fps,
    )
    ranking = best_tracklet.get("ranking") or {}
    result["candidate_segment_count"] = ranking.get("candidate_segment_count", 0)
    result["rejected_static_segment_count"] = ranking.get(
        "rejected_static_segment_count",
        0,
    )
    if not best_tracklet.get("applied"):
        result["fallback_reason"] = best_tracklet.get(
            "fallback_reason",
            "no_valid_ranked_segment",
        )
        return result

    selector.apply_ranked_tracklet(
        best_tracklet["tracklet_points"],
        segment_end_frame=(
            best_tracklet.get("extended_segment_end_frame")
            or best_tracklet["best_segment_end_frame"]
        ),
    )
    ranked_ball_positions = build_ball_positions_from_tracklet(
        best_tracklet["tracklet_points"],
        len(ball_positions),
    )
    segment_end_frame = (
        best_tracklet.get("extended_segment_end_frame")
        or best_tracklet["best_segment_end_frame"]
    )
    fit_result = fit_delivery_trajectory(
        best_tracklet["tracklet_points"],
        frame_size=(width, height),
        fps=fps,
        calibration_context=calibration_context,
        delivery_track_terminated_frame=segment_end_frame,
    )
    result.update(
        {
            "best_tracklet_applied": True,
            "best_segment_start_frame": best_tracklet.get("best_segment_start_frame"),
            "best_segment_end_frame": best_tracklet.get("best_segment_end_frame"),
            "best_segment_point_count": best_tracklet.get("best_segment_point_count"),
            "best_segment_duration_sec": best_tracklet.get(
                "best_segment_duration_sec"
            ),
            "best_segment_score": best_tracklet.get("best_segment_score"),
            "best_segment_reason": best_tracklet.get("best_segment_reason"),
            "extension_enabled": best_tracklet.get("extension_enabled", False),
            "extension_applied": best_tracklet.get("extension_applied", False),
            "backward_extension_points": best_tracklet.get(
                "backward_extension_points",
                0,
            ),
            "forward_extension_points": best_tracklet.get(
                "forward_extension_points",
                0,
            ),
            "extended_segment_start_frame": best_tracklet.get(
                "extended_segment_start_frame"
            ),
            "extended_segment_end_frame": best_tracklet.get(
                "extended_segment_end_frame"
            ),
            "extended_segment_point_count": best_tracklet.get(
                "extended_segment_point_count",
                len(best_tracklet["tracklet_points"]),
            ),
            "extension_rejection_reasons": best_tracklet.get(
                "extension_rejection_reasons",
                {},
            ),
            "extension_fallback_reason": best_tracklet.get(
                "extension_fallback_reason"
            ),
            "extension_preserved_original_segment": best_tracklet.get(
                "extension_preserved_original_segment",
                not best_tracklet.get("extension_applied", False),
            ),
            "extension_fit_delta": best_tracklet.get("extension_fit_delta", 0),
            "trajectory_fit_quality_after_extension": best_tracklet.get(
                "trajectory_fit_quality_after_extension"
            ),
            "online_selector_after_track_terminated_count": (
                count_after_track_terminated_from_frame(
                    raw_frame_ball_candidates,
                    best_tracklet["tracklet_points"],
                    frame_width=width,
                    frame_height=height,
                    segment_end_frame=segment_end_frame,
                )
            ),
            "ball_positions": ranked_ball_positions,
            "selected_ball_points": len(best_tracklet["tracklet_points"]),
            "fit_result": fit_result,
        }
    )
    return result


def analyze_video(args) -> dict:
    video_path = Path(args.video_path).expanduser()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video path does not exist: {video_path}")

    output_paths = build_output_paths(
        video_path,
        Path(args.output_dir),
        write_overlay=args.write_overlay,
    )
    calibration_context = load_calibration_context(args.calibration_context)

    model_info = load_diagnostic_model(args)
    ball_model = model_info["model"]
    stump_model = model_info["model"]
    model_path_label = model_info["model_path"]

    stump_class_names = map_model_classes(stump_model)
    ball_class_names = map_model_classes(ball_model)
    print(f"Diagnostic model path used: {model_path_label}")
    print(
        "Diagnostic model class names: "
        f"{json.dumps(model_info['raw_class_names'], sort_keys=True)}"
    )
    print(
        "Diagnostic mapped classes: "
        f"{json.dumps(ball_class_names, sort_keys=True)}"
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Could not open video.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Could not read video width/height.")

    speed_settings = get_analysis_mode_settings(args.speed_mode)
    inference_imgsz = int(args.imgsz or speed_settings.get("yolo_imgsz", 640))
    resize_width = speed_settings.get("resize_width")
    use_roi = bool(speed_settings.get("use_roi", True)) and not args.no_roi
    every_nth_frame = max(1, int(args.every_nth_frame or 1))
    ball_confidence = (
        float(args.ball_confidence)
        if args.ball_confidence is not None
        else float(args.conf)
    )
    full_frame_roi_mode = "roi_enabled" if use_roi else "full_frame_no_roi"
    online_best_tracklet_enabled = should_enable_online_best_tracklet(
        ball_tracking_mode="Balanced",
        speed_mode=speed_settings.get("mode", args.speed_mode),
    )
    defer_overlay_write = bool(args.write_overlay and online_best_tracklet_enabled)

    writer = None
    overlay_frame_buffer: list[dict] = []
    if args.write_overlay and not defer_overlay_write:
        output_paths["overlay_path"].parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_paths["overlay_path"]),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("Could not create debug overlay writer.")

    selector = TrajectoryBallSelector(width, height)
    kalman_tracker = BallKalmanTracker(max_missing_frames=10)
    previous_ball_center = None
    previous_roi_box = None
    locked_stump = None
    missing_ball_frames = 0
    max_missing_ball_frames = 12
    frame_detections = []
    ball_positions = []
    raw_frame_ball_candidates: dict[int, list[dict]] = {}
    rows = []
    rejection_reasons = Counter()

    processed_frames = 0
    selected_ball_points = 0
    total_raw_ball_candidates = 0
    total_raw_stump_candidates = 0
    roi_detected_frames = 0
    local_recovery_frames = 0
    kalman_predicted_frames = 0
    timing = {
        "video_read_time_sec": 0.0,
        "detection_time_sec": 0.0,
        "selection_time_sec": 0.0,
        "annotation_write_time_sec": 0.0,
        "full_frame_detection_time_ms": 0.0,
        "roi_detection_time_ms": 0.0,
        "total_time_sec": 0.0,
    }
    started_at = time.perf_counter()

    try:
        source_frame_index = 0
        while True:
            read_started = time.perf_counter()
            success, frame = capture.read()
            timing["video_read_time_sec"] += time.perf_counter() - read_started
            if not success:
                break

            if source_frame_index % every_nth_frame != 0:
                source_frame_index += 1
                continue

            if args.max_frames is not None and processed_frames >= args.max_frames:
                break

            frame_index = source_frame_index

            detection_ran = should_detect_ball(frame_index, speed_settings)
            inference_frame, detection_scale = resize_frame_for_inference(
                frame,
                resize_width,
            )
            run_stump = should_detect_stump(
                frame_index,
                speed_settings,
                locked_stump,
            )
            detection_result = {}
            raw_ball_detections = []
            raw_stump_detections = []
            ball_detections = []
            stump_detections = []
            source = "skipped"
            roi_box = None
            search_roi = None

            if detection_ran:
                detection_started = time.perf_counter()
                detection_result = run_pitch_roi_detection(
                    inference_frame,
                    stump_model=stump_model,
                    stump_class_names=stump_class_names,
                    confidence=args.conf,
                    imgsz=inference_imgsz,
                    previous_roi=previous_roi_box,
                    ball_model=ball_model,
                    ball_class_names=ball_class_names,
                    use_ensemble=False,
                    ball_confidence=ball_confidence,
                    speed_settings=speed_settings,
                    detect_stump=run_stump,
                    locked_stump_detections=(
                        [locked_stump]
                        if locked_stump and not run_stump
                        else None
                    ),
                    use_roi=use_roi,
                )
                timing["detection_time_sec"] += (
                    time.perf_counter() - detection_started
                )
                timing["full_frame_detection_time_ms"] += float(
                    detection_result.get("full_frame_time_ms", 0.0)
                )
                timing["roi_detection_time_ms"] += float(
                    detection_result.get("roi_time_ms", 0.0)
                )
                source = "primary"
                raw_ball_detections = scale_detections_to_original(
                    detection_result.get("ball_detections", []),
                    detection_scale,
                )
                raw_stump_detections = scale_detections_to_original(
                    detection_result.get("stump_detections", []),
                    detection_scale,
                )
                ball_detections = raw_ball_detections
                stump_detections = raw_stump_detections
                roi_box = scale_box_to_original(
                    detection_result.get("roi_box"),
                    detection_scale,
                )
                if detection_result.get("used_roi") and roi_box is not None:
                    previous_roi_box = detection_result["roi_box"]
                    roi_detected_frames += 1

                if (
                    not ball_detections
                    and speed_settings.get("enable_local_redetection", True)
                ):
                    search_center = (
                        previous_ball_center
                        or kalman_tracker.last_prediction
                    )
                    recovery_started = time.perf_counter()
                    recovery_result = run_local_redetection(
                        inference_frame,
                        search_center,
                        args.conf,
                        inference_imgsz,
                        missing_ball_frames + 1,
                        ball_model=ball_model,
                        ball_class_names=ball_class_names,
                        use_ensemble=False,
                    )
                    timing["detection_time_sec"] += (
                        time.perf_counter() - recovery_started
                    )
                    search_roi = scale_box_to_original(
                        recovery_result.get("search_roi"),
                        detection_scale,
                    )
                    if recovery_result.get("recovered"):
                        source = "local_redetection"
                        ball_detections = scale_detections_to_original(
                            recovery_result.get("ball_detections", []),
                            detection_scale,
                        )
                        raw_ball_detections = ball_detections
                        local_recovery_frames += 1

            if ball_detections:
                raw_frame_ball_candidates[frame_index] = [
                    {
                        "center": detection["center"],
                        "confidence": detection.get("confidence", 0.0),
                        "box": detection.get("box"),
                        "class_name": detection.get("class_name", "ball"),
                    }
                    for detection in ball_detections
                ]

            stump_detections = apply_locked_stump(stump_detections, locked_stump)
            frame_detections.append(
                {
                    "frame_index": frame_index,
                    "ball_detections": ball_detections,
                    "bat_detections": [],
                    "stump_detections": stump_detections,
                }
            )
            initial_frames = speed_settings.get("stump_detect_initial_frames")
            if locked_stump is None and initial_frames:
                if frame_index + 1 >= int(initial_frames):
                    locked_stump = lock_static_stump_detection(
                        frame_detections,
                        int(initial_frames),
                    )

            add_candidate_ids(ball_detections)
            previous_before_selection = previous_ball_center
            diagnostics = []
            selection_started = time.perf_counter()
            main_ball = selector.select(
                ball_detections,
                previous_ball_center,
                kalman_prediction=kalman_tracker.last_prediction,
                diagnostics=diagnostics,
                frame_index=frame_index,
            )
            timing["selection_time_sec"] += time.perf_counter() - selection_started

            if main_ball is not None:
                missing_ball_frames = 0
                previous_ball_center = main_ball["center"]
                kalman_tracker.update(previous_ball_center)
                ball_positions.append(previous_ball_center)
                selected_ball_points += 1
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
                    previous_ball_center = None

            quality = calculate_tracking_quality(
                ball_positions,
                processed_frames + 1,
            )
            tracking_quality = get_tracking_quality_label(
                quality["tracking_rate"],
                quality["interpolated_frames"],
                kalman_predicted_frames,
            )
            frame_rows = build_candidate_rows(
                frame_index=frame_index,
                fps=fps,
                width=width,
                height=height,
                model_path=model_path_label,
                class_names=ball_class_names,
                detection_source=source,
                detection_ran=detection_ran,
                used_roi=bool(detection_result.get("used_roi")),
                roi_box=roi_box,
                raw_ball_detections=ball_detections,
                raw_stump_count=len(raw_stump_detections),
                diagnostics=diagnostics,
                previous_center=previous_before_selection,
                calibration_context=calibration_context,
                accepted_track_count=selector.accepted_point_count,
                tracking_quality=tracking_quality,
            )
            rows.extend(frame_rows)
            for row in frame_rows:
                if row["candidate_rejected"] and row["rejection_reason"]:
                    for reason in str(row["rejection_reason"]).split(";"):
                        if reason:
                            rejection_reasons[reason] += 1

            total_raw_ball_candidates += len(ball_detections)
            total_raw_stump_candidates += len(raw_stump_detections)

            if defer_overlay_write:
                overlay_frame_buffer.append(
                    {
                        "frame": frame.copy(),
                        "frame_index": frame_index,
                        "ball_detections": ball_detections,
                        "stump_detections": stump_detections,
                        "diagnostics": diagnostics,
                        "roi_box": roi_box,
                        "search_roi": search_roi,
                    }
                )
            elif writer is not None:
                annotation_started = time.perf_counter()
                fit_result = fit_delivery_trajectory(
                    [
                        {
                            "frame_index": frame_no,
                            "x": position[0],
                            "y": position[1],
                        }
                        for frame_no, position in zip(
                            selector._accepted_frame_indices,
                            selector.accepted_positions,
                        )
                    ],
                    frame_size=(width, height),
                    fps=fps,
                    calibration_context=calibration_context,
                    delivery_track_terminated_frame=selector.selected_track_end_frame(),
                )
                overlay = draw_debug_overlay(
                    frame,
                    ball_detections,
                    stump_detections,
                    diagnostics,
                    selector.accepted_positions,
                    fit_result=fit_result,
                    roi_box=roi_box,
                    search_roi=search_roi,
                    label_rejections=args.label_rejections,
                )
                writer.write(ensure_frame_writer_size(overlay, width, height))
                timing["annotation_write_time_sec"] += (
                    time.perf_counter() - annotation_started
                )

            processed_frames += 1
            source_frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    timing["total_time_sec"] = time.perf_counter() - started_at

    best_tracklet_info = apply_best_tracklet_post_pass(
        selector=selector,
        raw_frame_ball_candidates=raw_frame_ball_candidates,
        width=width,
        height=height,
        fps=fps,
        speed_mode=speed_settings.get("mode", args.speed_mode),
        ball_positions=ball_positions,
        calibration_context=calibration_context,
    )
    if best_tracklet_info.get("best_tracklet_applied"):
        ball_positions = best_tracklet_info["ball_positions"]
        selected_ball_points = best_tracklet_info["selected_ball_points"]

    final_quality = calculate_tracking_quality(ball_positions, processed_frames)
    final_tracking_quality, _ = resolve_delivery_tracking_quality(
        selector,
        interpolated_frames=final_quality["interpolated_frames"],
        kalman_predicted_frames=kalman_predicted_frames,
    )
    selector_debug = selector.debug_summary(
        final_tracking_quality,
        fps=fps,
    )
    if best_tracklet_info.get("fit_result") is not None:
        fit_result = best_tracklet_info["fit_result"]
    else:
        fit_result = fit_delivery_trajectory(
            [
                {
                    "frame_index": frame_no,
                    "x": position[0],
                    "y": position[1],
                }
                for frame_no, position in zip(
                    selector._accepted_frame_indices,
                    selector.accepted_positions,
                )
            ],
            frame_size=(width, height),
            fps=fps,
            calibration_context=calibration_context,
            delivery_track_terminated_frame=selector.selected_track_end_frame(),
        )
    if best_tracklet_info.get("best_tracklet_applied"):
        selector_debug.update(
            {
                "best_segment_start_frame": best_tracklet_info.get(
                    "best_segment_start_frame"
                ),
                "best_segment_end_frame": best_tracklet_info.get(
                    "best_segment_end_frame"
                ),
                "best_segment_point_count": best_tracklet_info.get(
                    "best_segment_point_count",
                    0,
                ),
                "best_segment_duration_sec": best_tracklet_info.get(
                    "best_segment_duration_sec"
                ),
                "selected_segment_score": best_tracklet_info.get(
                    "best_segment_score"
                ),
                "selected_segment_reason": best_tracklet_info.get(
                    "best_segment_reason"
                ),
            }
        )
    fit_debug_fields = {
            "trajectory_fit_quality": fit_result.get("trajectory_fit_quality"),
            "fitted_trajectory_point_count": len(
                fit_result.get("fitted_trajectory_points", [])
            ),
            "observed_track_point_count": fit_result.get(
                "observed_point_count",
                0,
            ),
            "trajectory_fit_reason": fit_result.get("trajectory_fit_reason"),
            "trajectory_visualization_mode": fit_result.get(
                "trajectory_visualization_mode",
                "hidden",
            ),
    }
    if not best_tracklet_info.get("best_tracklet_applied"):
        fit_debug_fields.update(
            {
                "best_segment_start_frame": fit_result.get(
                    "best_segment_start_frame"
                ),
                "best_segment_end_frame": fit_result.get(
                    "best_segment_end_frame"
                ),
                "best_segment_point_count": fit_result.get(
                    "best_segment_point_count",
                    0,
                ),
                "best_segment_duration_sec": fit_result.get(
                    "best_segment_duration_sec"
                ),
                "selected_segment_score": fit_result.get(
                    "selected_segment_score",
                    0.0,
                ),
                "selected_segment_reason": fit_result.get(
                    "selected_segment_reason",
                    "",
                ),
            }
        )
    selector_debug.update(fit_debug_fields)
    final_calibration_context = build_calibration_context(
        calibration_context,
        frame_detections=frame_detections,
        frame_width=width,
        frame_height=height,
    )
    overlay_status = build_debug_overlay_status(
        final_tracking_quality=final_tracking_quality,
        fit_result=fit_result,
        best_tracklet_info=best_tracklet_info,
        calibration_context=final_calibration_context,
        selector_debug_summary=selector_debug,
    )
    ranked_frame_points = {}
    if best_tracklet_info.get("best_tracklet_applied"):
        ranked_frame_points = {
            int(frame_no): (int(position[0]), int(position[1]))
            for frame_no, position in zip(
                selector._accepted_frame_indices,
                selector.accepted_positions,
            )
        }
    if defer_overlay_write and overlay_frame_buffer:
        output_paths["overlay_path"].parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_paths["overlay_path"]),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("Could not create debug overlay writer.")
        annotation_started = time.perf_counter()
        for item in overlay_frame_buffer:
            overlay = draw_debug_overlay(
                item["frame"],
                item["ball_detections"],
                item["stump_detections"],
                item["diagnostics"],
                selector.accepted_positions,
                fit_result=fit_result,
                roi_box=item["roi_box"],
                search_roi=item["search_roi"],
                frame_index=item["frame_index"],
                label_rejections=args.label_rejections,
                overlay_status=overlay_status,
                ranked_frame_points=ranked_frame_points,
            )
            writer.write(ensure_frame_writer_size(overlay, width, height))
        writer.release()
        timing["annotation_write_time_sec"] += (
            time.perf_counter() - annotation_started
        )
    summary = build_summary(
        video_path=video_path,
        model_path=model_path_label,
        class_names=ball_class_names,
        total_frames=total_frames,
        processed_frames=processed_frames,
        total_raw_ball_candidates=total_raw_ball_candidates,
        selected_ball_points=selected_ball_points,
        rejected_candidate_count=sum(
            1 for row in rows if row["candidate_rejected"]
        ),
        rejection_reasons=rejection_reasons,
        stump_detections_count=total_raw_stump_candidates,
        calibration_context=final_calibration_context,
        final_tracking_quality=final_tracking_quality,
        selector_debug_summary=selector_debug,
        timing=timing,
        roi_detected_frames=roi_detected_frames,
        local_recovery_frames=local_recovery_frames,
        kalman_predicted_frames=kalman_predicted_frames,
        speed_settings=speed_settings,
        confidence_threshold=args.conf,
        ball_confidence_threshold=ball_confidence,
        image_size=inference_imgsz,
        full_frame_roi_mode=full_frame_roi_mode,
        every_nth_frame=every_nth_frame,
        best_tracklet_info=best_tracklet_info,
        overlay_status=overlay_status,
    )

    write_csv(output_paths["csv_path"], rows)
    write_json(output_paths["summary_path"], summary)
    return {
        "csv_path": str(output_paths["csv_path"]),
        "summary_path": str(output_paths["summary_path"]),
        "overlay_path": (
            str(output_paths["overlay_path"])
            if args.write_overlay
            else None
        ),
        "summary": summary,
    }


def build_output_paths(video_path: Path, output_dir: Path, *, write_overlay=False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_stem = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in video_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"ball_tracking_debug_{safe_stem}_{timestamp}"
    paths = {
        "csv_path": base.with_suffix(".csv"),
        "summary_path": base.with_suffix(".json"),
    }
    if write_overlay:
        paths["overlay_path"] = base.with_suffix(".mp4")
    return paths


def load_calibration_context(path):
    if not path:
        return normalize_calibration_context(None)
    with open(Path(path), "r", encoding="utf-8") as json_file:
        value = json.load(json_file)
    if isinstance(value, dict) and "calibration_context" in value:
        value = value["calibration_context"]
    return normalize_calibration_context(value)


def load_diagnostic_model(args):
    checked_paths = []

    if args.model_path:
        explicit_path = resolve_local_model_path(args.model_path)
        checked_paths.append(str(explicit_path))
        model = load_yolo_model_from_path(explicit_path)
        return {
            "model": model,
            "model_path": str(explicit_path),
            "raw_class_names": get_model_names(model),
            "checked_paths": checked_paths,
        }

    default_path = CRICKET_OBJECTS_MODEL_PATH
    checked_paths.append(str(default_path))
    if default_path.is_file():
        try:
            model = load_yolo_model_from_path(default_path)
            return {
                "model": model,
                "model_path": str(default_path),
                "raw_class_names": get_model_names(model),
                "checked_paths": checked_paths,
            }
        except Exception as error:
            checked_paths[-1] = f"{default_path} ({error})"

    registry_path = get_model_path(args.model_key)
    if registry_path is not None:
        checked_paths.append(str(registry_path))
    model = load_detection_model(model_key=args.model_key)
    if model is not None:
        return {
            "model": model,
            "model_path": str(registry_path or args.model_key),
            "raw_class_names": get_model_names(model),
            "checked_paths": checked_paths,
        }

    model_label = (get_model_info(args.model_key) or {}).get(
        "name",
        args.model_key,
    )
    raise RuntimeError(
        "Could not load a diagnostic YOLO model. "
        f"Model key: {model_label}. Checked paths: "
        f"{'; '.join(checked_paths) or '(none)'}"
    )


def resolve_local_model_path(model_path):
    path = Path(model_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_yolo_model_from_path(model_path):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        from ultralytics import YOLO

        return YOLO(str(model_path))
    except Exception as error:
        raise RuntimeError(f"Could not load YOLO model at {model_path}: {error}") from error


def add_candidate_ids(detections):
    for index, detection in enumerate(detections or []):
        detection["_debug_candidate_id"] = index


def scale_box_to_original(box, scale):
    if box is None:
        return None
    try:
        scale_value = float(scale)
    except (TypeError, ValueError):
        scale_value = 1.0
    inverse = 1.0 if scale_value in {0, 1.0} else 1.0 / scale_value
    x1, y1, x2, y2 = box
    return (
        int(x1 * inverse),
        int(y1 * inverse),
        int(x2 * inverse),
        int(y2 * inverse),
    )


def build_candidate_rows(
    *,
    frame_index,
    fps,
    width,
    height,
    model_path,
    class_names,
    detection_source,
    detection_ran,
    used_roi,
    roi_box,
    raw_ball_detections,
    raw_stump_count,
    diagnostics,
    previous_center,
    calibration_context,
    accepted_track_count,
    tracking_quality,
):
    base = {
        "frame_index": frame_index,
        "timestamp_sec": round(frame_index / fps, 4) if fps else 0,
        "frame_width": width,
        "frame_height": height,
        "model_path": model_path,
        "class_names": json.dumps(class_names, sort_keys=True),
        "detection_source": detection_source,
        "detection_ran": bool(detection_ran),
        "used_roi": bool(used_roi),
        "roi_x1": "",
        "roi_y1": "",
        "roi_x2": "",
        "roi_y2": "",
        "raw_ball_candidate_count": len(raw_ball_detections or []),
        "raw_stump_candidate_count": raw_stump_count,
        "accepted_track_count_so_far": accepted_track_count,
        "tracking_quality_so_far": tracking_quality,
    }
    if roi_box is not None:
        base["roi_x1"], base["roi_y1"], base["roi_x2"], base["roi_y2"] = roi_box

    if not raw_ball_detections:
        reason = (
            "detection_skipped_by_speed_mode"
            if not detection_ran
            else "no_raw_ball_candidates"
        )
        return [
            {
                **base,
                "candidate_id": "",
                "class_name": "",
                "confidence": "",
                "bbox_x1": "",
                "bbox_y1": "",
                "bbox_x2": "",
                "bbox_y2": "",
                "center_x": "",
                "center_y": "",
                "candidate_selected": False,
                "candidate_rejected": False,
                "rejection_reason": reason,
                "selection_score": "",
                "reference_x": point_x(previous_center),
                "reference_y": point_y(previous_center),
                "predicted_x": "",
                "predicted_y": "",
                "distance_from_previous": "",
                "distance_from_prediction": "",
                "inside_roi_or_corridor_if_available": "",
            }
        ]

    diagnostics_by_id = {}
    for item in diagnostics or []:
        diagnostics_by_id[item.get("candidate_id")] = item

    rows = []
    for index, detection in enumerate(raw_ball_detections):
        candidate_id = detection.get("_debug_candidate_id", index)
        diagnostic = diagnostics_by_id.get(candidate_id, {})
        box = detection.get("box") or detection.get("bbox") or ["", "", "", ""]
        center = detection.get("center") or ("", "")
        predicted_center = diagnostic.get("predicted_center")
        reference_center = diagnostic.get("reference_center") or previous_center
        rows.append(
            {
                **base,
                "candidate_id": candidate_id,
                "class_name": detection.get("class_name", "ball"),
                "confidence": round(float(detection.get("confidence", 0)), 6),
                "bbox_x1": box[0],
                "bbox_y1": box[1],
                "bbox_x2": box[2],
                "bbox_y2": box[3],
                "center_x": point_x(center),
                "center_y": point_y(center),
                "candidate_selected": bool(diagnostic.get("selected", False)),
                "candidate_rejected": bool(diagnostic.get("rejected", False)),
                "rejection_reason": diagnostic.get("rejection_reason", ""),
                "selection_score": numeric_or_blank(diagnostic.get("score")),
                "reference_x": point_x(reference_center),
                "reference_y": point_y(reference_center),
                "predicted_x": point_x(predicted_center),
                "predicted_y": point_y(predicted_center),
                "distance_from_previous": distance_or_blank(
                    center,
                    previous_center,
                ),
                "distance_from_prediction": distance_or_blank(
                    center,
                    predicted_center,
                ),
                "inside_roi_or_corridor_if_available": inside_roi_or_corridor(
                    center,
                    roi_box,
                    calibration_context,
                ),
            }
        )
    return rows


def build_summary(
    *,
    video_path,
    model_path,
    class_names,
    total_frames,
    processed_frames,
    total_raw_ball_candidates,
    selected_ball_points,
    rejected_candidate_count,
    rejection_reasons,
    stump_detections_count,
    calibration_context,
    final_tracking_quality,
    selector_debug_summary,
    timing,
    roi_detected_frames,
    local_recovery_frames,
    kalman_predicted_frames,
    speed_settings,
    confidence_threshold,
    ball_confidence_threshold,
    image_size,
    full_frame_roi_mode,
    every_nth_frame,
    best_tracklet_info=None,
    overlay_status=None,
):
    best_tracklet_info = best_tracklet_info or {}
    overlay_status = overlay_status or {}
    return {
        "video_path": str(video_path),
        "model_path_used": model_path,
        "model_class_names": class_names,
        "confidence_threshold_used": confidence_threshold,
        "ball_candidate_confidence_used": ball_confidence_threshold,
        "image_size_used": image_size,
        "full_frame_roi_mode": full_frame_roi_mode,
        "every_nth_frame": every_nth_frame,
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "total_raw_ball_candidates": total_raw_ball_candidates,
        "selected_ball_points": selected_ball_points,
        "rejected_candidate_count": rejected_candidate_count,
        "top_rejection_reasons": dict(rejection_reasons.most_common(10)),
        "stump_detections_count": stump_detections_count,
        "calibration_quality": calibration_context.get("calibration_quality"),
        "calibration_confidence": calibration_context.get("calibration_score"),
        "calibration_context": calibration_context,
        "final_tracking_quality": final_tracking_quality,
        "trajectory_fit_quality": selector_debug_summary.get(
            "trajectory_fit_quality"
        ),
        "trajectory_visualization_mode": selector_debug_summary.get(
            "trajectory_visualization_mode",
            "hidden",
        ),
        "fitted_trajectory_point_count": selector_debug_summary.get(
            "fitted_trajectory_point_count",
            0,
        ),
        "observed_track_point_count": selector_debug_summary.get(
            "observed_track_point_count",
            0,
        ),
        "best_segment_start_frame": selector_debug_summary.get(
            "best_segment_start_frame"
        ),
        "best_segment_end_frame": selector_debug_summary.get(
            "best_segment_end_frame"
        ),
        "best_segment_point_count": selector_debug_summary.get(
            "best_segment_point_count",
            0,
        ),
        "best_segment_duration_sec": selector_debug_summary.get(
            "best_segment_duration_sec"
        ),
        "selected_segment_score": selector_debug_summary.get(
            "selected_segment_score",
            0.0,
        ),
        "selected_segment_reason": selector_debug_summary.get(
            "selected_segment_reason",
            "",
        ),
        "online_best_tracklet_enabled": best_tracklet_info.get(
            "online_best_tracklet_enabled",
            False,
        ),
        "best_tracklet_applied": best_tracklet_info.get("best_tracklet_applied", False),
        "best_tracklet_fallback_reason": best_tracklet_info.get("fallback_reason"),
        "best_segment_score": best_tracklet_info.get("best_segment_score"),
        "best_segment_reason": best_tracklet_info.get("best_segment_reason"),
        "candidate_segment_count": best_tracklet_info.get("candidate_segment_count", 0),
        "rejected_static_segment_count": best_tracklet_info.get(
            "rejected_static_segment_count",
            0,
        ),
        "online_selector_original_start_frame": best_tracklet_info.get(
            "online_selector_original_start_frame"
        ),
        "online_selector_original_end_frame": best_tracklet_info.get(
            "online_selector_original_end_frame"
        ),
        "online_selector_after_track_terminated_count": best_tracklet_info.get(
            "online_selector_after_track_terminated_count",
            0,
        ),
        "extension_enabled": best_tracklet_info.get("extension_enabled", False),
        "extension_applied": best_tracklet_info.get("extension_applied", False),
        "backward_extension_points": best_tracklet_info.get(
            "backward_extension_points",
            0,
        ),
        "forward_extension_points": best_tracklet_info.get(
            "forward_extension_points",
            0,
        ),
        "extended_segment_start_frame": best_tracklet_info.get(
            "extended_segment_start_frame"
        ),
        "extended_segment_end_frame": best_tracklet_info.get(
            "extended_segment_end_frame"
        ),
        "extended_segment_point_count": best_tracklet_info.get(
            "extended_segment_point_count",
            0,
        ),
        "extension_rejection_reasons": best_tracklet_info.get(
            "extension_rejection_reasons",
            {},
        ),
        "extension_fallback_reason": best_tracklet_info.get(
            "extension_fallback_reason"
        ),
        "extension_preserved_original_segment": best_tracklet_info.get(
            "extension_preserved_original_segment",
            not best_tracklet_info.get("extension_applied", False),
        ),
        "extension_fit_delta": best_tracklet_info.get("extension_fit_delta", 0),
        "trajectory_fit_quality_after_extension": best_tracklet_info.get(
            "trajectory_fit_quality_after_extension"
        ),
        "overlay_status": overlay_status,
        "selector_debug_summary": selector_debug_summary,
        "roi_detected_frames": roi_detected_frames,
        "local_recovery_frames": local_recovery_frames,
        "kalman_predicted_frames": kalman_predicted_frames,
        "speed_mode": speed_settings.get("mode"),
        "timing_breakdown": {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in timing.items()
        },
    }


def build_debug_overlay_status(
    *,
    final_tracking_quality: str,
    fit_result: dict,
    best_tracklet_info: dict | None = None,
    calibration_context: dict | None = None,
    selector_debug_summary: dict | None = None,
) -> dict:
    """Build final overlay/report labels from ranked fit, not stale online state."""
    best_tracklet_info = best_tracklet_info or {}
    selector_debug_summary = selector_debug_summary or {}
    fit_quality = fit_result.get("trajectory_fit_quality") or "Poor"
    tracking_quality = final_tracking_quality or "Poor"

    if best_tracklet_info.get("extension_applied"):
        segment_start = best_tracklet_info.get("extended_segment_start_frame")
        segment_end = best_tracklet_info.get("extended_segment_end_frame")
    elif best_tracklet_info.get("best_tracklet_applied"):
        segment_start = best_tracklet_info.get("best_segment_start_frame")
        segment_end = best_tracklet_info.get("best_segment_end_frame")
    else:
        segment_start = fit_result.get("best_segment_start_frame")
        segment_end = fit_result.get("best_segment_end_frame")

    segment_text = None
    if segment_start is not None and segment_end is not None:
        segment_text = f"{int(segment_start)}-{int(segment_end)}"

    calibration_quality = (calibration_context or {}).get("calibration_quality")
    calibration_disabled = calibration_quality in {None, "", "Disabled"}
    line = "Unknown" if calibration_disabled else fit_result.get("estimated_line") or "Unknown"
    length = "Unknown" if calibration_disabled else fit_result.get("estimated_length") or "Unknown"
    bounce = "Unknown" if calibration_disabled else fit_result.get("estimated_bounce") or "Unknown"

    return {
        "tracking_quality": tracking_quality,
        "trajectory_fit_quality": fit_quality,
        "segment_start_frame": segment_start,
        "segment_end_frame": segment_end,
        "segment_text": segment_text,
        "track_label": f"Track: {tracking_quality}",
        "fit_label": f"Fit: {fit_quality}",
        "segment_label": (
            f"Segment: {segment_text}" if segment_text is not None else None
        ),
        "line": line,
        "length": length,
        "bounce": bounce,
        "calibration_context": calibration_context,
        "online_selector_tracking_quality": selector_debug_summary.get(
            "tracking_quality"
        ),
        "best_tracklet_applied": bool(
            best_tracklet_info.get("best_tracklet_applied")
        ),
    }


def draw_debug_status_labels(frame, overlay_status: dict | None) -> None:
    """Draw ranked debug status lines below the metric cards."""
    overlay_status = overlay_status or {}
    labels = [
        overlay_status.get("track_label"),
        overlay_status.get("fit_label"),
        overlay_status.get("segment_label"),
    ]
    y = 112
    for label in labels:
        if not label:
            continue
        draw_label(frame, label, 16, y, (30, 30, 30))
        y += 28


def draw_ranked_tracklet_point(frame, point) -> None:
    x, y = int(point[0]), int(point[1])
    cv2.circle(frame, (x, y), 6, (40, 220, 40), -1)
    cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)


def draw_rejected_candidate_debug(
    frame,
    detection,
    diagnostic,
    *,
    frame_height: int,
) -> None:
    """Draw dim rejection markers below the main trajectory layer."""
    x1, y1, x2, y2 = detection["box"]
    center_x, center_y = detection["center"]
    color = (90, 90, 200)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    cv2.circle(frame, (center_x, center_y), 2, color, -1)
    label = str(diagnostic.get("rejection_reason") or "rejected")[:24]
    label_y = min(max(y2 + 18, 30), max(frame_height - 8, 30))
    draw_label(frame, label, x1, label_y, color)


def draw_debug_overlay(
    frame,
    ball_detections,
    stump_detections,
    diagnostics,
    accepted_positions,
    fit_result,
    roi_box,
    search_roi,
    *,
    frame_index=None,
    label_rejections=False,
    overlay_status=None,
    ranked_frame_points=None,
):
    overlay = frame.copy()
    overlay_status = overlay_status or {}
    ranked_frame_points = ranked_frame_points or {}
    best_tracklet_applied = bool(overlay_status.get("best_tracklet_applied"))
    tracking_quality = overlay_status.get("tracking_quality") or "Poor"
    frame_height = overlay.shape[0]
    if roi_box is not None:
        draw_pitch_roi(overlay, roi_box)
    if search_roi is not None:
        draw_search_roi(overlay, search_roi)

    diagnostics_by_id = {
        item.get("candidate_id"): item
        for item in diagnostics or []
    }

    if best_tracklet_applied and label_rejections:
        for index, detection in enumerate(ball_detections or []):
            diagnostic = diagnostics_by_id.get(
                detection.get("_debug_candidate_id", index),
                {},
            )
            if diagnostic.get("rejected"):
                draw_rejected_candidate_debug(
                    overlay,
                    detection,
                    diagnostic,
                    frame_height=frame_height,
                )
    elif not best_tracklet_applied:
        for index, detection in enumerate(ball_detections or []):
            candidate_id = detection.get("_debug_candidate_id", index)
            diagnostic = diagnostics_by_id.get(candidate_id, {})
            if diagnostic.get("rejected"):
                if label_rejections:
                    draw_rejected_candidate_debug(
                        overlay,
                        detection,
                        diagnostic,
                        frame_height=frame_height,
                    )
                continue
            x1, y1, x2, y2 = detection["box"]
            center_x, center_y = detection["center"]
            if diagnostic.get("selected"):
                color = (40, 220, 40)
                label = f"selected {detection.get('confidence', 0):.2f}"
                thickness = 3
            else:
                if not label_rejections:
                    continue
                color = (0, 220, 255)
                label = f"raw {detection.get('confidence', 0):.2f}"
                thickness = 2
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
            cv2.circle(overlay, (center_x, center_y), 4, color, -1)
            if diagnostic.get("selected") or label_rejections:
                draw_label(overlay, label[:32], x1, y1, color)

    for detection in stump_detections or []:
        box = detection.get("box") or detection.get("bbox")
        if not box:
            continue
        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 140, 0), 2)

    accepted_track = [
        (int(point[0]), int(point[1]))
        for point in (accepted_positions or [])[-35:]
    ]
    draw_trajectory_lines(
        overlay,
        accepted_track,
        color=(40, 220, 40),
        thickness=3,
    )
    draw_fitted_trajectory_overlay(
        overlay,
        observed_points=fit_result.get("observed_trajectory_points", []),
        fitted_points=fit_result.get("fitted_trajectory_points", []),
        visualization_mode=fit_result.get("trajectory_visualization_mode", "hidden"),
        trajectory_quality=tracking_quality,
        tracking_quality=tracking_quality,
        line=overlay_status.get("line"),
        length=overlay_status.get("length"),
        calibration_context=overlay_status.get("calibration_context"),
    )
    if best_tracklet_applied and frame_index is not None:
        ranked_point = ranked_frame_points.get(int(frame_index))
        if ranked_point is not None:
            draw_ranked_tracklet_point(overlay, ranked_point)
    draw_debug_status_labels(overlay, overlay_status)
    return overlay


def inside_roi_or_corridor(center, roi_box, calibration_context):
    if not center:
        return ""
    if roi_box is not None:
        return inside_box(center, roi_box)
    corridor = (calibration_context or {}).get("pitch_corridor") or {}
    corridor_box = corridor.get("bbox")
    if corridor_box:
        return inside_box(center, corridor_box)
    return ""


def inside_box(center, box):
    if center is None or box is None:
        return ""
    x, y = center
    x1, y1, x2, y2 = box
    return bool(float(x1) <= float(x) <= float(x2) and float(y1) <= float(y) <= float(y2))


def distance_or_blank(point_a, point_b):
    if point_a is None or point_b is None:
        return ""
    try:
        return round(hypot(float(point_a[0]) - float(point_b[0]), float(point_a[1]) - float(point_b[1])), 4)
    except (TypeError, ValueError, IndexError):
        return ""


def point_x(point):
    if point is None or point == "":
        return ""
    try:
        return point[0]
    except (TypeError, IndexError):
        return ""


def point_y(point):
    if point is None or point == "":
        return ""
    try:
        return point[1]
    except (TypeError, IndexError):
        return ""


def numeric_or_blank(value):
    if value is None:
        return ""
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return ""


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        result = analyze_video(args)
    except Exception as error:
        print(f"Ball-tracking diagnostic failed: {error}", file=sys.stderr)
        return 2

    print("Ball-tracking diagnostic complete:")
    print(json.dumps({key: value for key, value in result.items() if key != "summary"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
