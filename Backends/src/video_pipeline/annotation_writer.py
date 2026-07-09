"""Processed-video and review-frame annotation helpers."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

from Backends.src.config.paths import REVIEW_FRAMES_DIR
from Backends.src.utils.cv2_loader import cv2

REVIEW_FRAMES_CSV = REVIEW_FRAMES_DIR / "review_frames.csv"


def draw_label(frame, text, x, y, color):
    y = max(y, 25)
    label_width = len(text) * 10 + 10
    cv2.rectangle(frame, (x, y - 25), (x + label_width, y), color, -1)
    cv2.putText(
        frame,
        text,
        (x + 5, y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )


def draw_pitch_roi(frame, roi_box):
    if roi_box is None:
        return
    x1, y1, x2, y2 = roi_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 80), 2)
    draw_label(frame, "Pitch ROI", x1, y1, (40, 160, 40))


def draw_search_roi(frame, roi_box):
    if roi_box is None:
        return
    x1, y1, x2, y2 = roi_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 80, 220), 2)
    draw_label(frame, "Recovery ROI", x1, y1, (150, 40, 130))


def _calibration_point_to_xy(point):
    try:
        return int(round(float(point["x"]))), int(round(float(point["y"])))
    except (TypeError, ValueError, KeyError):
        return None


def _draw_calibration_line(frame, line, color, thickness=2):
    if not line:
        return False
    start = _calibration_point_to_xy(line.get("start"))
    end = _calibration_point_to_xy(line.get("end"))
    if start is None or end is None:
        return False
    cv2.line(frame, start, end, color, thickness)
    return True


def draw_replay_calibration_overlay(frame, replay_calibration_report):
    """Draw manual replay calibration geometry when available."""
    report = replay_calibration_report or {}
    geometry = report.get("pitch_geometry") or {}
    if not report.get("available") or not geometry.get("available"):
        return False

    corridor = geometry.get("pitch_corridor") or []
    corridor_points = [_calibration_point_to_xy(point) for point in corridor]
    corridor_points = [point for point in corridor_points if point is not None]
    if len(corridor_points) >= 3:
        for index, point in enumerate(corridor_points):
            cv2.line(frame, point, corridor_points[(index + 1) % len(corridor_points)], (255, 180, 80), 1)

    _draw_calibration_line(frame, geometry.get("near_wicket_line"), (255, 200, 120), 1)
    _draw_calibration_line(frame, geometry.get("far_wicket_line"), (255, 200, 120), 1)
    line_drawn = _draw_calibration_line(frame, geometry.get("stump_line"), (255, 80, 0), 3)
    if line_drawn:
        start = _calibration_point_to_xy((geometry.get("stump_line") or {}).get("start")) or (12, 92)
        draw_label(frame, "Calibrated stump line", start[0], start[1], (180, 80, 0))
    draw_label(
        frame,
        "Replay calibration: single-camera estimated geometry",
        12,
        92,
        (120, 80, 0),
    )
    return line_drawn


def draw_ball_detections(frame, ball_detections):
    for detection in ball_detections or []:
        x1, y1, x2, y2 = detection.get("bbox") or detection["box"]
        center_x, center_y = detection["center"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.circle(frame, (center_x, center_y), 4, (0, 255, 255), -1)
        draw_label(
            frame,
            f"ball {detection['confidence']:.2f}",
            x1,
            y1,
            (0, 180, 180),
        )


def draw_clean_ball_markers(frame, ball_detections):
    """Draw minimal ball markers for the clean overlay mode."""
    for detection in ball_detections or []:
        center = detection.get("center")
        if center is None:
            box = detection.get("bbox") or detection.get("box")
            if box is None or len(box) < 4:
                continue
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)
        else:
            center_x, center_y = int(center[0]), int(center[1])
        cv2.circle(frame, (center_x, center_y), 6, (0, 255, 255), 2)
        cv2.circle(frame, (center_x, center_y), 3, (0, 255, 255), -1)


def draw_clean_stump_markers(frame, stump_detections):
    """Draw stump boxes without debug labels."""
    for detection in stump_detections or []:
        box = detection.get("bbox") or detection.get("box")
        if box is None or len(box) < 4:
            continue
        x1, y1, x2, y2 = (int(value) for value in box[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 140, 0), 2)


def draw_trajectory_lines(frame, trajectory_points, *, color=(0, 255, 255), thickness=3):
    """Draw contiguous polyline segments (no validation). Prefer draw_safe_trajectory_lines."""
    points = [point for point in (trajectory_points or []) if point is not None]
    for index in range(1, len(points)):
        cv2.line(frame, points[index - 1], points[index], color, thickness)


def draw_safe_trajectory_lines(
    frame,
    trajectory_points,
    *,
    frame_size=None,
    pitch_roi=None,
    stump_context=None,
    impact_info=None,
    color=(0, 255, 255),
    projected_color=(0, 165, 255),
    thickness=3,
    show_uncertain_label=True,
    prepared=None,
):
    """Validate cricket path, then draw only safe pre-contact (+ optional projection) segments.

    Ball markers / analysis results are unchanged — this protects drawing only.
    """
    from Backends.src.cricket_path_validity import prepare_safe_trajectory_for_draw

    if prepared is None:
        if frame_size is None and frame is not None and hasattr(frame, "shape"):
            height, width = frame.shape[:2]
            frame_size = {"width": int(width), "height": int(height)}
        prepared = prepare_safe_trajectory_for_draw(
            trajectory_points,
            frame_size=frame_size,
            pitch_roi=pitch_roi,
            stump_context=stump_context,
            impact_info=impact_info,
        )

    if prepared.get("draw_allowed"):
        for segment in prepared.get("draw_segments") or []:
            draw_trajectory_lines(frame, segment, color=color, thickness=thickness)
        for segment in prepared.get("projected_draw_segments") or []:
            # Orange dashed-style projection vs cyan observed pre-contact path.
            draw_trajectory_lines(
                frame,
                segment,
                color=projected_color,
                thickness=max(2, thickness - 1),
            )
        if prepared.get("projection_used"):
            draw_label(frame, "Projected continuation (no bat contact)", 12, 28, (0, 120, 200))
        elif prepared.get("impact_frame") is not None:
            draw_label(frame, "Pre-contact delivery path", 12, 28, (0, 160, 160))
    elif show_uncertain_label and frame is not None:
        # Only label when there were enough raw points but validity failed.
        raw_count = len([p for p in (trajectory_points or []) if p is not None])
        quality = (prepared.get("quality") or "").lower()
        if raw_count >= 5 and quality in {"poor", "unavailable"}:
            # ponytail: small corner label is enough when path fails validity.
            draw_label(frame, "Trajectory uncertain", 12, 28, (40, 40, 180))

    return prepared


def _draw_dashed_line(frame, start, end, color, thickness=2, dash_px=8, gap_px=6):
    x1, y1 = start
    x2, y2 = end
    length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if length < 1:
        return
    step = dash_px + gap_px
    for offset in range(0, int(length), step):
        t0 = offset / length
        t1 = min((offset + dash_px) / length, 1.0)
        p0 = (int(x1 + (x2 - x1) * t0), int(y1 + (y2 - y1) * t0))
        p1 = (int(x1 + (x2 - x1) * t1), int(y1 + (y2 - y1) * t1))
        cv2.line(frame, p0, p1, color, thickness)


def _physics_points_to_xy(points):
    result = []
    for point in points or []:
        try:
            result.append((int(round(float(point["x"]))), int(round(float(point["y"])))))
        except (TypeError, ValueError, KeyError):
            continue
    return result


def draw_physics_trajectory_overlay(
    frame,
    physics_report,
    *,
    color=(0, 255, 255),
    projected_color=(0, 165, 255),
    thickness=3,
):
    """Draw the physics-assisted delivery layer: solid pre-impact fit, dashed projection.

    Returns True when a trusted physics path was drawn.
    """
    from Backends.src.physics_trajectory import PROJECTED_PATH_NOTE

    report = physics_report or {}
    quality = report.get("physics_quality") or "Unavailable"
    fitted = _physics_points_to_xy(report.get("fitted_delivery_path"))

    if quality not in {"Good", "Partial"} or len(fitted) < 2:
        if len(report.get("pre_impact_path") or []) >= 5:
            draw_label(frame, "Trajectory uncertain", 12, 28, (40, 40, 180))
        return False

    draw_trajectory_lines(frame, fitted, color=color, thickness=thickness)

    projected = _physics_points_to_xy(report.get("projected_path"))
    if len(projected) >= 2:
        # Dashed + connected to the last fitted point so the estimate reads as separate.
        for prev, nxt in zip([fitted[-1]] + projected[:-1], projected):
            _draw_dashed_line(frame, prev, nxt, projected_color, max(2, thickness - 1))
        draw_label(frame, PROJECTED_PATH_NOTE, 12, 56, (0, 120, 200))

    impact = report.get("impact") or {}
    impact_point = impact.get("impact_point")
    if isinstance(impact_point, dict) and impact.get("impact_detected"):
        _draw_event_marker(frame, (impact_point.get("x"), impact_point.get("y")), "Impact", (0, 220, 255))
    return True


def add_physics_trajectory_overlay_to_video(video_path, physics_report):
    """Rewrite a processed delivery video with the physics-assisted trajectory layer.

    No-op unless the physics fit is Good/Partial — never draws a misleading path.
    """
    report = physics_report or {}
    if report.get("physics_quality") not in {"Good", "Partial"}:
        return Path(video_path)
    if len(report.get("fitted_delivery_path") or []) < 2:
        return Path(video_path)

    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return video_path

    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temp_path = video_path.with_name(f"{video_path.stem}_physics{video_path.suffix}")
    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        return video_path

    while True:
        success, frame = capture.read()
        if not success:
            break
        draw_physics_trajectory_overlay(frame, report)
        writer.write(ensure_frame_writer_size(frame, width, height))
    capture.release()
    writer.release()
    temp_path.replace(video_path)
    return video_path


def delivery_overlay_metrics(
    calibration_context,
    *,
    line="Unknown",
    length="Unknown",
    bounce_point=None,
):
    """Return line/length/bounce labels for the clean delivery overlay."""
    if not (calibration_context or {}).get("enabled"):
        return "Unknown", "Unknown", "Unknown"
    bounce = "Found" if bounce_point is not None else "Unknown"
    return line or "Unknown", length or "Unknown", bounce


def draw_fitted_trajectory_overlay(
    frame,
    *,
    observed_points=None,
    fitted_points=None,
    visualization_mode="hidden",
    trajectory_quality=None,
    fit_quality=None,
    bounce_point=None,
    impact_point=None,
    line=None,
    length=None,
    speed=None,
    tracking_quality=None,
    calibration_context=None,
    pitch_roi=None,
    stump_context=None,
    impact_info=None,
):
    """Draw the polished CricVision trajectory layer.

    Poor/hidden fits keep only small observed markers; they do not draw a
    confident curve. Fitted curves go through path-validity safe drawing.
    """
    quality = trajectory_quality or (
        "Good" if visualization_mode == "full_fit" else
        "Partial" if visualization_mode == "partial_fit" else
        "Poor"
    )
    _draw_pitch_corridor_from_context(frame, calibration_context)
    for point in observed_points or []:
        x, y = int(point[0]), int(point[1])
        cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 5, (0, 0, 255), 1)

    if visualization_mode in {"partial_fit", "full_fit"} and fitted_points:
        # ponytail: reuse path-validity gate instead of raw polyline for merge safety.
        draw_safe_trajectory_lines(
            frame,
            fitted_points,
            pitch_roi=pitch_roi,
            stump_context=stump_context,
            impact_info=impact_info,
            color=(0, 0, 255),
            thickness=3 if visualization_mode == "partial_fit" else 4,
            show_uncertain_label=False,
        )
    if bounce_point is not None:
        _draw_event_marker(frame, bounce_point, "Bounce", (0, 80, 255))
    if impact_point is not None:
        _draw_event_marker(frame, impact_point, "Impact", (0, 220, 255))
    track_quality = tracking_quality or quality
    _draw_tracking_confidence(
        frame,
        track_quality,
        fit_quality or quality,
    )
    overlay_line, overlay_length, overlay_bounce = delivery_overlay_metrics(
        calibration_context,
        line=line,
        length=length,
        bounce_point=bounce_point,
    )
    cards_y = 90 if (fit_quality or quality) else 58
    _draw_metric_cards(
        frame,
        speed=speed,
        line=overlay_line,
        length=overlay_length,
        bounce=overlay_bounce,
        tracking_quality=track_quality,
        y=cards_y,
    )


def _draw_event_marker(frame, point, label, color):
    try:
        x, y = int(point[0]), int(point[1])
    except (TypeError, ValueError, IndexError):
        return
    cv2.circle(frame, (x, y), 8, color, -1)
    cv2.circle(frame, (x, y), 14, (255, 255, 255), 2)
    draw_label(frame, label, x + 10, y - 10, color)


def _draw_tracking_confidence(frame, track_quality, fit_quality=None):
    label = str(track_quality or "Poor")
    color = {
        "Good": (40, 180, 40),
        "Partial": (0, 160, 255),
        "Medium": (0, 160, 255),
        "Poor": (40, 40, 220),
        "None": (40, 40, 220),
    }.get(label, (0, 160, 255))
    draw_label(frame, f"Track: {label}", 16, 34, color)
    if fit_quality:
        fit_label = str(fit_quality)
        fit_color = {
            "Good": (40, 180, 40),
            "Partial": (0, 160, 255),
            "Medium": (0, 160, 255),
            "Poor": (40, 40, 220),
        }.get(fit_label, (0, 160, 255))
        draw_label(frame, f"Fit: {fit_label}", 16, 62, fit_color)


def _draw_metric_cards(
    frame,
    *,
    speed=None,
    line=None,
    length=None,
    bounce=None,
    tracking_quality=None,
    y=58,
):
    cards = [
        ("Speed", speed if speed not in {None, ""} else "N/A"),
        ("Line", line or "Unknown"),
        ("Length", length or "Unknown"),
        ("Bounce", bounce or "Unknown"),
        ("Tracking", tracking_quality or "Unknown"),
    ]
    x, start_y = 16, y
    card_width, card_height = 124, 42
    gap = 8
    for index, (title, value) in enumerate(cards):
        x1 = x + index * (card_width + gap)
        y1 = start_y
        x2 = x1 + card_width
        y2 = y1 + card_height
        cv2.rectangle(frame, (x1, y1), (x2, y2), (18, 18, 18), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1)
        cv2.putText(
            frame,
            title,
            (x1 + 8, y1 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (190, 190, 190),
            1,
        )
        cv2.putText(
            frame,
            str(value)[:16],
            (x1 + 8, y1 + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
        )


def _draw_pitch_corridor_from_context(frame, calibration_context):
    corridor = (calibration_context or {}).get("pitch_corridor") or {}
    polygon = corridor.get("polygon") or []
    if len(polygon) < 4:
        return
    points = []
    for point in polygon:
        try:
            points.append((int(point[0]), int(point[1])))
        except (TypeError, ValueError, IndexError):
            return
    for index in range(len(points)):
        cv2.line(
            frame,
            points[index],
            points[(index + 1) % len(points)],
            (80, 255, 80),
            1,
        )


def ensure_frame_writer_size(frame, width, height):
    """Resize a frame when annotation output does not match the writer size."""
    if frame is None:
        return frame
    frame_height, frame_width = frame.shape[:2]
    if frame_width == width and frame_height == height:
        return frame
    return cv2.resize(frame, (width, height))


def validate_processed_video_path(video_path):
    """Validate that a processed video can be previewed in the UI."""
    path = Path(str(video_path)) if video_path else None
    result = {
        "valid": False,
        "exists": False,
        "file_size": 0,
        "width": 0,
        "height": 0,
        "error": "",
        "can_preview": False,
        "path": str(path) if path else "",
    }
    if path is None or not path.is_file():
        result["error"] = "Processed video file is missing."
        return result

    result["exists"] = True
    result["file_size"] = int(path.stat().st_size)
    if result["file_size"] <= 0:
        result["error"] = "Processed video file is empty."
        return result

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        result["error"] = "Processed video could not be opened."
        return result

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    success, frame = capture.read()
    capture.release()
    if not success or frame is None:
        result["error"] = "Processed video has no readable frames."
        return result

    frame_height, frame_width = frame.shape[:2]
    if width <= 0 or height <= 0:
        width, height = frame_width, frame_height
    if width <= 0 or height <= 0:
        result["error"] = "Processed video has invalid dimensions."
        return result

    result["width"] = width
    result["height"] = height
    result["valid"] = True
    result["can_preview"] = True
    return result


def _draw_box_detections(frame, detections, color, label):
    for detection in detections or []:
        box = detection.get("bbox") or detection.get("box")
        if box is None or len(box) < 4:
            continue
        x1, y1, x2, y2 = (int(value) for value in box[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_label(frame, label, x1, y1, color)


def write_annotated_video(
    frames,
    output_path,
    *,
    fps=25,
    frame_detections=None,
    enabled=True,
):
    """Write tiny or production frames with optional shared-timeline overlays."""
    if not enabled:
        return None

    if frames is None:
        return None
    frames = list(frames)
    if not frames:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        return None

    detections_by_frame = {}
    if isinstance(frame_detections, dict):
        detections_by_frame = frame_detections
    else:
        for fallback_index, item in enumerate(frame_detections or []):
            if isinstance(item, dict):
                detections_by_frame[int(item.get("frame_index", fallback_index))] = item

    try:
        for frame_index, raw_frame in enumerate(frames):
            frame = raw_frame.copy()
            detections = detections_by_frame.get(frame_index, {})
            draw_ball_detections(frame, detections.get("ball_detections"))
            _draw_box_detections(
                frame,
                detections.get("bat_detections"),
                (0, 200, 0),
                "bat",
            )
            _draw_box_detections(
                frame,
                detections.get("stump_detections"),
                (255, 100, 0),
                "stump",
            )
            writer.write(ensure_frame_writer_size(frame, width, height))
    finally:
        writer.release()

    if not output_path.is_file() or output_path.stat().st_size == 0:
        return None
    return output_path


def convert_to_browser_mp4(input_path, output_path):
    import imageio_ffmpeg

    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file() or input_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Processed video source is missing or empty: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
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
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0 or not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError(
            "Browser MP4 conversion failed. Download the raw processed video instead."
        )
    validation = validate_processed_video_path(output_path)
    if not validation["valid"]:
        raise RuntimeError(validation["error"] or "Converted video failed validation.")
    return output_path


def add_delivery_trajectory_overlay_to_video(
    video_path,
    *,
    trajectory_fit_result,
    overall_tracking_quality="Poor",
    estimated_line="Unknown",
    estimated_length="Unknown",
    estimated_bounce_point=None,
    calibration_context=None,
):
    """Rewrite a processed delivery video with the final fitted trajectory layer."""
    fit_result = trajectory_fit_result or {}
    visualization_mode = fit_result.get("trajectory_visualization_mode", "hidden")
    fitted_points = fit_result.get("fitted_trajectory_points") or []
    observed_points = fit_result.get("observed_trajectory_points") or []
    if visualization_mode == "hidden" and not fitted_points and not observed_points:
        return Path(video_path)

    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return video_path

    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temp_path = video_path.with_name(
        f"{video_path.stem}_trajectory{video_path.suffix}"
    )
    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        return video_path

    fit_quality = fit_result.get("trajectory_fit_quality") or "Poor"
    overlay_line, overlay_length, _overlay_bounce = delivery_overlay_metrics(
        calibration_context,
        line=estimated_line,
        length=estimated_length,
        bounce_point=estimated_bounce_point,
    )
    while True:
        success, frame = capture.read()
        if not success:
            break
        draw_fitted_trajectory_overlay(
            frame,
            observed_points=observed_points,
            fitted_points=fitted_points,
            visualization_mode=visualization_mode,
            trajectory_quality=fit_quality,
            fit_quality=fit_quality,
            bounce_point=estimated_bounce_point,
            line=overlay_line,
            length=overlay_length,
            tracking_quality=overall_tracking_quality,
            calibration_context=calibration_context,
        )
        writer.write(ensure_frame_writer_size(frame, width, height))
    capture.release()
    writer.release()
    temp_path.replace(video_path)
    return video_path


def add_replay_calibration_overlay_to_video(video_path, replay_calibration_report):
    """Rewrite a processed video with manual replay calibration geometry."""
    report = replay_calibration_report or {}
    if not report.get("available"):
        return Path(video_path)

    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return video_path

    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temp_path = video_path.with_name(f"{video_path.stem}_replay_calibration{video_path.suffix}")
    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        return video_path

    while True:
        success, frame = capture.read()
        if not success:
            break
        draw_replay_calibration_overlay(frame, report)
        writer.write(ensure_frame_writer_size(frame, width, height))
    capture.release()
    writer.release()
    temp_path.replace(video_path)
    return video_path


def add_impact_marker_to_video(video_path, impact_info):
    from Backends.src.analysis.impact_detection import draw_impact_marker

    if not impact_info or impact_info.get("impact_frame") is None:
        return Path(video_path)

    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return video_path

    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temp_path = video_path.with_name(f"{video_path.stem}_impact{video_path.suffix}")
    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        return video_path

    frame_index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        draw_impact_marker(frame, impact_info, frame_index)
        writer.write(ensure_frame_writer_size(frame, width, height))
        frame_index += 1
    capture.release()
    writer.release()
    temp_path.replace(video_path)
    return video_path


def _write_review_metadata(row):
    REVIEW_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "source",
        "frame_index",
        "reason",
        "file_name",
        "model_name",
        "confidence",
        "bbox",
        "note",
    ]
    write_header = not REVIEW_FRAMES_CSV.exists()
    with open(REVIEW_FRAMES_CSV, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_review_frame(
    frame,
    timestamp,
    frame_index,
    reason,
    detections=None,
    source="video_analysis",
    note="",
):
    REVIEW_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    safe_reason = reason.replace(" ", "_").lower()
    file_name = f"{safe_reason}_{timestamp}_{frame_index:04d}.jpg"
    output_path = REVIEW_FRAMES_DIR / file_name
    review_frame = frame.copy()

    for detection in detections or []:
        x1, y1, x2, y2 = detection["box"]
        center_x, center_y = detection["center"]
        confidence = detection.get("confidence", 0)
        cv2.rectangle(review_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.circle(review_frame, (center_x, center_y), 4, (0, 165, 255), -1)
        draw_label(review_frame, f"{reason} {confidence:.2f}", x1, y1, (0, 120, 220))

    cv2.imwrite(str(output_path), review_frame)
    for detection in detections or [None]:
        _write_review_metadata(
            {
                "timestamp": timestamp,
                "source": source,
                "frame_index": frame_index,
                "reason": reason,
                "file_name": file_name,
                "model_name": "" if detection is None else detection.get("model_name", ""),
                "confidence": "" if detection is None else f"{detection.get('confidence', 0):.4f}",
                "bbox": "" if detection is None else ",".join(map(str, detection.get("box", ""))),
                "note": note,
            }
        )
    return output_path
