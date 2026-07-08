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
