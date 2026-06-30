"""Processed-video and review-frame annotation helpers."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

from Backends.src.utils.cv2_loader import cv2

REVIEW_FRAMES_DIR = Path("outputs/review_frames")
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
            writer.write(frame)
    finally:
        writer.release()

    if not output_path.is_file() or output_path.stat().st_size == 0:
        return None
    return output_path


def convert_to_browser_mp4(input_path, output_path):
    import imageio_ffmpeg

    subprocess.run(
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
        check=True,
    )
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
        writer.write(frame)
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
