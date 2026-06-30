"""Safe OpenCV video opening and lightweight metadata helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

from Backends.src.utils.cv2_loader import cv2


def open_video(video_path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return None
    return capture


def read_video_metadata(capture) -> dict:
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    return {
        "fps": float(fps),
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }


def iter_video_frames(capture, max_frames=None):
    frame_index = 0
    while max_frames is None or frame_index < max_frames:
        success, frame = capture.read()
        if not success:
            break
        yield frame_index, frame
        frame_index += 1


def extract_first_video_frame(uploaded_video):
    uploaded_video.seek(0)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
            temp_input.write(uploaded_video.read())
            temp_path = Path(temp_input.name)

        capture = open_video(temp_path)
        if capture is None:
            return None
        success, frame = capture.read()
        capture.release()
        return frame if success else None
    finally:
        uploaded_video.seek(0)
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def write_video_frames(frames, output_path, fps=25):
    if not frames:
        return False

    output_path = Path(output_path)
    height, width = frames[0].shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        return False
    for frame in frames:
        writer.write(frame)
    writer.release()
    return True
