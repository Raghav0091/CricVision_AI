"""Tiny generated-video fixtures for pipeline integration tests."""

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


def create_synthetic_cricket_video(
    tmp_path,
    frame_count=24,
    width=320,
    height=240,
):
    """Create a tiny cricket-like video and return its temporary path."""
    candidates = [
        (".avi", "MJPG"),
        (".mp4", "mp4v"),
    ]
    writer = None
    video_path = None
    for suffix, codec in candidates:
        candidate = Path(tmp_path) / f"synthetic_cricket{suffix}"
        candidate_writer = cv2.VideoWriter(
            str(candidate),
            cv2.VideoWriter_fourcc(*codec),
            12,
            (width, height),
        )
        if candidate_writer.isOpened():
            writer = candidate_writer
            video_path = candidate
            break
        candidate_writer.release()

    if writer is None or video_path is None:
        pytest.skip("No OpenCV video writer codec is available.")

    impact_frame = frame_count // 2
    for frame_index in range(frame_count):
        frame = np.full((height, width, 3), (25, 70, 25), dtype=np.uint8)

        for stump_x in (270, 278, 286):
            cv2.line(frame, (stump_x, 90), (stump_x, 205), (235, 235, 235), 3)

        if abs(frame_index - impact_frame) <= 3:
            cv2.rectangle(frame, (145, 105), (158, 190), (160, 110, 55), -1)

        if frame_index <= impact_frame:
            ball_x = 35 + frame_index * 10
            ball_y = 135
        else:
            ball_x = 35 + impact_frame * 10 + (frame_index - impact_frame) * 8
            ball_y = 135 - (frame_index - impact_frame) * 5
        cv2.circle(frame, (ball_x, ball_y), 6, (20, 20, 230), -1)
        writer.write(frame)

    writer.release()
    if not video_path.is_file() or video_path.stat().st_size == 0:
        pytest.skip("OpenCV did not produce a readable synthetic video.")
    return video_path
