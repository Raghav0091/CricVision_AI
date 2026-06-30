"""Integration tests for pure processed-video annotation writing."""

import numpy as np
import pytest

pytest.importorskip("cv2")

from Backends.src.video_pipeline.annotation_writer import write_annotated_video
from Backends.src.video_pipeline.video_reader import iter_video_frames, open_video


def _dummy_frames(frame_count=8, width=160, height=120):
    return [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(frame_count)]


def _dummy_detections(frame_count=8):
    timeline = []
    for frame_index in range(frame_count):
        center_x = 25 + frame_index * 10
        timeline.append(
            {
                "frame_index": frame_index,
                "ball_detections": [
                    {
                        "box": [center_x - 4, 56, center_x + 4, 64],
                        "center": [center_x, 60],
                        "confidence": 0.9,
                    }
                ],
                "bat_detections": (
                    [{"box": [82, 35, 92, 95], "confidence": 0.8}]
                    if frame_index in {3, 4}
                    else []
                ),
                "stump_detections": [{"box": [135, 40, 145, 105], "confidence": 0.95}],
            }
        )
    return timeline


def test_annotation_writer_creates_video_with_dummy_detections(tmp_path):
    output_path = tmp_path / "annotated.mp4"
    result = write_annotated_video(
        _dummy_frames(),
        output_path,
        fps=12,
        frame_detections=_dummy_detections(),
    )
    if result is None:
        pytest.skip("OpenCV mp4v writer is unavailable.")

    assert result == output_path
    capture = open_video(output_path)
    if capture is None:
        pytest.skip("OpenCV cannot reopen its generated mp4v output.")
    assert len(list(iter_video_frames(capture))) == 8
    capture.release()


def test_annotation_writer_disabled_does_not_create_output(tmp_path):
    output_path = tmp_path / "disabled.mp4"
    result = write_annotated_video(
        _dummy_frames(),
        output_path,
        enabled=False,
    )
    assert result is None
    assert not output_path.exists()


def test_annotation_writer_handles_missing_detections(tmp_path):
    output_path = tmp_path / "no_detections.mp4"
    result = write_annotated_video(
        _dummy_frames(frame_count=3),
        output_path,
        frame_detections=None,
    )
    if result is None:
        pytest.skip("OpenCV mp4v writer is unavailable.")
    assert output_path.is_file()
