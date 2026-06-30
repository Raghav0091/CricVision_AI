"""Integration tests for video_reader using a tiny generated clip."""

from Backends.src.video_pipeline.video_reader import (
    iter_video_frames,
    open_video,
    read_video_metadata,
)
from tests.helpers.synthetic_video import create_synthetic_cricket_video


def test_synthetic_video_metadata_and_iteration(tmp_path):
    video_path = create_synthetic_cricket_video(tmp_path)
    capture = open_video(video_path)
    assert capture is not None

    metadata = read_video_metadata(capture)
    assert metadata["fps"] > 0
    assert metadata["frame_count"] >= 20
    assert metadata["width"] == 320
    assert metadata["height"] == 240

    frames = list(iter_video_frames(capture))
    capture.release()
    assert len(frames) == metadata["frame_count"]
    assert frames[0][0] == 0
    assert frames[-1][0] == len(frames) - 1
    assert frames[0][1].shape[:2] == (240, 320)


def test_video_reader_handles_invalid_path(tmp_path):
    assert open_video(tmp_path / "missing-video.mp4") is None
