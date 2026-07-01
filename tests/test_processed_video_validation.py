"""Processed video validation tests avoid real models and videos."""

from pathlib import Path

import pytest

from Backends.src.video_pipeline.annotation_writer import validate_processed_video_path


def test_validate_processed_video_path_handles_missing_file(tmp_path):
    missing = tmp_path / "missing.mp4"
    result = validate_processed_video_path(missing)

    assert result["valid"] is False
    assert result["can_preview"] is False
    assert "missing" in result["error"].lower()


def test_validate_processed_video_path_handles_empty_file(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")

    result = validate_processed_video_path(empty)

    assert result["valid"] is False
    assert result["exists"] is True
    assert result["file_size"] == 0


def test_validate_processed_video_path_accepts_readable_video(tmp_path, monkeypatch):
    output = tmp_path / "preview.mp4"
    output.write_bytes(b"fake-video")

    class FakeFrame:
        shape = (48, 64, 3)

    class FakeCapture:
        def isOpened(self):
            return True

        def get(self, prop):
            if prop == 3:
                return 64
            if prop == 4:
                return 48
            return 0

        def read(self):
            return True, FakeFrame()

        def release(self):
            return None

    class FakeCV2:
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4

        @staticmethod
        def VideoCapture(_path):
            return FakeCapture()

    monkeypatch.setattr(
        "Backends.src.video_pipeline.annotation_writer.cv2",
        FakeCV2(),
    )

    result = validate_processed_video_path(output)

    assert result["valid"] is True
    assert result["can_preview"] is True
    assert result["width"] == 64
    assert result["height"] == 48
