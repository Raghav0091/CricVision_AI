"""Persistent upload preparation for the CricVision Pro video workflow."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
from typing import BinaryIO, NoReturn
from uuid import uuid4

import cv2

from ..schemas.video_analysis import VideoAnalysisPreparedResponse


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VIDEO_ANALYSIS_ROOT = PROJECT_ROOT / "outputs" / "video_analysis"
MAX_VIDEO_BYTES = 500 * 1024 * 1024
ACCEPTED_VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
ANALYSIS_DIRECTORIES = (
    "raw",
    "normalized",
    "calibration",
    "detections",
    "tracking",
    "trajectory",
    "replay",
    "reports",
)
ANALYSIS_ID_PATTERN = re.compile(r"^analysis_\d{8}_\d{6}_[0-9a-f]{6}$")


class VideoAnalysisServiceError(Exception):
    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def prepare_video(
    upload: BinaryIO,
    original_filename: str | None,
) -> VideoAnalysisPreparedResponse:
    safe_original_name, suffix = _validate_filename(original_filename)
    analysis_id = _new_analysis_id()
    analysis_dir = VIDEO_ANALYSIS_ROOT / analysis_id

    try:
        for directory in ANALYSIS_DIRECTORIES:
            (analysis_dir / directory).mkdir(parents=True, exist_ok=False)

        stored_filename = f"original_video{suffix}"
        raw_path = analysis_dir / "raw" / stored_filename
        file_size_bytes = _save_upload(upload, raw_path)
        reference_path = analysis_dir / "calibration" / "reference_frame.jpg"
        video_metadata = _read_video_metadata(raw_path, reference_path)
        created_at = datetime.now(timezone.utc)

        record = VideoAnalysisPreparedResponse(
            success=True,
            analysis_id=analysis_id,
            status="prepared",
            original_filename=safe_original_name,
            stored_filename=stored_filename,
            file_size_bytes=file_size_bytes,
            created_at=created_at,
            original_video_url=f"/static/video-analysis/{analysis_id}/raw/{stored_filename}",
            reference_frame_url=f"/static/video-analysis/{analysis_id}/calibration/reference_frame.jpg",
            message="Video uploaded and prepared for scene calibration.",
            **video_metadata,
        )
        _write_metadata(analysis_dir, record)
        return record
    except VideoAnalysisServiceError:
        shutil.rmtree(analysis_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(analysis_dir, ignore_errors=True)
        raise VideoAnalysisServiceError(
            f"Video preparation failed: {type(exc).__name__}.",
            status_code=500,
        ) from exc


def load_video_analysis(analysis_id: str) -> VideoAnalysisPreparedResponse:
    if not ANALYSIS_ID_PATTERN.fullmatch(analysis_id):
        raise VideoAnalysisServiceError("Invalid analysis ID.", status_code=404)

    analysis_dir = VIDEO_ANALYSIS_ROOT / analysis_id
    metadata_path = analysis_dir / "reports" / "analysis_metadata.json"
    if not metadata_path.is_file():
        raise VideoAnalysisServiceError("Video analysis not found.", status_code=404)

    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        record = VideoAnalysisPreparedResponse.model_validate(data)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise VideoAnalysisServiceError(
            "Stored video analysis metadata is unavailable.",
            status_code=500,
        ) from exc

    if Path(record.stored_filename).name != record.stored_filename:
        raise VideoAnalysisServiceError(
            "Stored video analysis metadata is invalid.",
            status_code=500,
        )
    raw_path = analysis_dir / "raw" / record.stored_filename
    reference_path = analysis_dir / "calibration" / "reference_frame.jpg"
    if not raw_path.is_file() or not reference_path.is_file():
        raise VideoAnalysisServiceError(
            "Stored video analysis files are missing.",
            status_code=404,
        )
    return record


def _new_analysis_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"analysis_{timestamp}_{uuid4().hex[:6]}"


def _validate_filename(original_filename: str | None) -> tuple[str, str]:
    basename = Path(original_filename or "").name
    suffix = Path(basename).suffix.lower()
    if suffix not in ACCEPTED_VIDEO_SUFFIXES:
        supported = ", ".join(sorted(ACCEPTED_VIDEO_SUFFIXES))
        raise VideoAnalysisServiceError(
            f"Unsupported video format. Choose one of: {supported}."
        )
    safe_stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", Path(basename).stem).strip(" ._")
    return f"{safe_stem or 'video'}{suffix}", suffix


def _save_upload(upload: BinaryIO, destination: Path) -> int:
    total_bytes = 0
    try:
        with destination.open("xb") as output:
            while chunk := upload.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_VIDEO_BYTES:
                    _fail(
                        "Video exceeds the 500 MB upload limit.",
                        status_code=413,
                    )
                output.write(chunk)
    except VideoAnalysisServiceError:
        destination.unlink(missing_ok=True)
        raise
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise VideoAnalysisServiceError(
            "The uploaded video could not be saved.",
            status_code=500,
        ) from exc

    if total_bytes == 0:
        destination.unlink(missing_ok=True)
        _fail("The uploaded video is empty.")
    return total_bytes


def _read_video_metadata(video_path: Path, reference_path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        _fail("OpenCV could not open the uploaded video.")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    finally:
        capture.release()

    if not math.isfinite(fps) or fps <= 0:
        _fail("The uploaded video has an invalid FPS value.")
    if frame_count <= 0:
        _fail("The uploaded video contains no readable frames.")
    if width <= 0 or height <= 0:
        _fail("The uploaded video has invalid dimensions.")

    reference_frame_index = frame_count // 2
    _extract_reference_frame(video_path, reference_path, reference_frame_index)
    return {
        "duration_seconds": round(frame_count / fps, 3),
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "codec": _decode_fourcc(fourcc),
        "reference_frame_index": reference_frame_index,
    }


def _extract_reference_frame(
    video_path: Path,
    destination: Path,
    frame_index: int,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    frame = None
    try:
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, candidate = capture.read()
            if ok:
                frame = candidate
    finally:
        capture.release()

    if frame is None:
        capture = cv2.VideoCapture(str(video_path))
        try:
            if capture.isOpened():
                for index in range(frame_index + 1):
                    ok, candidate = capture.read()
                    if not ok:
                        break
                    if index == frame_index:
                        frame = candidate
        finally:
            capture.release()

    if frame is None:
        _fail("Could not extract the middle calibration reference frame.")
    saved = cv2.imwrite(
        str(destination),
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    if not saved:
        _fail("Could not save the calibration reference frame.")


def _decode_fourcc(value: int) -> str | None:
    if value <= 0:
        return None
    codec = "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4))
    codec = "".join(character for character in codec if character.isprintable()).strip()
    return codec or None


def _write_metadata(
    analysis_dir: Path,
    record: VideoAnalysisPreparedResponse,
) -> None:
    metadata_path = analysis_dir / "reports" / "analysis_metadata.json"
    try:
        metadata_path.write_text(
            json.dumps(record.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise VideoAnalysisServiceError(
            "Analysis metadata could not be saved.",
            status_code=500,
        ) from exc


def _fail(message: str, *, status_code: int = 400) -> NoReturn:
    raise VideoAnalysisServiceError(message, status_code=status_code)
