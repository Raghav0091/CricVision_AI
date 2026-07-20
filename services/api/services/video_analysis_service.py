"""Persistent upload preparation for the CricVision Pro video workflow."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any, BinaryIO, NoReturn
from uuid import uuid4

import cv2
import numpy as np

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
# ponytail: prefer the earliest readable frame; only scan a small early window
# when frame 0 is black/blurry/corrupt. Never jump to the middle of the clip.
EARLY_REFERENCE_WINDOW = 24
MIN_MEAN_BRIGHTNESS = 14.0
MAX_MEAN_BRIGHTNESS = 246.0
MIN_LAPLACIAN_VARIANCE = 18.0


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

    selection = select_early_reference_frame(
        video_path,
        reference_path,
        frame_count=frame_count,
    )
    return {
        "duration_seconds": round(frame_count / fps, 3),
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "codec": _decode_fourcc(fourcc),
        "reference_frame_index": selection["reference_frame_index"],
        "reference_frame_selection": selection["reference_frame_selection"],
    }


def select_early_reference_frame(
    video_path: Path,
    destination: Path,
    *,
    frame_count: int,
) -> dict[str, Any]:
    """Pick the earliest clean stable frame from a small early window."""
    window = max(1, min(EARLY_REFERENCE_WINDOW, frame_count))
    best_usable: tuple[int, float, Any] | None = None
    first_decoded: tuple[int, Any] | None = None
    scanned = 0

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            _fail("Could not open the video while selecting a reference frame.")
        for index in range(window):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            scanned += 1
            if first_decoded is None:
                first_decoded = (index, frame)
            usable, score = _reference_frame_quality(frame)
            if usable and (best_usable is None or index < best_usable[0]):
                best_usable = (index, score, frame)
                # Earliest usable frame wins; stop as soon as frame 0 (or first
                # later usable candidate) is accepted.
                break
    finally:
        capture.release()

    if best_usable is not None:
        frame_index, score, frame = best_usable
        reason = (
            "frame_0_usable"
            if frame_index == 0
            else "earliest_early_window_fallback"
        )
    elif first_decoded is not None:
        frame_index, frame = first_decoded
        score = _laplacian_variance(frame)
        reason = "earliest_decoded_fallback"
    else:
        _fail("Could not extract an early calibration reference frame.")

    saved = cv2.imwrite(
        str(destination),
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    if not saved:
        _fail("Could not save the calibration reference frame.")

    selection = {
        "strategy": "earliest_clean_stable",
        "window_scanned": scanned,
        "window_limit": window,
        "selected_index": frame_index,
        "score": round(float(score), 3),
        "reason": reason,
    }
    return {
        "reference_frame_index": frame_index,
        "reference_frame_selection": selection,
    }


def replace_reference_frame(
    analysis_id: str,
    *,
    frame_index: int,
    frame: Any,
    selection: dict[str, Any] | None = None,
) -> int:
    """Overwrite the stored reference JPEG and persist the new index."""
    load_video_analysis(analysis_id)
    analysis_dir = VIDEO_ANALYSIS_ROOT / analysis_id
    reference_path = analysis_dir / "calibration" / "reference_frame.jpg"
    saved = cv2.imwrite(
        str(reference_path),
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    if not saved:
        raise VideoAnalysisServiceError(
            "Could not save the refreshed calibration reference frame.",
            status_code=500,
        )

    metadata_path = analysis_dir / "reports" / "analysis_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VideoAnalysisServiceError(
            "Analysis metadata could not be updated.",
            status_code=500,
        ) from exc
    metadata["reference_frame_index"] = int(frame_index)
    if selection is not None:
        metadata["reference_frame_selection"] = selection
    metadata["updated_at"] = (
        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    try:
        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise VideoAnalysisServiceError(
            "Analysis metadata could not be saved.",
            status_code=500,
        ) from exc
    return int(frame_index)


def _reference_frame_quality(frame: Any) -> tuple[bool, float]:
    if frame is None or getattr(frame, "size", 0) == 0:
        return False, 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    if (
        mean_brightness < MIN_MEAN_BRIGHTNESS
        or mean_brightness > MAX_MEAN_BRIGHTNESS
    ):
        return False, mean_brightness
    sharpness = _laplacian_variance(gray)
    if sharpness < MIN_LAPLACIAN_VARIANCE:
        return False, sharpness
    # Prefer brighter-but-sharp early frames only as a soft score.
    return True, sharpness + (mean_brightness / 255.0)


def _laplacian_variance(frame_or_gray: Any) -> float:
    if len(frame_or_gray.shape) == 3:
        gray = cv2.cvtColor(frame_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame_or_gray
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


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
