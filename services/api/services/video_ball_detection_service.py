"""Every-frame raw ball detection for persistent Video Analysis records."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any

import cv2

from ..schemas.video_analysis import (
    BallDetectorResultMetadata,
    BallCandidate,
    FrameDetectionRecord,
    NormalizedBox,
    NormalizedPoint,
    PixelPoint,
    VideoBallDetectionResultLinks,
    VideoBallDetectionResultResponse,
    VideoBallDetectionsDocument,
    VideoBallDetectionSettings,
    VideoBallDetectionSummary,
)
from .ball_detection_clip import (
    BALL_INFERENCE_LOCK,
    PROJECT_ROOT,
    extract_ball_candidates,
    load_ball_model,
    transcode_browser_mp4,
)
from .ball_detector_registry import (
    BallDetectorModelMissing,
    resolve_ball_detector_model,
)
from .video_analysis_service import (
    VIDEO_ANALYSIS_ROOT,
    VideoAnalysisServiceError,
    load_video_analysis,
)
from .video_ball_detection_job_store import (
    utc_now,
    video_ball_detection_job_store,
)
from .video_calibration_service import load_video_calibration


FRAME_STRIDE = 1
IMAGE_SIZE = 960
CONFIDENCE_THRESHOLD = 0.15
MAX_DETECTIONS = 20
DETECTIONS_JSON_FILENAME = "detections.json"
DETECTIONS_CSV_FILENAME = "detections.csv"
DETECTION_SUMMARY_FILENAME = "detection_summary.json"
DETECTION_OVERLAY_FILENAME = "detection_overlay.mp4"
class VideoBallDetectionError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status: str = "failed",
        status_code: int = 500,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.status_code = status_code


def mark_video_ball_detection_queued(
    analysis_id: str,
    job_id: str,
    ball_detector_model_key: str,
    ball_detector_model_name: str,
) -> None:
    now = utc_now()
    _update_analysis_metadata(
        analysis_id,
        ball_detection_status="detection_queued",
        ball_detection_job_id=job_id,
        ball_detection_started_at=_iso(now),
        ball_detection_completed_at=None,
        ball_detector_model_key=ball_detector_model_key,
        ball_detector_model_name=ball_detector_model_name,
        detection_summary_url=None,
        detection_overlay_url=None,
        updated_at=_iso(now),
    )


def run_video_ball_detection_job(
    analysis_id: str,
    job_id: str,
) -> None:
    try:
        summary, _ = _process_video_ball_detection(
            analysis_id,
            job_id,
        )
        links = VideoBallDetectionResultLinks(
            processed_video_url=summary.processed_video_url,
            detections_json_url=summary.detections_json_url,
            detections_csv_url=summary.detections_csv_url,
            detection_summary_url=summary.detection_summary_url,
        )
        video_ball_detection_job_store.update(
            job_id,
            success=True,
            status="ready",
            progress=100,
            current_frame=summary.frames_processed,
            total_frames=summary.total_frames,
            model_path_used=summary.model_path_used,
            error_message=None,
            result=links.model_dump(mode="json"),
            message="Every-frame ball detection completed.",
        )
        _update_analysis_metadata(
            analysis_id,
            ball_detection_status="detection_complete",
            ball_detection_completed_at=_iso(summary.completed_at),
            detection_summary_url=summary.detection_summary_url,
            detection_overlay_url=summary.processed_video_url,
            updated_at=_iso(summary.completed_at),
        )
    except VideoBallDetectionError as exc:
        _mark_job_failed(analysis_id, job_id, exc.status, exc.message)
    except Exception as exc:
        _mark_job_failed(
            analysis_id,
            job_id,
            "failed",
            f"Every-frame ball detection failed: {type(exc).__name__}.",
        )


def load_video_ball_detection_result(
    analysis_id: str,
) -> VideoBallDetectionResultResponse:
    load_video_analysis(analysis_id)
    output_dir = _detection_output_dir(analysis_id)
    summary_path = output_dir / DETECTION_SUMMARY_FILENAME
    detections_path = output_dir / DETECTIONS_JSON_FILENAME
    csv_path = output_dir / DETECTIONS_CSV_FILENAME
    overlay_path = output_dir / DETECTION_OVERLAY_FILENAME
    if not summary_path.is_file():
        raise VideoBallDetectionError(
            "Every-frame ball detection has not completed.",
            status_code=404,
        )
    try:
        summary = VideoBallDetectionSummary.model_validate(
            json.loads(summary_path.read_text(encoding="utf-8"))
        )
        document = VideoBallDetectionsDocument.model_validate(
            json.loads(detections_path.read_text(encoding="utf-8"))
        )
    except FileNotFoundError as exc:
        raise VideoBallDetectionError(
            "Saved ball-detection output files are missing.",
            status_code=404,
        ) from exc
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise VideoBallDetectionError(
            "Saved ball-detection results are unavailable.",
            status_code=500,
        ) from exc

    if (
        summary.analysis_id != analysis_id
        or document.analysis_id != analysis_id
        or not csv_path.is_file()
        or not overlay_path.is_file()
    ):
        raise VideoBallDetectionError(
            "Saved ball-detection results are incomplete.",
            status_code=404,
        )
    frame_candidate_counts = [
        len(frame.detections) for frame in document.frames
    ]
    if len(frame_candidate_counts) != summary.total_frames:
        raise VideoBallDetectionError(
            "Saved ball-detection frame coverage is incomplete.",
            status_code=500,
        )
    return VideoBallDetectionResultResponse(
        success=True,
        status="ready",
        analysis_id=analysis_id,
        summary=summary,
        frame_candidate_counts=frame_candidate_counts,
        message="Every-frame ball detection completed.",
    )


def _process_video_ball_detection(
    analysis_id: str,
    job_id: str,
) -> tuple[VideoBallDetectionSummary, list[int]]:
    job = video_ball_detection_job_store.get(job_id)
    if job is None:
        raise VideoBallDetectionError("Ball-detection job not found.")
    analysis = load_video_analysis(analysis_id)
    analysis_dir = VIDEO_ANALYSIS_ROOT / analysis_id
    raw_path = analysis_dir / "raw" / analysis.stored_filename
    if not raw_path.is_file():
        raise VideoBallDetectionError("Original analysis video is missing.")

    output_dir = _detection_output_dir(analysis_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_path = output_dir / "detection_overlay_intermediate.avi"
    encoded_path = output_dir / "detection_overlay_encoded.mp4"
    overlay_path = output_dir / DETECTION_OVERLAY_FILENAME
    for stale_path in (
        output_dir / DETECTIONS_JSON_FILENAME,
        output_dir / DETECTIONS_CSV_FILENAME,
        output_dir / DETECTION_SUMMARY_FILENAME,
        overlay_path,
        intermediate_path,
        encoded_path,
    ):
        stale_path.unlink(missing_ok=True)
    started_at = utc_now()
    started_clock = time.perf_counter()
    total_frames = analysis.frame_count
    if total_frames <= 0:
        raise VideoBallDetectionError(
            "Original analysis video contains zero frames."
        )

    video_ball_detection_job_store.update(
        job_id,
        status="loading_model",
        message="Loading ball model...",
    )
    try:
        selected_model = resolve_ball_detector_model(
            job.get("ball_detector_model_key")
        )
    except BallDetectorModelMissing as exc:
        raise VideoBallDetectionError(
            str(exc),
            status="ball_detector_missing",
        ) from exc
    model_path = selected_model.path
    model_path_used = model_path.relative_to(PROJECT_ROOT).as_posix()
    model_warning = selected_model.fallback_reason
    detector_metadata = BallDetectorResultMetadata(
        requested_key=selected_model.requested_key,
        selected_key=selected_model.selected_key,
        display_name=selected_model.display_name,
        model_file=model_path.name,
    )
    try:
        model = load_ball_model(model_path)
    except Exception as exc:
        raise VideoBallDetectionError(
            f"Ball detector could not be loaded: {type(exc).__name__}."
        ) from exc
    model_class_names = _model_class_names(getattr(model, "names", {}))
    device_argument, device_used = _select_device()
    video_ball_detection_job_store.update(
        job_id,
        status="processing",
        model_path_used=model_path_used,
        ball_detector_model_key=selected_model.model_key,
        ball_detector_model_name=selected_model.display_name,
        message=f"Processing frame 0 of {total_frames}.",
    )
    _update_analysis_metadata(
        analysis_id,
        ball_detection_status="detecting_ball",
        updated_at=_iso(utc_now()),
    )

    corridor = _load_calibration_corridor(analysis_id)
    frames: list[FrameDetectionRecord] = []
    csv_rows: list[dict[str, Any]] = []
    confidences: list[float] = []
    writer = None
    capture = cv2.VideoCapture(str(raw_path))
    if not capture.isOpened():
        capture.release()
        raise VideoBallDetectionError("OpenCV could not open the original video.")

    try:
        input_fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(input_fps) or input_fps <= 0:
            raise VideoBallDetectionError(
                "Original analysis video has an invalid FPS value."
            )
        if width <= 0 or height <= 0:
            raise VideoBallDetectionError(
                "Original analysis video has invalid dimensions."
            )
        if capture_frame_count <= 0:
            raise VideoBallDetectionError(
                "Original analysis video contains zero frames."
            )
        if capture_frame_count != total_frames:
            raise VideoBallDetectionError(
                "Stored frame count no longer matches the original video."
            )
        writer = cv2.VideoWriter(
            str(intermediate_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            input_fps,
            (width, height),
        )
        if not writer.isOpened():
            raise VideoBallDetectionError(
                "Could not create the full detection-overlay video."
            )

        with BALL_INFERENCE_LOCK:
            for frame_index in range(total_frames):
                ok, frame = capture.read()
                if not ok:
                    raise VideoBallDetectionError(
                        f"Video decoding stopped at frame {frame_index} "
                        f"of {total_frames}."
                    )
                try:
                    results = model.predict(
                        source=frame,
                        imgsz=IMAGE_SIZE,
                        conf=CONFIDENCE_THRESHOLD,
                        max_det=MAX_DETECTIONS,
                        device=device_argument,
                        verbose=False,
                    )
                    raw_candidates = extract_ball_candidates(
                        results,
                        getattr(model, "names", {}),
                        strict=True,
                    )
                except Exception as exc:
                    raise VideoBallDetectionError(
                        f"Ball detection failed on frame {frame_index}: "
                        f"{type(exc).__name__}."
                    ) from exc

                candidates: list[BallCandidate] = []
                for candidate_index, raw_candidate in enumerate(
                    raw_candidates,
                    start=1,
                ):
                    candidate = _build_candidate(
                        analysis_id=analysis_id,
                        frame_index=frame_index,
                        candidate_index=candidate_index,
                        raw_candidate=raw_candidate,
                        frame_width=width,
                        frame_height=height,
                        corridor=corridor,
                    )
                    if candidate is None:
                        continue
                    candidates.append(candidate)
                    confidences.append(candidate.confidence)
                    csv_rows.append(
                        _candidate_csv_row(
                            analysis_id,
                            frame_index,
                            frame_index / input_fps,
                            len(candidates),
                            candidate,
                        )
                    )
                    _draw_candidate(frame, candidate)

                _draw_frame_debug_panel(frame, frame_index, total_frames)
                writer.write(frame)
                frames.append(
                    FrameDetectionRecord(
                        frame_index=frame_index,
                        timestamp_seconds=round(frame_index / input_fps, 6),
                        processed=True,
                        detections=candidates,
                    )
                )
                processed_frames = frame_index + 1
                progress = int(processed_frames / total_frames * 100)
                video_ball_detection_job_store.update(
                    job_id,
                    status="processing",
                    progress=progress,
                    current_frame=processed_frames,
                    total_frames=total_frames,
                    message=(
                        f"Processing frame {processed_frames} "
                        f"of {total_frames}."
                    ),
                )
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if len(frames) != total_frames:
        raise VideoBallDetectionError(
            "Not every original video frame was processed."
        )
    video_ball_detection_job_store.update(
        job_id,
        status="writing_video",
        progress=100,
        current_frame=total_frames,
        message="Generating detection video...",
    )
    try:
        transcode_browser_mp4(
            intermediate_path,
            encoded_path,
            timeout_seconds=600,
        )
        encoded_path.replace(overlay_path)
    except Exception as exc:
        encoded_path.unlink(missing_ok=True)
        raise VideoBallDetectionError(
            f"Could not encode a browser-compatible detection video: "
            f"{type(exc).__name__}."
        ) from exc
    finally:
        intermediate_path.unlink(missing_ok=True)

    output_frame_count, output_fps = _verify_output_video(overlay_path)
    if output_frame_count != total_frames:
        overlay_path.unlink(missing_ok=True)
        raise VideoBallDetectionError(
            "Detection-overlay frame count does not match the original video."
        )

    video_ball_detection_job_store.update(
        job_id,
        status="saving_results",
        progress=100,
        current_frame=total_frames,
        message="Saving detection results...",
    )
    settings = VideoBallDetectionSettings(
        frame_stride=FRAME_STRIDE,
        imgsz=IMAGE_SIZE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        max_det=MAX_DETECTIONS,
    )
    document = VideoBallDetectionsDocument(
        analysis_id=analysis_id,
        detector=detector_metadata,
        model_path_used=model_path_used,
        model_class_names=model_class_names,
        settings=settings,
        frames=frames,
    )
    detections_json_path = output_dir / DETECTIONS_JSON_FILENAME
    detections_csv_path = output_dir / DETECTIONS_CSV_FILENAME
    _write_json(detections_json_path, document.model_dump(mode="json"))
    _write_csv(detections_csv_path, csv_rows)

    completed_at = utc_now()
    relative_base = f"/static/video-analysis/{analysis_id}/detections"
    frames_with_candidates = sum(bool(frame.detections) for frame in frames)
    total_candidates = len(confidences)
    inside_count = sum(
        candidate.inside_pitch_corridor is True
        for frame in frames
        for candidate in frame.detections
    )
    outside_count = sum(
        candidate.inside_pitch_corridor is False
        for frame in frames
        for candidate in frame.detections
    )
    unknown_count = total_candidates - inside_count - outside_count
    summary = VideoBallDetectionSummary(
        analysis_id=analysis_id,
        status="ready",
        created_at=started_at,
        completed_at=completed_at,
        original_video_url=analysis.original_video_url,
        processed_video_url=f"{relative_base}/{DETECTION_OVERLAY_FILENAME}",
        detections_json_url=f"{relative_base}/{DETECTIONS_JSON_FILENAME}",
        detections_csv_url=f"{relative_base}/{DETECTIONS_CSV_FILENAME}",
        detection_summary_url=f"{relative_base}/{DETECTION_SUMMARY_FILENAME}",
        detector=detector_metadata,
        model_path_used=model_path_used,
        model_warning=model_warning,
        model_class_names=model_class_names,
        device_used=device_used,
        imgsz=IMAGE_SIZE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        frame_stride=FRAME_STRIDE,
        max_det=MAX_DETECTIONS,
        total_frames=total_frames,
        frames_processed=len(frames),
        frames_with_candidates=frames_with_candidates,
        frames_without_candidates=total_frames - frames_with_candidates,
        total_candidates=total_candidates,
        frames_with_multiple_candidates=sum(
            len(frame.detections) > 1 for frame in frames
        ),
        candidates_inside_pitch_corridor=inside_count,
        candidates_outside_pitch_corridor=outside_count,
        candidates_without_corridor_information=unknown_count,
        best_confidence=round(max(confidences, default=0.0), 6),
        average_confidence=round(
            sum(confidences) / total_candidates
            if total_candidates
            else 0.0,
            6,
        ),
        average_candidates_per_detected_frame=round(
            total_candidates / frames_with_candidates
            if frames_with_candidates
            else 0.0,
            6,
        ),
        processing_duration_seconds=round(
            time.perf_counter() - started_clock,
            3,
        ),
        output_video_frame_count=output_frame_count,
        input_fps=round(input_fps, 6),
        output_fps=round(output_fps, 6),
        input_duration_seconds=round(total_frames / input_fps, 6),
        output_duration_seconds=round(output_frame_count / output_fps, 6),
        message="Every-frame ball detection completed.",
    )
    _write_json(
        output_dir / DETECTION_SUMMARY_FILENAME,
        summary.model_dump(mode="json"),
    )
    return summary, [len(frame.detections) for frame in frames]


def _build_candidate(
    *,
    analysis_id: str,
    frame_index: int,
    candidate_index: int,
    raw_candidate: dict[str, Any],
    frame_width: int,
    frame_height: int,
    corridor: list[NormalizedPoint] | None,
) -> BallCandidate | None:
    try:
        values = [float(value) for value in raw_candidate["bbox_xyxy"]]
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            return None
        x1, y1, x2, y2 = values
        x1 = max(0.0, min(float(frame_width), x1))
        y1 = max(0.0, min(float(frame_height), y1))
        x2 = max(x1, min(float(frame_width), x2))
        y2 = max(y1, min(float(frame_height), y2))
        width_pixels = x2 - x1
        height_pixels = y2 - y1
        if width_pixels <= 0 or height_pixels <= 0:
            return None
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center_normalized = NormalizedPoint(
            x=_clamp(center_x / frame_width),
            y=_clamp(center_y / frame_height),
        )
        inside_corridor = (
            _point_inside_polygon(center_normalized, corridor)
            if corridor is not None
            else None
        )
        return BallCandidate(
            candidate_id=(
                f"frame_{frame_index:06d}_candidate_{candidate_index:03d}"
            ),
            class_id=int(raw_candidate["class_id"]),
            class_name=str(raw_candidate["class_name"]),
            confidence=float(raw_candidate["confidence"]),
            bbox_xyxy=[
                round(x1, 3),
                round(y1, 3),
                round(x2, 3),
                round(y2, 3),
            ],
            bbox_normalized=NormalizedBox(
                x=_clamp(x1 / frame_width),
                y=_clamp(y1 / frame_height),
                width=max(
                    0.000001,
                    _clamp(x2 / frame_width) - _clamp(x1 / frame_width),
                ),
                height=max(
                    0.000001,
                    _clamp(y2 / frame_height) - _clamp(y1 / frame_height),
                ),
            ),
            center=PixelPoint(
                x=round(center_x, 3),
                y=round(center_y, 3),
            ),
            center_normalized=center_normalized,
            width_pixels=round(width_pixels, 3),
            height_pixels=round(height_pixels, 3),
            area_pixels=round(width_pixels * height_pixels, 3),
            inside_pitch_corridor=inside_corridor,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _candidate_csv_row(
    analysis_id: str,
    frame_index: int,
    timestamp_seconds: float,
    candidate_index: int,
    candidate: BallCandidate,
) -> dict[str, Any]:
    x1, y1, x2, y2 = candidate.bbox_xyxy
    return {
        "analysis_id": analysis_id,
        "frame_index": frame_index,
        "timestamp_seconds": round(timestamp_seconds, 6),
        "candidate_index": candidate_index,
        "class_id": candidate.class_id,
        "class_name": candidate.class_name,
        "confidence": candidate.confidence,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "center_x": candidate.center.x,
        "center_y": candidate.center.y,
        "width_pixels": candidate.width_pixels,
        "height_pixels": candidate.height_pixels,
        "area_pixels": candidate.area_pixels,
        "normalized_center_x": candidate.center_normalized.x,
        "normalized_center_y": candidate.center_normalized.y,
        "inside_pitch_corridor": (
            ""
            if candidate.inside_pitch_corridor is None
            else candidate.inside_pitch_corridor
        ),
    }


def _draw_candidate(frame, candidate: BallCandidate) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (round(value) for value in candidate.bbox_xyxy)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1, min(width - 1, x2))
    y2 = max(y1, min(height - 1, y2))
    yellow = (0, 230, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), yellow, 2, cv2.LINE_AA)
    cv2.circle(
        frame,
        (round(candidate.center.x), round(candidate.center.y)),
        3,
        yellow,
        -1,
        cv2.LINE_AA,
    )
    label = f"Ball {candidate.confidence:.2f}"
    cv2.putText(
        frame,
        label,
        (x1, max(18, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        yellow,
        1,
        cv2.LINE_AA,
    )


def _draw_frame_debug_panel(
    frame,
    frame_index: int,
    total_frames: int,
) -> None:
    cv2.rectangle(frame, (10, 10), (310, 64), (18, 18, 18), -1)
    cv2.putText(
        frame,
        f"Frame {frame_index + 1}/{total_frames}",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Raw detections - no tracking",
        (20, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 230, 255),
        1,
        cv2.LINE_AA,
    )


def _load_calibration_corridor(
    analysis_id: str,
) -> list[NormalizedPoint] | None:
    try:
        calibration = load_video_calibration(analysis_id)
    except (VideoAnalysisServiceError, ValueError):
        return None
    return calibration.pitch_geometry.corridor


def _point_inside_polygon(
    point: NormalizedPoint,
    polygon: list[NormalizedPoint],
) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if (current.y > point.y) != (previous.y > point.y):
            intersection_x = (
                (previous.x - current.x)
                * (point.y - current.y)
                / (previous.y - current.y)
                + current.x
            )
            if point.x < intersection_x:
                inside = not inside
        previous = current
    return inside


def _model_class_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=str)]
    if isinstance(names, (list, tuple)):
        return [str(value) for value in names]
    return [str(names)] if names else []


def _select_device() -> tuple[int | str, str]:
    try:
        import torch

        if torch.cuda.is_available():
            return 0, "cuda:0"
    except Exception:
        pass
    return "cpu", "cpu"


def _verify_output_video(path: Path) -> tuple[int, float]:
    if not path.is_file() or path.stat().st_size == 0:
        raise VideoBallDetectionError(
            "Detection-overlay video was not created."
        )
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoBallDetectionError(
            "Browser-compatible detection video could not be verified."
        )
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise VideoBallDetectionError(
            "Detection-overlay video metadata is invalid."
        )
    return frame_count, fps


def _detection_output_dir(analysis_id: str) -> Path:
    return VIDEO_ANALYSIS_ROOT / analysis_id / "detections"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(path)
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        raise VideoBallDetectionError(
            f"{path.name} could not be saved."
        ) from exc


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "analysis_id",
        "frame_index",
        "timestamp_seconds",
        "candidate_index",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
        "width_pixels",
        "height_pixels",
        "area_pixels",
        "normalized_center_x",
        "normalized_center_y",
        "inside_pitch_corridor",
    ]
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(path)
    except (OSError, csv.Error) as exc:
        temporary_path.unlink(missing_ok=True)
        raise VideoBallDetectionError(
            "detections.csv could not be saved."
        ) from exc


def _update_analysis_metadata(
    analysis_id: str,
    **updates: Any,
) -> None:
    metadata_path = (
        VIDEO_ANALYSIS_ROOT
        / analysis_id
        / "reports"
        / "analysis_metadata.json"
    )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VideoBallDetectionError(
            "Analysis metadata could not be updated."
        ) from exc
    metadata.update(updates)
    temporary_path = metadata_path.with_suffix(".json.tmp")
    try:
        temporary_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(metadata_path)
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        raise VideoBallDetectionError(
            "Analysis metadata could not be saved."
        ) from exc


def _mark_job_failed(
    analysis_id: str,
    job_id: str,
    status: str,
    message: str,
) -> None:
    output_dir = _detection_output_dir(analysis_id)
    for temporary_path in (
        output_dir / "detection_overlay_intermediate.avi",
        output_dir / "detection_overlay_encoded.mp4",
        output_dir / f"{DETECTIONS_JSON_FILENAME}.tmp",
        output_dir / f"{DETECTIONS_CSV_FILENAME}.tmp",
        output_dir / f"{DETECTION_SUMMARY_FILENAME}.tmp",
    ):
        temporary_path.unlink(missing_ok=True)
    video_ball_detection_job_store.update(
        job_id,
        success=False,
        status=status,
        error_message=message,
        message=message,
    )
    try:
        _update_analysis_metadata(
            analysis_id,
            ball_detection_status="detection_failed",
            ball_detection_job_id=job_id,
            ball_detection_completed_at=_iso(utc_now()),
            updated_at=_iso(utc_now()),
        )
    except VideoBallDetectionError:
        pass


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
