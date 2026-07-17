"""Detection-only annotation for short experimental delivery clips."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import shutil
import subprocess
from threading import Lock
from typing import Any
from collections.abc import Callable

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_CLIP_DIR = PROJECT_ROOT / "outputs" / "delivery_clips" / "raw"
PROCESSED_CLIP_DIR = PROJECT_ROOT / "outputs" / "delivery_clips" / "processed"
BALL_MODEL_PATHS = (
    PROJECT_ROOT / "Models" / "Copy of ball_only_E2_1280_baseline.pt",
    PROJECT_ROOT / "Models" / "ball_detector" / "best.pt",
    PROJECT_ROOT / "Models" / "cricket_objects" / "best.pt",
)
BALL_CLASS_NAMES = {"ball", "cricket_ball", "sports_ball"}
_INFERENCE_LOCK = Lock()


def resolve_ball_model_path() -> Path | None:
    return next((path for path in BALL_MODEL_PATHS if path.is_file()), None)


@lru_cache(maxsize=1)
def _load_ball_model(model_path: str):
    # Lazy by design: importing the API never imports Ultralytics or loads weights.
    from ultralytics import YOLO

    return YOLO(model_path)


def process_ball_detection_clip(
    raw_path: Path,
    *,
    delivery_index: int,
    output_stem: str,
    processing_mode: str = "quality",
    on_progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    model_path = resolve_ball_model_path()
    if model_path is None:
        return _failure(
            "ball_detector_missing",
            delivery_index,
            (
                "Ball detector model not found at "
                "Models/Copy of ball_only_E2_1280_baseline.pt"
            ),
        )

    try:
        model = _load_ball_model(str(model_path))
    except Exception as exc:
        return _failure(
            "model_inference_failed",
            delivery_index,
            f"Ball detector could not be loaded: {type(exc).__name__}.",
            model_path,
        )

    PROCESSED_CLIP_DIR.mkdir(parents=True, exist_ok=True)
    intermediate_path = PROCESSED_CLIP_DIR / f"{output_stem}_intermediate.avi"
    processed_path = PROCESSED_CLIP_DIR / f"{output_stem}_ball_detected.mp4"
    capture = cv2.VideoCapture(str(raw_path))
    if not capture.isOpened():
        return _failure(
            "video_processing_failed",
            delivery_index,
            "Could not open the uploaded delivery clip.",
            model_path,
        )

    writer = None
    processing_failure = None
    frame_count = 0
    processed_frames = 0
    frames_with_ball = 0
    confidences: list[float] = []
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not 1 <= fps <= 240:
            fps = 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        settings = {
            "fast": {"imgsz": 640, "frame_stride": 3, "confidence": 0.15, "max_clip_seconds": 2.5},
            "balanced": {"imgsz": 768, "frame_stride": 2, "confidence": 0.15, "max_clip_seconds": 2.5},
            "quality": {"imgsz": 960, "frame_stride": 1, "confidence": 0.15, "max_clip_seconds": 2.5},
        }.get(processing_mode, {"imgsz": 960, "frame_stride": 1, "confidence": 0.15, "max_clip_seconds": 2.5})
        max_frames = max(1, round(fps * settings["max_clip_seconds"]))
        if width <= 0 or height <= 0:
            return _failure(
                "video_processing_failed",
                delivery_index,
                "Could not read video dimensions from the uploaded delivery clip.",
                model_path,
            )

        writer = cv2.VideoWriter(
            str(intermediate_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            processing_failure = _failure(
                "video_writer_failed",
                delivery_index,
                "Could not create the processed delivery video.",
                model_path,
            )
        else:
            with _INFERENCE_LOCK:
                while True:
                    ok, frame = capture.read()
                    if not ok or frame_count >= max_frames:
                        break
                    frame_count += 1
                    if (frame_count - 1) % settings["frame_stride"] == 0:
                        try:
                            results = model.predict(
                                source=frame,
                                imgsz=settings["imgsz"],
                                conf=settings["confidence"],
                                verbose=False,
                            )
                        except Exception as exc:
                            processing_failure = _failure(
                                "model_inference_failed",
                                delivery_index,
                                f"Ball detection failed: {type(exc).__name__}.",
                                model_path,
                            )
                            break
                        processed_frames += 1
                        best = _best_ball_detection(results, getattr(model, "names", {}))
                        if best is not None:
                            frames_with_ball += 1
                            confidences.append(best["confidence"])
                            _draw_ball_detection(frame, best)
                    writer.write(frame)
                    if on_progress and frame_count % 6 == 0:
                        total = min(source_frame_count, max_frames) if source_frame_count > 0 else max_frames
                        on_progress(min(95, round(frame_count / total * 95)))
    except Exception as exc:
        processing_failure = _failure(
            "video_processing_failed",
            delivery_index,
            f"Could not process uploaded delivery clip: {type(exc).__name__}.",
            model_path,
        )
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if processing_failure is not None:
        intermediate_path.unlink(missing_ok=True)
        return processing_failure

    if frame_count == 0:
        intermediate_path.unlink(missing_ok=True)
        return _failure(
            "video_processing_failed",
            delivery_index,
            "Could not read frames from uploaded delivery clip.",
            model_path,
        )

    try:
        _transcode_browser_mp4(intermediate_path, processed_path)
    except Exception as exc:
        processed_path.unlink(missing_ok=True)
        return _failure(
            "video_writer_failed",
            delivery_index,
            f"Could not encode a browser-playable processed clip: {type(exc).__name__}.",
            model_path,
        )
    finally:
        intermediate_path.unlink(missing_ok=True)

    return {
        "success": True,
        "status": "ready",
        "delivery_index": delivery_index,
        "model_path_used": model_path.relative_to(PROJECT_ROOT).as_posix(),
        "frame_count": frame_count,
        "processed_frames": processed_frames,
        "frames_with_ball": frames_with_ball,
        "best_confidence": round(max(confidences, default=0.0), 4),
        "average_confidence": round(
            sum(confidences) / len(confidences) if confidences else 0.0,
            4,
        ),
        "processed_filename": processed_path.name,
        "message": "Ball detection completed.",
    }


def _best_ball_detection(results, model_names) -> dict[str, Any] | None:
    best = None
    for result in results or []:
        names = getattr(result, "names", None) or model_names
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        classes = _to_list(getattr(boxes, "cls", []))
        confidence_values = _to_list(getattr(boxes, "conf", []))
        coordinates = _to_list(getattr(boxes, "xyxy", []))
        for class_id, confidence, xyxy in zip(
            classes,
            confidence_values,
            coordinates,
        ):
            class_name = _class_name(names, int(class_id))
            if not _is_ball_class(class_name, names):
                continue
            confidence = float(confidence)
            if best is not None and confidence <= best["confidence"]:
                continue
            x1, y1, x2, y2 = (round(float(value)) for value in xyxy[:4])
            best = {
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2),
            }
    return best


def _draw_ball_detection(frame, detection: dict[str, Any]) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = detection["bbox"]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1, min(width - 1, x2))
    y2 = max(y1, min(height - 1, y2))
    yellow = (0, 230, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), yellow, 2, cv2.LINE_AA)
    center = (round((x1 + x2) / 2), round((y1 + y2) / 2))
    cv2.circle(frame, center, 3, yellow, -1, cv2.LINE_AA)
    label = f"Ball {detection['confidence']:.2f}"
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        1,
    )
    label_y = max(text_height + baseline + 3, y1)
    cv2.rectangle(
        frame,
        (x1, label_y - text_height - baseline - 4),
        (min(width - 1, x1 + text_width + 6), label_y + 2),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x1 + 3, label_y - baseline - 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        yellow,
        1,
        cv2.LINE_AA,
    )


def _transcode_browser_mp4(source: Path, destination: Path) -> None:
    try:
        import imageio_ffmpeg

        executable = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("ffmpeg_unavailable")
    completed = subprocess.run(
        [
            executable,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0 or not destination.is_file():
        raise RuntimeError(completed.stderr.strip() or "ffmpeg_failed")


def _failure(
    status: str,
    delivery_index: int,
    message: str,
    model_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "success": False,
        "status": status,
        "delivery_index": delivery_index,
        "model_path_used": (
            model_path.relative_to(PROJECT_ROOT).as_posix()
            if model_path is not None
            else None
        ),
        "message": message,
    }


def _to_list(value) -> list:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value or [])


def _class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _is_ball_class(class_name: str, names) -> bool:
    normalized = class_name.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in BALL_CLASS_NAMES:
        return True
    return isinstance(names, (dict, list, tuple)) and len(names) == 1
