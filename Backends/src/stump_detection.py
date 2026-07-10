"""Optional dedicated stump detector wrapper — no fake detections."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from Backends.src.config.paths import MODELS_DIR
from Backends.src.video_pipeline.detection_pipeline import map_model_classes

STUMP_DETECTOR_FILENAME = "best.pt"
STUMP_DETECTOR_DIR = MODELS_DIR / "stump_detector"


def get_stump_detector_model_path() -> Path:
    return STUMP_DETECTOR_DIR / STUMP_DETECTOR_FILENAME


def stump_detector_available() -> bool:
    return get_stump_detector_model_path().is_file()


def model_has_stump_classes(model: Any) -> bool:
    if model is None:
        return False
    try:
        class_names = map_model_classes(model)
    except Exception:
        return False
    return any(name == "stump" for name in class_names.values())


@lru_cache(maxsize=1)
def _load_stump_detector_cached():
    path = get_stump_detector_model_path()
    if not path.is_file():
        return None
    try:
        from ultralytics import YOLO

        return YOLO(str(path))
    except Exception:
        return None


def load_stump_detector():
    """Load dedicated stump detector once; returns None if weights missing."""
    if not stump_detector_available():
        return None
    return _load_stump_detector_cached()


def run_stump_detection(frame: Any, model=None, conf: float = 0.25) -> list[dict[str, Any]]:
    """Run stump detection on one frame; empty list if model/frame missing."""
    if frame is None:
        return []
    detector = model if model is not None else load_stump_detector()
    if detector is None:
        return []
    try:
        class_names = map_model_classes(detector)
        confidence = float(conf) if conf is not None else 0.25
        results = detector.predict(frame, conf=confidence, verbose=False, imgsz=640)
        if not results:
            return []
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        detections: list[dict[str, Any]] = []
        for box in result.boxes:
            class_id = int(box.cls[0].cpu().numpy())
            class_name = class_names.get(class_id)
            if class_name != "stump":
                continue
            score = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
            detections.append(
                {
                    "class_name": "stump",
                    "confidence": score,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "box": (x1, y1, x2, y2),
                    "center": {"x": int((x1 + x2) / 2), "y": int((y1 + y2) / 2)},
                    "source": "stump_detector" if model is None else "live_calibration",
                }
            )
        return detections
    except Exception:
        return []


def resolve_live_stump_detections(
    frame: Any,
    primary_model=None,
    conf: float = 0.25,
) -> list[dict[str, Any]]:
    """Reuse ball+stump model when it already exposes stump; else dedicated detector."""
    if frame is None:
        return []
    if model_has_stump_classes(primary_model):
        return run_stump_detection(frame, model=primary_model, conf=conf)
    if stump_detector_available():
        return run_stump_detection(frame, model=None, conf=conf)
    return []
