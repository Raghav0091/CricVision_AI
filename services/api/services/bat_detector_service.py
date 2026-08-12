"""Lazy bat detection for using a cricket bat as a calibration reference object.

Mirrors the guided-detection pattern in `stump_detector_service.py`. A bat
substitutes for stumps as the vertical calibration anchor when real stumps are
unavailable — the caller supplies the bat's actual measured length, since
unlike a stump's fixed 0.7112m height, bat length varies by bat.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_BAT_MODEL_KEY = "cricshot10k"

BAT_DETECTOR_MODEL_PATHS: dict[str, Path] = {
    "cricshot10k": PROJECT_ROOT / "Models" / "CricShot10k" / "Bat_Detection_Model.pt",
    "bat_v2": PROJECT_ROOT / "Models" / "Bat_detector" / "best (2).pt",
}

# Exact match, not substring: bat_v2's classes include both "bat" and
# "batter" — a substring check would wrongly treat batter boxes as bats.
BAT_CLASS_NAMES = ("bat",)

DEFAULT_DETECT_CONF = 0.25

# Striker (far end) sits smaller/lower-confidence in frame than the near
# non-striker end; give it a lower detection threshold by default rather
# than forcing one global conf that under-serves the harder end.
DEFAULT_CONF_BY_END: dict[str, float] = {
    "striker": 0.05,
    "non_striker": DEFAULT_DETECT_CONF,
}

# Reuse the same default search regions as stump guided detection: far end
# (near camera... actually mid-upper) and near end (lower, larger in frame).
DEFAULT_FAR_GUIDE = {"x": 0.18, "y": 0.15, "width": 0.64, "height": 0.50}
DEFAULT_NEAR_GUIDE = {"x": 0.05, "y": 0.45, "width": 0.90, "height": 0.55}


def default_bat_guides() -> dict[str, dict[str, float]]:
    return {
        "striker": dict(DEFAULT_FAR_GUIDE),
        "non_striker": dict(DEFAULT_NEAR_GUIDE),
    }


@lru_cache(maxsize=len(BAT_DETECTOR_MODEL_PATHS))
def _load_model(model_path: str):
    from ultralytics import YOLO

    return YOLO(model_path)


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


def _is_bat_class(class_name: str, names) -> bool:
    normalized = class_name.strip().lower()
    if normalized in BAT_CLASS_NAMES:
        return True
    return isinstance(names, (dict, list, tuple)) and len(names) == 1


def _normalized_box_to_pixels(
    box: dict[str, float], frame_width: int, frame_height: int
) -> tuple[int, int, int, int]:
    x1 = max(0, min(frame_width - 1, round(box["x"] * frame_width)))
    y1 = max(0, min(frame_height - 1, round(box["y"] * frame_height)))
    x2 = max(x1 + 1, min(frame_width, round((box["x"] + box["width"]) * frame_width)))
    y2 = max(y1 + 1, min(frame_height, round((box["y"] + box["height"]) * frame_height)))
    return x1, y1, x2, y2


def _detect_end(
    model,
    image: Image.Image,
    source_box: str,
    normalized_box: dict[str, float],
    frame_width: int,
    frame_height: int,
    conf: float,
) -> dict[str, Any]:
    crop_bounds = _normalized_box_to_pixels(normalized_box, frame_width, frame_height)
    x1, y1, x2, y2 = crop_bounds
    crop = image.crop(crop_bounds)
    results = model.predict(source=crop, verbose=False, conf=conf)

    best: dict[str, Any] | None = None
    for result in results or []:
        names = getattr(result, "names", None) or getattr(model, "names", {})
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        classes = _to_list(getattr(boxes, "cls", []))
        confidences = _to_list(getattr(boxes, "conf", []))
        coordinates = _to_list(getattr(boxes, "xyxy", []))
        for class_id, confidence, xyxy in zip(classes, confidences, coordinates):
            class_name = _class_name(names, int(class_id))
            if not _is_bat_class(class_name, names):
                continue
            confidence = float(confidence)
            if best is not None and confidence <= best["confidence"]:
                continue
            bx1, by1, bx2, by2 = (float(value) for value in xyxy[:4])
            full_x1 = max(0, min(frame_width, round(x1 + bx1)))
            full_y1 = max(0, min(frame_height, round(y1 + by1)))
            full_x2 = max(full_x1, min(frame_width, round(x1 + bx2)))
            full_y2 = max(full_y1, min(frame_height, round(y1 + by2)))
            best = {
                "found": True,
                "confidence": round(confidence, 4),
                "bbox": {
                    "x": full_x1,
                    "y": full_y1,
                    "width": full_x2 - full_x1,
                    "height": full_y2 - full_y1,
                },
                "source_box": source_box,
                "class_name": class_name,
            }

    return best or {
        "found": False,
        "confidence": 0.0,
        "bbox": None,
        "source_box": source_box,
    }


def detect_bats_guided(
    image: Image.Image,
    guides: dict[str, dict[str, float]] | None = None,
    *,
    conf: float | dict[str, float] = DEFAULT_CONF_BY_END,
    model_key: str = DEFAULT_BAT_MODEL_KEY,
) -> dict[str, Any]:
    """Detect one upright bat inside each guide box (striker end, non-striker end).

    Returns the same `found`/`bbox`/`confidence` shape as stump guided
    detection so calibration code can treat a bat bbox as a drop-in
    replacement anchor source. The bbox base (bottom edge) is the
    ground-contact point (world height 0); the bbox top corresponds to the
    caller-supplied measured bat length, not a fixed constant.
    """
    guides = guides or default_bat_guides()
    model_path = BAT_DETECTOR_MODEL_PATHS.get(model_key)
    if model_path is None:
        known = ", ".join(sorted(BAT_DETECTOR_MODEL_PATHS))
        return {
            "success": False,
            "status": "bat_detector_error",
            "message": f"Unknown bat detector key '{model_key}'. Expected one of: {known}.",
            "detections": None,
        }
    if not model_path.is_file():
        return {
            "success": False,
            "status": "bat_detector_missing",
            "message": f"Bat detector model not found at {model_path}.",
            "detections": None,
        }

    try:
        model = _load_model(str(model_path))
    except Exception as exc:
        return {
            "success": False,
            "status": "bat_detector_error",
            "message": f"Bat detector could not be loaded: {type(exc).__name__}.",
            "detections": None,
        }

    conf_by_end = conf if isinstance(conf, dict) else {"striker": conf, "non_striker": conf}

    frame_width, frame_height = image.size
    try:
        detections = {
            end: _detect_end(
                model,
                image,
                end,
                guides[end],
                frame_width,
                frame_height,
                conf_by_end.get(end, DEFAULT_DETECT_CONF),
            )
            for end in ("striker", "non_striker")
        }
    except Exception as exc:
        return {
            "success": False,
            "status": "bat_detector_error",
            "message": f"Bat detection failed: {type(exc).__name__}.",
            "detections": None,
        }

    both = all(item["found"] for item in detections.values())
    return {
        "success": both,
        "status": "detection_complete" if both else "bats_not_found",
        "message": (
            "Both bats detected."
            if both
            else "Could not detect a bat at both ends. Make sure each bat "
            "stands upright inside its guide box, in view, and clearly separated "
            "from the background."
        ),
        "detections": detections,
    }


def bat_anchor_points(
    detection: dict[str, Any],
    *,
    bat_height_m: float,
) -> dict[str, Any] | None:
    """Convert a found bat bbox into base (0m) / top (bat_height_m) pixel anchors."""
    if not detection.get("found") or not detection.get("bbox"):
        return None
    bbox = detection["bbox"]
    center_x = bbox["x"] + bbox["width"] / 2
    return {
        "base_px": {"x": center_x, "y": bbox["y"] + bbox["height"]},
        "top_px": {"x": center_x, "y": bbox["y"]},
        "base_height_m": 0.0,
        "top_height_m": bat_height_m,
    }
