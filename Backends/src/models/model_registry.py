"""Central registry for the models used by CricVision AI.

The public ``path`` values are stable, project-relative paths. ``path_candidates``
lets older/local weight filenames continue to work without changing the registry
contract or copying large model files.
"""

from copy import deepcopy
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_REGISTRY = {
    "current_best": {
        "name": "Current Best Ball + Stump Model",
        "path": "Models/cricket_objects/best.pt",
        "type": "yolo_detection",
        "task": "bowling_analysis",
        "classes": ["ball", "stump"],
        "default": True,
    },
    "cricshot_ball": {
        "name": "CricShot10k Ball Detector",
        "path": "Models/cricshot10k/ball_detector.pt",
        "path_candidates": ["Models/CricShot10k/Ball_Detection_Model.pt"],
        "type": "yolo_detection",
        "task": "ball_detection_testing",
        "classes": ["ball"],
        "default": False,
    },
    "cricshot_bat": {
        "name": "CricShot10k Bat Detector",
        "path": "Models/cricshot10k/bat_detector.pt",
        "path_candidates": ["Models/CricShot10k/Bat_Detection_Model.pt"],
        "type": "yolo_detection",
        "task": "bat_detection",
        "classes": ["bat"],
        "default": False,
    },
    "player_type": {
        "name": "CricShot10k Player Type Detector",
        "path": "Models/cricshot10k/player_type_detector.pt",
        "path_candidates": ["Models/CricShot10k/Player_Type_Detection_Model.pt"],
        "type": "yolo_detection",
        "task": "player_detection",
        "classes": ["batter", "bowler", "keeper", "fielder", "umpire"],
        "default": False,
    },
    "striker_segmentation": {
        "name": "Striker Bat Segmentation",
        "path": "Models/cricshot10k/striker_bat_segmentation.pt",
        "path_candidates": ["Models/CricShot10k/Striker_Bat_Segmentation_Model.pt"],
        "type": "yolo_segmentation",
        "task": "shot_preprocessing",
        "classes": ["batter", "bat"],
        "default": False,
    },
    "shot_classifier": {
        "name": "EfficientNetV2 + GRU Shot Classifier",
        "path": "Models/cricshot10k/shot_classifier.keras",
        "path_candidates": [
            "Models/CricShot10k/Efficientnetv2-s_GRU_128_NEEDS_CROPPED_SEGMENTED_SHOTS.keras"
        ],
        "type": "keras_sequence_classifier",
        "task": "shot_classification",
        "classes": [],
        "default": False,
    },
}


def get_model_info(model_key: str) -> Optional[dict]:
    """Return a defensive copy of a registry entry, or ``None`` if unknown."""
    info = MODEL_REGISTRY.get(model_key)
    return deepcopy(info) if info is not None else None


def get_available_models(task: Optional[str] = None) -> dict:
    """Return registered models, optionally filtered by task.

    Availability here means registered/configured; use :func:`model_exists` when
    the caller needs to check whether the weight file is present.
    """
    return {
        key: deepcopy(info)
        for key, info in MODEL_REGISTRY.items()
        if task is None or info.get("task") == task
    }


def _candidate_paths(model_key: str) -> list[Path]:
    info = MODEL_REGISTRY.get(model_key)
    if info is None:
        return []

    paths = [info["path"], *info.get("path_candidates", [])]
    return [PROJECT_ROOT / Path(path) for path in paths]


def get_model_path(model_key: str) -> Optional[Path]:
    """Resolve a model path without raising for unknown or missing models.

    If no candidate exists, the canonical expected path is returned so callers
    can display a useful warning.
    """
    candidates = _candidate_paths(model_key)
    if not candidates:
        return None

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def model_exists(model_key: str) -> bool:
    path = get_model_path(model_key)
    return bool(path and path.is_file())


def validate_model_paths() -> dict:
    """Return a status record for every model; never raise for missing files."""
    statuses = {}
    for key, info in MODEL_REGISTRY.items():
        path = get_model_path(key)
        found = bool(path and path.is_file())
        statuses[key] = {
            "name": info["name"],
            "found": found,
            "status": "Found" if found else "Missing",
            "path": str(path) if path else info["path"],
            "warning": "" if found else f"Model file not found: {path or info['path']}",
        }
    return statuses

