"""Central registry for the models used by CricVision AI.

The public ``path``/``local_path`` values are stable, project-relative paths.
``path_candidates`` lets older/local weight filenames continue to work without
copying large model files. Remote metadata is only descriptive here; status
checks never download weights.
"""

from copy import deepcopy
from pathlib import Path
from typing import Optional

from Backends.src.config.paths import PROJECT_ROOT


MODEL_REGISTRY = {
    "current_best": {
        "name": "Current Best Ball + Stump Model",
        "local_path": "Models/cricket_objects/best.pt",
        "path": "Models/cricket_objects/best.pt",
        "type": "yolo_detection",
        "task": "bowling_analysis",
        "classes": ["ball", "stump"],
        "default": True,
    },
    "cricshot_ball": {
        "name": "CricShot10k Ball Detector",
        "local_path": "Models/CricShot10k/ball_detector.pt",
        "path": "Models/CricShot10k/ball_detector.pt",
        "path_candidates": ["Models/CricShot10k/Ball_Detection_Model.pt"],
        "remote_key": "cricshot_ball",
        "filename": "ball_detector.pt",
        "type": "yolo_detection",
        "task": "ball_detection",
        "classes": ["ball"],
        "default": False,
    },
    "ball_only_e2_1280_baseline": {
        "name": "Ball Only E2 1280 Baseline",
        "local_path": "Models/Copy of ball_only_E2_1280_baseline.pt",
        "path": "Models/Copy of ball_only_E2_1280_baseline.pt",
        "type": "yolo_detection",
        "task": "ball_detection",
        "classes": ["ball"],
        "default": False,
    },
    "cricshot_bat": {
        "name": "CricShot10k Bat Detector",
        "local_path": "Models/CricShot10k/bat_detector.pt",
        "path": "Models/CricShot10k/bat_detector.pt",
        "path_candidates": ["Models/CricShot10k/Bat_Detection_Model.pt"],
        "remote_key": "cricshot_bat",
        "filename": "bat_detector.pt",
        "type": "yolo_detection",
        "task": "bat_detection",
        "classes": ["bat"],
        "default": False,
    },
    "player_type": {
        "name": "CricShot10k Player Type Detector",
        "local_path": "Models/CricShot10k/player_type_detector.pt",
        "path": "Models/CricShot10k/player_type_detector.pt",
        "path_candidates": ["Models/CricShot10k/Player_Type_Detection_Model.pt"],
        "remote_key": "player_type",
        "filename": "player_type_detector.pt",
        "type": "yolo_detection",
        "task": "player_detection",
        "classes": ["batter", "bowler", "keeper", "fielder", "umpire"],
        "default": False,
        "experimental": True,
        "status": "registered_not_wired",
    },
    "striker_segmentation": {
        "name": "Striker Bat Segmentation",
        "local_path": "Models/CricShot10k/striker_bat_segmentation.pt",
        "path": "Models/CricShot10k/striker_bat_segmentation.pt",
        "path_candidates": ["Models/CricShot10k/Striker_Bat_Segmentation_Model.pt"],
        "remote_key": "striker_segmentation",
        "filename": "striker_bat_segmentation.pt",
        "type": "yolo_segmentation",
        "task": "segmentation",
        "classes": ["batter", "bat"],
        "default": False,
        "experimental": True,
        "status": "registered_not_wired",
    },
    "shot_classifier": {
        "name": "EfficientNetV2 + GRU Shot Classifier",
        "local_path": "Models/CricShot10k/shot_classifier.keras",
        "path": "Models/CricShot10k/shot_classifier.keras",
        "path_candidates": [
            "Models/CricShot10k/Efficientnetv2-s_GRU_128_NEEDS_CROPPED_SEGMENTED_SHOTS.keras"
        ],
        "remote_key": "shot_classifier",
        "filename": "shot_classifier.keras",
        "type": "keras_sequence_classifier",
        "task": "shot_classification",
        "classes": [],
        "default": False,
        "lazy_only": True,
        "experimental": True,
        "status": "lazy_only_not_wired",
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

    raw_paths = [
        info.get("local_path"),
        info.get("path"),
        *info.get("path_candidates", []),
    ]
    paths = []
    seen = set()
    for raw_path in raw_paths:
        if not raw_path:
            continue
        candidate = PROJECT_ROOT / Path(raw_path)
        if candidate in seen:
            continue
        seen.add(candidate)
        paths.append(candidate)
    return paths


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
    """Return a non-downloading status record for every registered model."""
    statuses = {}
    for key, info in MODEL_REGISTRY.items():
        path = get_model_path(key)
        found = bool(path and path.is_file())
        remote_key = info.get("remote_key")
        remote_available = bool(remote_key)
        lazy_only = bool(info.get("lazy_only"))
        if found:
            status = "Local ready (lazy only)" if lazy_only else "Local ready"
            warning = ""
        elif remote_available:
            status = "Remote available (lazy only)" if lazy_only else "Remote available"
            warning = (
                f"{info['name']} is not local; it will download from Hugging Face "
                "the first time it is used."
            )
        else:
            status = "Missing"
            warning = f"Model file not found: {path or info['path']}"

        statuses[key] = {
            "name": info["name"],
            "found": found,
            "local_ready": found,
            "remote_available": remote_available,
            "lazy_only": lazy_only,
            "status": status,
            "path": str(path) if path else info["path"],
            "local_path": info.get("local_path", info["path"]),
            "remote_key": remote_key,
            "filename": info.get("filename"),
            "warning": warning,
        }
    return statuses
