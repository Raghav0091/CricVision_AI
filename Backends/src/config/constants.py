"""Constants shared by more than one production module."""

DETECTION_PRESETS = {
    "Fast Bowling Mode": {
        "imgsz": 960,
        "confidence": 0.15,
    },
    "Balanced Mode": {
        "imgsz": 768,
        "confidence": 0.25,
    },
    "High Precision Mode": {
        "imgsz": 960,
        "confidence": 0.35,
    },
}

ENSEMBLE_MODEL_NAME = "Ensemble: All Ball Models + Stumps"
LOW_CONFIDENCE_REVIEW_THRESHOLD = 0.35
MAX_REASONABLE_BALL_JUMP_PX = 180
