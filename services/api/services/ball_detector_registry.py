"""Trusted ball-detector registry for the Video Analysis workflow."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BALL_DETECTOR_MODEL_KEY = "e4c_best_overall"
MODEL_LOAD_LOCK = Lock()


@dataclass(frozen=True)
class BallDetectorModel:
    key: str
    display_name: str
    path: Path
    description: str

    @property
    def model_file(self) -> str:
        return self.path.name


BALL_DETECTOR_MODELS: dict[str, BallDetectorModel] = {
    "e2_baseline": BallDetectorModel(
        key="e2_baseline",
        display_name="E2 — Original 1280 Baseline",
        path=PROJECT_ROOT / "Models" / "Copy of ball_only_E2_1280_baseline.pt",
        description="Original 1280 baseline detector",
    ),
    "e3_motion_blur": BallDetectorModel(
        key="e3_motion_blur",
        display_name="E3 — Motion Blur Robustness",
        path=PROJECT_ROOT / "Models" / "ball_only_E3_1280_motion_blur.pt",
        description="Detector trained for improved motion-blur robustness",
    ),
    "e4c_best_overall": BallDetectorModel(
        key="e4c_best_overall",
        display_name="E4C — Best Overall",
        path=(
            PROJECT_ROOT
            / "Models"
            / "Copy of ball_only_E4C_1280_random_sampling_control.pt"
        ),
        description="Current strongest overall candidate",
    ),
}


class InvalidBallDetectorModelKey(ValueError):
    pass


class BallDetectorModelMissing(FileNotFoundError):
    def __init__(self, model: BallDetectorModel) -> None:
        super().__init__(
            f"Selected {model.display_name} ball detector could not be found."
        )
        self.model = model


def get_ball_detector_model(model_key: str) -> BallDetectorModel:
    """Resolve a registered model key to a verified repository-local weight file."""
    model = BALL_DETECTOR_MODELS.get(model_key)
    if model is None:
        allowed = ", ".join(BALL_DETECTOR_MODELS)
        raise InvalidBallDetectorModelKey(
            f"Unknown ball detector model key '{model_key}'. Allowed keys: {allowed}."
        )
    if not model.path.is_file():
        raise BallDetectorModelMissing(model)
    return model


@lru_cache(maxsize=len(BALL_DETECTOR_MODELS))
def _load_cached_ball_detector_model(model_key: str):
    model = get_ball_detector_model(model_key)
    # Lazy by design: importing the API never imports Ultralytics or loads weights.
    from ultralytics import YOLO

    return YOLO(str(model.path))


def load_ball_detector_model(model_key: str):
    """Load once per registered key, then reuse the YOLO instance."""
    model = get_ball_detector_model(model_key)
    with MODEL_LOAD_LOCK:
        return model, _load_cached_ball_detector_model(model_key)
