"""Approved detector models for the Video Analysis workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUTOMATIC_MODEL_KEY = "automatic"


@dataclass(frozen=True)
class BallDetectorModel:
    key: str
    display_name: str
    description: str
    paths: tuple[Path, ...]

    @property
    def available(self) -> bool:
        return any(path.is_file() for path in self.paths)


@dataclass(frozen=True)
class ResolvedBallDetectorModel:
    requested_key: str
    model_key: str
    selected_key: str
    display_name: str
    description: str
    path: Path
    fallback_reason: str | None = None


E2_PATH = PROJECT_ROOT / "Models" / "Copy of ball_only_E2_1280_baseline.pt"
E3_PATH = PROJECT_ROOT / "Models" / "ball_only_E3_1280_motion_blur.pt"
E4C_PATH = (
    PROJECT_ROOT
    / "Models"
    / "Copy of ball_only_E4C_1280_random_sampling_control.pt"
)
LEGACY_FALLBACK_PATH = PROJECT_ROOT / "Models" / "ball_detector" / "best.pt"


BALL_DETECTOR_MODELS: dict[str, BallDetectorModel] = {
    AUTOMATIC_MODEL_KEY: BallDetectorModel(
        key=AUTOMATIC_MODEL_KEY,
        display_name="Automatic",
        description="Prefer E2, then use the legacy detector when E2 is unavailable.",
        paths=(E2_PATH, LEGACY_FALLBACK_PATH),
    ),
    "e2": BallDetectorModel(
        key="e2",
        display_name="E2 - Original 1280 Baseline",
        description="Original 1280 baseline detector.",
        paths=(E2_PATH,),
    ),
    "e3": BallDetectorModel(
        key="e3",
        display_name="E3 - Motion Blur",
        description="Detector trained for motion-blur robustness.",
        paths=(E3_PATH,),
    ),
    "e4c": BallDetectorModel(
        key="e4c",
        display_name="E4C - Best Overall",
        description="Current strongest overall detector candidate.",
        paths=(E4C_PATH,),
    ),
}


class BallDetectorModelMissing(FileNotFoundError):
    pass


def list_ball_detector_models() -> tuple[BallDetectorModel, ...]:
    return tuple(BALL_DETECTOR_MODELS.values())


def resolve_ball_detector_model(
    requested_key: str | None,
) -> ResolvedBallDetectorModel:
    normalized = (requested_key or AUTOMATIC_MODEL_KEY).strip().lower()
    model = BALL_DETECTOR_MODELS.get(normalized)
    fallback_reason = None
    if model is None:
        fallback_reason = (
            f"Unknown detector key '{normalized}' was replaced with automatic selection."
        )
        model = BALL_DETECTOR_MODELS[AUTOMATIC_MODEL_KEY]

    path = next((candidate for candidate in model.paths if candidate.is_file()), None)
    if path is None:
        raise BallDetectorModelMissing(
            f"No local weight file is available for {model.display_name}."
        )

    selected_key = model.key
    if model.key == AUTOMATIC_MODEL_KEY:
        selected_key = "e2" if path == E2_PATH else "legacy"
        if path == LEGACY_FALLBACK_PATH:
            fallback_reason = fallback_reason or (
                "E2 is unavailable; automatic selection used the legacy detector."
            )

    return ResolvedBallDetectorModel(
        requested_key=normalized,
        model_key=model.key,
        selected_key=selected_key,
        display_name=model.display_name,
        description=model.description,
        path=path,
        fallback_reason=fallback_reason,
    )
