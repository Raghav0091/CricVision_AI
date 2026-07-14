from pathlib import Path
from typing import Any, Protocol

from .detection_schema import DetectionCandidate


class DetectionModel(Protocol):
    def predict(self, frame: Any) -> list[DetectionCandidate]: ...


class DetectorAdapter:
    """Model-agnostic boundary. It never loads missing weights or invents detections."""

    def __init__(self, model: DetectionModel | None = None, model_path: str | Path | None = None) -> None:
        self.model = model
        self.model_path = Path(model_path) if model_path else None

    @property
    def available(self) -> bool:
        return self.model is not None and (self.model_path is None or self.model_path.is_file())

    def detect(self, frame: Any) -> list[DetectionCandidate]:
        if not self.available:
            return []
        return self.model.predict(frame)
