from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass(frozen=True)
class DetectionCandidate:
    class_name: str
    confidence: float
    bbox: BoundingBox
    frame_index: int

    @property
    def center(self) -> tuple[float, float]:
        return self.bbox.center
