from dataclasses import dataclass, field
from typing import Protocol

from detection.detection_schema import DetectionCandidate


@dataclass(frozen=True)
class TrackingResult:
    accepted_points: list[tuple[int, float, float]] = field(default_factory=list)
    quality: str = "Unavailable"
    reason: str = "moving_ball_tracker_not_implemented"


class MovingBallTracker(Protocol):
    def track(self, candidates: list[DetectionCandidate]) -> TrackingResult: ...
