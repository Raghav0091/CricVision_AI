from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class TrajectoryEstimate:
    points: list[tuple[float, float, float]] = field(default_factory=list)
    quality: str = "Unavailable"
    reason: str = "physics_trajectory_not_connected"


class PhysicsTrajectoryAdapter:
    """Boundary for the existing backend physics module; no estimates are fabricated."""

    def __init__(self, estimator: Callable[..., TrajectoryEstimate] | None = None) -> None:
        self.estimator = estimator

    def estimate(self, *args: object, **kwargs: object) -> TrajectoryEstimate:
        if self.estimator is None:
            return TrajectoryEstimate()
        return self.estimator(*args, **kwargs)
