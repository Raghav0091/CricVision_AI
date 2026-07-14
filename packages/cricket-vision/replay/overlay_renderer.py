from dataclasses import dataclass, field


@dataclass(frozen=True)
class OverlayFrame:
    frame_index: int
    observed_points: list[tuple[float, float]] = field(default_factory=list)
    fitted_points: list[tuple[float, float]] = field(default_factory=list)


@dataclass(frozen=True)
class ReplayOverlay:
    frames: list[OverlayFrame] = field(default_factory=list)
    tracking_quality: str = "Unavailable"
    message: str = "Replay rendering is not implemented yet."
