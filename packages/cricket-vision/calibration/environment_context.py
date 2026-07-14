from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CalibrationQuality(str, Enum):
    UNAVAILABLE = "Unavailable"
    POOR = "Poor"
    PARTIAL = "Partial"
    GOOD = "Good"


@dataclass(frozen=True)
class EnvironmentContext:
    frame_width: int
    frame_height: int
    striker_stump_center: tuple[float, float]
    non_striker_stump_center: tuple[float, float]
    pitch_axis: tuple[tuple[float, float], tuple[float, float]]
    pitch_corridor: tuple[tuple[float, float], ...]
    quality: CalibrationQuality
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
