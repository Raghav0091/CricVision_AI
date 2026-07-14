from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DeliveryReport:
    delivery_id: str
    tracking_quality: str = "Unavailable"
    calibration_quality: str = "Unavailable"
    replay_path: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
