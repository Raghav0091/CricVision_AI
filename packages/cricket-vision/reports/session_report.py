from dataclasses import dataclass, field

from .delivery_report import DeliveryReport


@dataclass(frozen=True)
class SessionReport:
    session_id: str
    deliveries: list[DeliveryReport] = field(default_factory=list)
