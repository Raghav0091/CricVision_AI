from threading import Lock
from uuid import uuid4

from ..schemas.delivery import DeliveryCreate, DeliveryRecord
from ..schemas.session import utc_now


class DeliveryStore:
    def __init__(self) -> None:
        self._records: dict[str, DeliveryRecord] = {}
        self._lock = Lock()

    def create(self, request: DeliveryCreate) -> DeliveryRecord:
        record = DeliveryRecord(id=str(uuid4()), created_at=utc_now(), **request.model_dump())
        with self._lock:
            self._records[record.id] = record
        return record

    def get(self, delivery_id: str) -> DeliveryRecord | None:
        return self._records.get(delivery_id)


delivery_store = DeliveryStore()
