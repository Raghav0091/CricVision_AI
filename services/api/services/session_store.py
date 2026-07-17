from threading import Lock
from uuid import uuid4

from collections.abc import Callable
from typing import Any

from ..schemas.session import SessionCreate, SessionDeliveryRecord, SessionRecord, utc_now


class SessionStore:
    """Process-local store for the architecture scaffold."""

    def __init__(self) -> None:
        self._records: dict[str, SessionRecord] = {}
        self._lock = Lock()

    def create(self, request: SessionCreate) -> SessionRecord:
        now = utc_now()
        is_experimental = request.session_type == "experimental_delivery_test"
        record = SessionRecord(
            id=f"session_{uuid4().hex}",
            created_at=now,
            updated_at=now,
            status="capturing" if is_experimental else "created",
            **request.model_dump(),
        )
        with self._lock:
            self._records[record.id] = record
        return record.model_copy(deep=True)

    def get(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            record = self._records.get(session_id)
            return record.model_copy(deep=True) if record else None

    def list(self) -> list[SessionRecord]:
        with self._lock:
            records = [record.model_copy(deep=True) for record in self._records.values()]
        return sorted(records, key=lambda record: record.created_at, reverse=True)

    def add_delivery(self, session_id: str, delivery: SessionDeliveryRecord) -> SessionRecord | None:
        with self._lock:
            record = self._records.get(session_id)
            if record is None:
                return None
            deliveries = [item for item in record.deliveries if item.delivery_index != delivery.delivery_index]
            deliveries.append(delivery)
            deliveries.sort(key=lambda item: item.delivery_index)
            updated = record.model_copy(update={
                "deliveries": deliveries,
                "delivery_count": len(deliveries),
                "analysis_status": "processing",
                "updated_at": utc_now(),
            })
            self._records[session_id] = updated
            return updated.model_copy(deep=True)

    def complete_capture(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            record = self._records.get(session_id)
            if record is None:
                return None
            updated = record.model_copy(update={
                "status": "complete",
                "capture_status": "capture_complete",
                "updated_at": utc_now(),
            })
            self._records[session_id] = updated
            return updated.model_copy(deep=True)

    def synchronize(
        self,
        session_id: str,
        job_lookup: Callable[[str], dict[str, Any] | None],
    ) -> SessionRecord | None:
        with self._lock:
            record = self._records.get(session_id)
            if record is None:
                return None
            deliveries = []
            changed = False
            for delivery in record.deliveries:
                job = job_lookup(delivery.job_id) if delivery.job_id else None
                if not job:
                    deliveries.append(delivery)
                    continue
                status = job.get("status")
                if status == "processing":
                    updates = {
                        "analysis_status": "processing",
                        "progress": int(job.get("progress", 0)),
                    }
                elif job.get("success") and status == "ready":
                    updates = {
                        "analysis_status": "ready",
                        "progress": 100,
                        "processed_video_url": job.get("processed_video_url"),
                        "frames_processed": int(job.get("processed_frames", 0)),
                        "frames_with_ball": int(job.get("frames_with_ball", 0)),
                        "best_confidence": float(job.get("best_confidence", 0.0)),
                        "average_confidence": float(job.get("average_confidence", 0.0)),
                        "model_path_used": job.get("model_path_used"),
                        "error_message": None,
                    }
                else:
                    updates = {
                        "analysis_status": "failed",
                        "progress": 100,
                        "error_message": job.get("message") or "Ball detection failed.",
                    }
                updated_delivery = delivery.model_copy(update=updates)
                changed = changed or updated_delivery != delivery
                deliveries.append(updated_delivery)

            ready = sum(item.analysis_status == "ready" for item in deliveries)
            processing = sum(item.analysis_status in {"queued", "processing"} for item in deliveries)
            failed = sum(item.analysis_status == "failed" for item in deliveries)
            if not deliveries:
                analysis_status = "not_started"
            elif processing:
                analysis_status = "partially_ready" if ready else "processing"
            elif failed == len(deliveries):
                analysis_status = "failed"
            elif failed:
                analysis_status = "partially_ready"
            else:
                analysis_status = "ready"
            updated_record = record.model_copy(update={
                "deliveries": deliveries,
                "delivery_count": len(deliveries),
                "analysis_status": analysis_status,
                "updated_at": utc_now() if changed else record.updated_at,
            })
            self._records[session_id] = updated_record
            return updated_record.model_copy(deep=True)


session_store = SessionStore()
