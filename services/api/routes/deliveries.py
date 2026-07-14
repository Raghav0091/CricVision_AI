from fastapi import APIRouter, HTTPException

from ..schemas.delivery import DeliveryCreate, DeliveryRecord
from ..services.delivery_store import delivery_store


router = APIRouter(prefix="/deliveries", tags=["deliveries"])


@router.post("", response_model=DeliveryRecord, status_code=201)
def create_delivery(request: DeliveryCreate) -> DeliveryRecord:
    return delivery_store.create(request)


@router.get("/{delivery_id}", response_model=DeliveryRecord)
def get_delivery(delivery_id: str) -> DeliveryRecord:
    record = delivery_store.get(delivery_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Delivery not found")
    return record
