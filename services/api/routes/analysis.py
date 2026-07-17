from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile

from ..schemas.analysis import (
    AnalysisJob,
    AnalysisRequest,
    BallDetectionClipResponse,
)
from ..services.ball_detection_clip import (
    RAW_CLIP_DIR,
    process_ball_detection_clip,
)
from ..services.job_store import clip_job_store, job_store
from ..services.session_store import session_store
from ..schemas.session import SessionDeliveryRecord


router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("", response_model=AnalysisJob, status_code=202)
def queue_analysis(request: AnalysisRequest) -> AnalysisJob:
    return job_store.create(request)


MAX_CLIP_BYTES = 200 * 1024 * 1024
ALLOWED_VIDEO_SUFFIXES = {".webm", ".mp4", ".mov", ".avi", ".mkv"}


@router.post("/ball-detection-clip", response_model=BallDetectionClipResponse)
def detect_ball_in_clip(
    request: Request,
    background_tasks: BackgroundTasks,
    video: Annotated[UploadFile, File()],
    delivery_index: Annotated[int, Form()],
    source_mode: Annotated[str, Form()] = "experimental_delivery_test",
    processing_mode: Annotated[str, Form()] = "quality",
    session_id: Annotated[str | None, Form()] = None,
) -> BallDetectionClipResponse:
    if not 1 <= delivery_index <= 6:
        return BallDetectionClipResponse(
            success=False,
            status="invalid_upload",
            delivery_index=delivery_index,
            message="delivery_index must be between 1 and 6.",
        )
    if source_mode != "experimental_delivery_test":
        return BallDetectionClipResponse(
            success=False,
            status="invalid_upload",
            delivery_index=delivery_index,
            message="Unsupported source_mode.",
        )
    if session_id is not None:
        session = session_store.get(session_id)
        if session is None or session.session_type != "experimental_delivery_test":
            return BallDetectionClipResponse(
                success=False,
                status="invalid_upload",
                delivery_index=delivery_index,
                message="Experimental delivery session not found.",
            )
    suffix = Path(video.filename or "delivery.webm").suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        suffix = ".webm"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_stem = f"delivery_{delivery_index:03d}_{timestamp}"
    RAW_CLIP_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_CLIP_DIR / f"{output_stem}_raw{suffix}"

    total_bytes = 0
    try:
        with raw_path.open("wb") as destination:
            while chunk := video.file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_CLIP_BYTES:
                    raw_path.unlink(missing_ok=True)
                    return BallDetectionClipResponse(
                        success=False,
                        status="upload_too_large",
                        delivery_index=delivery_index,
                        message="Uploaded delivery clip exceeds the 200 MB limit.",
                    )
                destination.write(chunk)
    except Exception as exc:
        raw_path.unlink(missing_ok=True)
        return BallDetectionClipResponse(
            success=False,
            status="invalid_upload",
            delivery_index=delivery_index,
            message=f"Could not save uploaded delivery clip: {type(exc).__name__}.",
        )

    if total_bytes == 0:
        raw_path.unlink(missing_ok=True)
        return BallDetectionClipResponse(
            success=False,
            status="invalid_upload",
            delivery_index=delivery_index,
            message="Uploaded delivery clip is empty.",
        )

    job = clip_job_store.create(delivery_index, session_id)
    raw_video_url = str(
        request.url_for(
            "delivery-clips-static",
            path=f"raw/{raw_path.name}",
        )
    )
    if session_id is not None:
        session_store.add_delivery(
            session_id,
            SessionDeliveryRecord(
                delivery_index=delivery_index,
                raw_video_url=raw_video_url,
                job_id=job["job_id"],
                analysis_status="processing",
            ),
        )
    processed_url_base = str(
        request.url_for(
            "delivery-clips-static",
            path="processed/placeholder.mp4",
        )
    ).rsplit("/", 1)[0]
    background_tasks.add_task(
        _run_ball_detection_job,
        job["job_id"], raw_path, delivery_index, output_stem, processing_mode, processed_url_base,
    )
    return BallDetectionClipResponse(**job)


def _run_ball_detection_job(job_id: str, raw_path: Path, delivery_index: int, output_stem: str, processing_mode: str, processed_url_base: str) -> None:
    try:
        result = process_ball_detection_clip(
            raw_path,
            delivery_index=delivery_index,
            output_stem=output_stem,
            processing_mode=processing_mode,
            on_progress=lambda progress: clip_job_store.update(job_id, progress=progress),
        )
    except Exception as exc:
        result = {
            "success": False,
            "status": "failed",
            "delivery_index": delivery_index,
            "message": f"Ball detection job failed: {type(exc).__name__}.",
        }
    processed_filename = result.pop("processed_filename", None)
    if processed_filename:
        result["processed_video_url"] = f"{processed_url_base}/{processed_filename}"
    if not result.get("success") and result.get("status") != "ball_detector_missing":
        result["status"] = "failed"
    # ponytail: job_id is already on the record; passing it again conflicts with
    # ClipJobStore.update's positional job_id argument and leaves jobs processing.
    clip_job_store.update(job_id, progress=100, **result)


@router.get("/jobs/{job_id}", response_model=BallDetectionClipResponse)
def get_ball_detection_job(job_id: str) -> BallDetectionClipResponse:
    record = clip_job_store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Ball detection job not found.")
    return BallDetectionClipResponse(**record)
