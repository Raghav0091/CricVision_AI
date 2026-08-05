"""Safe API surface for Pitch-Space Delivery Analysis Lab V1."""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse

from ..schemas.pitch_space_analysis import (
    PitchSpaceDeliveryAnalysisV1,
    RecentPitchSpaceAnalyses,
)
from ..services.pitch_space_analysis_service import (
    clear_pitch_space_analysis,
    list_recent_pitch_space_analyses,
    load_pitch_space_analysis,
    overlay_path,
    run_pitch_space_analysis,
    source_video_path,
)
from ..services.video_analysis_service import (
    VideoAnalysisServiceError,
    prepare_video,
)


router = APIRouter(prefix="/pitch-space-analysis", tags=["pitch-space-analysis"])


def _raise_http(exc: VideoAnalysisServiceError) -> None:
    raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.post("/upload", response_model=PitchSpaceDeliveryAnalysisV1, status_code=201)
def upload_and_run_pitch_space_analysis(
    video: Annotated[UploadFile, File()],
) -> PitchSpaceDeliveryAnalysisV1:
    started = time.perf_counter()
    try:
        prepared = prepare_video(video.file, video.filename)
        upload_ms = (time.perf_counter() - started) * 1000
        return run_pitch_space_analysis(prepared.analysis_id, upload_ms=upload_ms)
    except VideoAnalysisServiceError as exc:
        _raise_http(exc)


@router.get("/recent", response_model=RecentPitchSpaceAnalyses)
def recent_pitch_space_analyses(
    limit: int = Query(default=20, ge=1, le=100),
) -> RecentPitchSpaceAnalyses:
    return list_recent_pitch_space_analyses(limit)


@router.post("/{analysis_id}/run", response_model=PitchSpaceDeliveryAnalysisV1)
def run_existing_pitch_space_analysis(analysis_id: str) -> PitchSpaceDeliveryAnalysisV1:
    try:
        return run_pitch_space_analysis(analysis_id)
    except VideoAnalysisServiceError as exc:
        _raise_http(exc)


@router.get("/{analysis_id}", response_model=PitchSpaceDeliveryAnalysisV1)
def get_pitch_space_analysis(analysis_id: str) -> PitchSpaceDeliveryAnalysisV1:
    try:
        return load_pitch_space_analysis(analysis_id)
    except VideoAnalysisServiceError as exc:
        _raise_http(exc)


@router.post("/{analysis_id}/clear", status_code=204)
def clear_existing_pitch_space_analysis(analysis_id: str) -> Response:
    try:
        clear_pitch_space_analysis(analysis_id)
        return Response(status_code=204)
    except VideoAnalysisServiceError as exc:
        _raise_http(exc)


@router.get("/{analysis_id}/video")
def get_pitch_space_source_video(analysis_id: str) -> FileResponse:
    try:
        path = source_video_path(analysis_id)
        return FileResponse(path)
    except VideoAnalysisServiceError as exc:
        _raise_http(exc)


@router.get("/{analysis_id}/overlay")
def get_pitch_space_overlay(analysis_id: str) -> FileResponse:
    try:
        path = overlay_path(analysis_id)
        return FileResponse(path, media_type="image/jpeg")
    except VideoAnalysisServiceError as exc:
        _raise_http(exc)
