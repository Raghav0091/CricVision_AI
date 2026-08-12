"""Quick Test: upload prep for browser-recorded clips, no calibration required.

Detection and tracking reuse the existing /video-analysis endpoints directly
(no calibration is needed for either) — this router only exists to fix up a
browser-recorded clip before it reaches the shared uploader.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..schemas.video_analysis import VideoAnalysisPreparedResponse
from ..services.ball_detection_clip import transcode_browser_mp4
from ..services.video_analysis_service import VideoAnalysisServiceError, prepare_video


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/quick-test", tags=["quick-test"])


@router.post(
    "/prepare",
    response_model=VideoAnalysisPreparedResponse,
    status_code=201,
)
def prepare_quick_test_video(
    video: Annotated[UploadFile, File()],
) -> VideoAnalysisPreparedResponse:
    """Re-mux a browser-recorded clip before handing it to the shared uploader.

    MediaRecorder webm blobs often report zero frames via
    cv2.CAP_PROP_FRAME_COUNT even though every frame is readable, which makes
    the shared `prepare_video` metadata check reject them. Transcoding first
    gives it a well-formed file, same as real uploaded video files already are.
    """
    suffix = Path(video.filename or "clip.webm").suffix or ".webm"
    with tempfile.TemporaryDirectory(prefix="quick_test_upload_") as tmp_dir:
        raw_path = Path(tmp_dir) / f"raw{suffix}"
        with raw_path.open("wb") as raw_file:
            shutil.copyfileobj(video.file, raw_file)

        transcoded_path = Path(tmp_dir) / "transcoded.mp4"
        try:
            transcode_browser_mp4(raw_path, transcoded_path, timeout_seconds=600)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Could not process the recorded clip: {type(exc).__name__}.",
            ) from exc

        try:
            with transcoded_path.open("rb") as transcoded_file:
                record = prepare_video(transcoded_file, "quick_test.mp4")
        except VideoAnalysisServiceError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail=exc.message,
            ) from exc

    logger.info("Prepared quick test video %s", record.analysis_id)
    return record
