"""Release Point V1 orchestration for the active Video Analysis API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from Backends.src.release_point.bowler_tracker import BowlerTracker
from Backends.src.release_point.pose_provider import _validate_clean_original_video_path
from Backends.src.release_point.release_engine import (
    ReleaseEstimator,
    candidate_score_to_dict,
)
from Backends.src.release_point.rtmpose_provider import (
    RTMPoseProvider,
    RTMPoseProviderConfig,
    RTMPoseProviderUnavailable,
    infer_bowling_arm,
)

from ..schemas.release_point import (
    ReleaseAnalysisInput,
    ReleaseCandidateScore,
    ReleaseResult,
    ReleaseResultDocument,
    VideoReleasePointResultLinks,
    VideoReleasePointResultResponse,
)
from ..schemas.video_analysis import (
    VideoBallDetectionsDocument,
    VideoBallTrackingDocument,
)
from .video_analysis_service import (
    VIDEO_ANALYSIS_ROOT,
    VideoAnalysisServiceError,
    load_video_analysis,
)
from .video_ball_detection_job_store import utc_now
from .video_release_point_job_store import video_release_point_job_store


RELEASE_POINT_FILENAME = "release_point_v1.json"
CALIBRATION_FILENAME = "calibration.json"
CALIBRATION_V2_FILENAME = "calibration_v2.json"
CAMERA_POSE_FILENAME = "camera_pose.json"
DETECTIONS_FILENAME = "detections.json"
TRACKING_FILENAME = "tracking_result.json"
TEST_ONLY_POSE_PROVIDERS = {"fake_pose", "fakeposeprovider"}
POSE_PROVIDER_ENV = "CRICVISION_RELEASE_POSE_PROVIDER"
POSE_MODEL_ENV = "CRICVISION_RELEASE_POSE_MODEL"
POSE_DEVICE_ENV = "CRICVISION_RELEASE_POSE_DEVICE"


class VideoReleasePointError(Exception):
    def __init__(self, message: str, *, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class _PoseContext:
    bowler_pose_sequence: dict[str, Any] | None
    provenance: dict[str, Any]
    quality_flags: list[str]


def load_release_analysis_input(analysis_id: str) -> ReleaseAnalysisInput:
    analysis = load_video_analysis(analysis_id)
    analysis_dir = VIDEO_ANALYSIS_ROOT / analysis_id
    raw_video_path = analysis_dir / "raw" / analysis.stored_filename
    detections_path = analysis_dir / "detections" / DETECTIONS_FILENAME
    tracking_path = analysis_dir / "tracking" / TRACKING_FILENAME
    calibration_path = analysis_dir / "calibration" / CALIBRATION_FILENAME
    calibration_v2_path = analysis_dir / "calibration" / CALIBRATION_V2_FILENAME
    camera_pose_path = analysis_dir / "calibration" / CAMERA_POSE_FILENAME

    required = {
        "raw original video": raw_video_path,
        "detections.json": detections_path,
        "tracking_result.json": tracking_path,
        "calibration.json": calibration_path,
    }
    missing = [label for label, path in required.items() if not path.is_file()]
    if missing:
        raise VideoReleasePointError(
            f"Release Point V1 input is missing: {', '.join(missing)}.",
            status_code=409,
        )
    _validate_clean_original_video_path(raw_video_path)

    return ReleaseAnalysisInput(
        analysis_id=analysis_id,
        raw_video_path=str(raw_video_path),
        fps=analysis.fps,
        frame_count=analysis.frame_count,
        width=analysis.width,
        height=analysis.height,
        detections_path=str(detections_path),
        tracking_path=str(tracking_path),
        calibration_path=str(calibration_path),
        calibration_v2_path=(
            str(calibration_v2_path) if calibration_v2_path.is_file() else None
        ),
        camera_pose_path=str(camera_pose_path) if camera_pose_path.is_file() else None,
    )


def validate_video_release_point_input(analysis_id: str) -> ReleaseAnalysisInput:
    release_input = load_release_analysis_input(analysis_id)
    _load_detection_document(release_input)
    _load_tracking_document(release_input)
    _read_json(Path(release_input.calibration_path), "calibration.json")
    return release_input


def mark_video_release_point_queued(analysis_id: str, job_id: str) -> None:
    now = utc_now()
    _update_analysis_metadata(
        analysis_id,
        release_point_status="release_point_queued",
        release_point_job_id=job_id,
        release_point_started_at=_iso(now),
        release_point_completed_at=None,
        release_point_url=None,
        updated_at=_iso(now),
    )


def run_video_release_point_job(
    analysis_id: str,
    job_id: str,
    bowler_pose_sequence: dict[str, Any] | None = None,
) -> None:
    try:
        document = _process_video_release_point(
            analysis_id,
            job_id,
            bowler_pose_sequence=bowler_pose_sequence,
        )
        links = VideoReleasePointResultLinks(
            release_json_url=_release_json_url(analysis_id)
        )
        is_ready = document.result.status == "ready"
        video_release_point_job_store.update(
            job_id,
            success=is_ready,
            status=document.result.status,
            progress=100,
            error_message=None if is_ready else document.message,
            result=links.model_dump(mode="json"),
            message=document.message,
        )
        _update_analysis_metadata(
            analysis_id,
            release_point_status=(
                "release_point_complete" if is_ready else "release_point_unresolved"
            ),
            release_point_completed_at=_iso(utc_now()),
            release_point_url=_release_json_url(analysis_id),
            updated_at=_iso(utc_now()),
        )
    except (VideoReleasePointError, VideoAnalysisServiceError) as exc:
        message = getattr(exc, "message", str(exc))
        _mark_job_failed(analysis_id, job_id, message)
    except Exception as exc:
        _mark_job_failed(
            analysis_id,
            job_id,
            f"Release Point V1 failed: {type(exc).__name__}.",
        )


def load_video_release_point_result(
    analysis_id: str,
) -> VideoReleasePointResultResponse:
    load_video_analysis(analysis_id)
    path = _release_result_path(analysis_id)
    if not path.is_file():
        raise VideoReleasePointError(
            "Release Point V1 has not completed.",
            status_code=404,
        )
    try:
        document = ReleaseResultDocument.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise VideoReleasePointError(
            "Saved Release Point V1 result is unavailable.",
            status_code=500,
        ) from exc
    if document.analysis_id != analysis_id:
        raise VideoReleasePointError(
            "Saved Release Point V1 result does not match the analysis.",
            status_code=500,
        )
    return VideoReleasePointResultResponse(
        success=document.result.status == "ready",
        status=document.result.status,
        analysis_id=analysis_id,
        release_json_url=_release_json_url(analysis_id),
        result=document.result,
        candidate_scores=document.candidate_scores,
        quality_summary=document.quality_summary,
        message=document.message,
    )


def _process_video_release_point(
    analysis_id: str,
    job_id: str,
    *,
    bowler_pose_sequence: dict[str, Any] | None,
) -> ReleaseResultDocument:
    release_input = validate_video_release_point_input(analysis_id)
    started_at = utc_now()

    _update_job(job_id, "loading_inputs", 10, "Loading Release Point V1 inputs.")
    detections = _load_detection_document(release_input)
    tracking = _load_tracking_document(release_input)
    calibration = _read_json(Path(release_input.calibration_path), "calibration.json")
    calibration_v2 = (
        _read_json(Path(release_input.calibration_v2_path), "calibration_v2.json")
        if release_input.calibration_v2_path
        else None
    )
    camera_pose = (
        _read_json(Path(release_input.camera_pose_path), "camera_pose.json")
        if release_input.camera_pose_path
        else None
    )
    pose_context = _resolve_pose_context(
        release_input,
        calibration=calibration,
        calibration_v2=calibration_v2,
        camera_pose=camera_pose,
        tracking=tracking,
        bowler_pose_sequence=bowler_pose_sequence,
    )

    _update_job(
        job_id,
        "generating_candidates",
        35,
        "Generating plausible release candidates.",
    )
    estimator = ReleaseEstimator()

    _update_job(
        job_id,
        "scoring_candidates",
        65,
        "Extracting release evidence and scoring candidates.",
    )
    estimate = estimator.estimate(
        analysis_id=analysis_id,
        fps=release_input.fps,
        detections_document=detections,
        tracking_document=tracking,
        bowler_pose_sequence=pose_context.bowler_pose_sequence,
        provenance=_provenance(
            detections,
            tracking,
            calibration,
            calibration_v2,
            camera_pose,
            pose_context,
        ),
    )

    _update_job(job_id, "saving_results", 88, "Saving Release Point V1 result.")
    completed_at = utc_now()
    document = ReleaseResultDocument(
        schema_version=estimate.result.get("schema_version", "1.0"),
        analysis_id=analysis_id,
        created_at=started_at,
        completed_at=completed_at,
        result=ReleaseResult.model_validate(estimate.result),
        candidate_scores=[
            ReleaseCandidateScore.model_validate(candidate_score_to_dict(score))
            for score in estimate.candidate_scores
        ],
        quality_summary=estimate.quality_summary,
        message=estimate.message,
    )
    _write_json(
        _release_result_path(analysis_id),
        document.model_dump(mode="json"),
    )
    return document


def _load_detection_document(release_input: ReleaseAnalysisInput) -> dict[str, Any]:
    data = _read_json(Path(release_input.detections_path), "detections.json")
    try:
        document = VideoBallDetectionsDocument.model_validate(data)
    except ValueError as exc:
        raise VideoReleasePointError(
            "detections.json is malformed.",
            status_code=400,
        ) from exc
    if document.analysis_id != release_input.analysis_id:
        raise VideoReleasePointError(
            "detections.json does not match this analysis.",
            status_code=400,
        )
    return document.model_dump(mode="json")


def _load_tracking_document(release_input: ReleaseAnalysisInput) -> dict[str, Any]:
    data = _read_json(Path(release_input.tracking_path), "tracking_result.json")
    try:
        document = VideoBallTrackingDocument.model_validate(data)
    except ValueError as exc:
        raise VideoReleasePointError(
            "tracking_result.json is malformed.",
            status_code=400,
        ) from exc
    if document.analysis_id != release_input.analysis_id:
        raise VideoReleasePointError(
            "tracking_result.json does not match this analysis.",
            status_code=400,
        )
    if document.status != "ready" or len(document.primary_track) == 0:
        raise VideoReleasePointError(
            "A ready primary ball track is required before Release Point V1.",
            status_code=409,
        )
    return document.model_dump(mode="json")


def _provenance(
    detections: dict[str, Any],
    tracking: dict[str, Any],
    calibration: dict[str, Any],
    calibration_v2: dict[str, Any] | None,
    camera_pose: dict[str, Any] | None,
    pose_context: _PoseContext,
) -> dict[str, Any]:
    detector = detections.get("detector") or {}
    tracking_settings = tracking.get("settings") or {}
    bowler_pose_sequence = pose_context.bowler_pose_sequence or {}
    pose_provider = pose_context.provenance
    calibration_sources = ["legacy_2d"]
    if calibration_v2 is not None:
        calibration_sources.append("calibration_v2")
    if camera_pose is not None:
        calibration_sources.append("camera_pose")
    return {
        "pose_provider": pose_provider.get("name"),
        "pose_model": pose_provider.get("model"),
        "pose_schema": pose_provider.get("schema"),
        "pose_evidence_real": bool(pose_provider.get("evidence_real")),
        "pose_status": pose_provider.get("status"),
        "bowling_arm": pose_provider.get("bowling_arm"),
        "bowler_id": bowler_pose_sequence.get("bowler_id"),
        "ball_detector_model_key": (
            detections.get("ball_detector_model_key")
            or detector.get("key")
        ),
        "ball_detector_model_name": (
            detections.get("ball_detector_model_name")
            or detector.get("name")
        ),
        "tracking_version": tracking_settings.get("tracker_version"),
        "calibration_sources": calibration_sources,
        "calibration_mode": calibration.get("mode"),
        "quality_flags": list(pose_context.quality_flags),
    }


def _resolve_pose_context(
    release_input: ReleaseAnalysisInput,
    *,
    calibration: dict[str, Any],
    calibration_v2: dict[str, Any] | None,
    camera_pose: dict[str, Any] | None,
    tracking: dict[str, Any],
    bowler_pose_sequence: dict[str, Any] | None,
) -> _PoseContext:
    _ = calibration, calibration_v2, camera_pose, tracking
    _validate_clean_original_video_path(release_input.raw_video_path)

    if bowler_pose_sequence is not None:
        provider = dict(bowler_pose_sequence.get("provider") or {})
        provider_name = str(provider.get("name") or "").strip().lower()
        if provider_name in TEST_ONLY_POSE_PROVIDERS:
            raise VideoReleasePointError(
                "FakePoseProvider is test-only and cannot be used in production Release Point analysis.",
                status_code=400,
            )
        if not provider_name:
            raise VideoReleasePointError(
                "Bowler pose sequence is missing provider provenance.",
                status_code=400,
            )
        return _PoseContext(
            bowler_pose_sequence=bowler_pose_sequence,
            provenance={
                "name": provider.get("name"),
                "model": provider.get("model"),
                "schema": provider.get("schema"),
                "status": "ran",
                "evidence_real": True,
                "bowling_arm": bowler_pose_sequence.get("bowling_arm"),
            },
            quality_flags=list(bowler_pose_sequence.get("quality_flags") or []),
        )

    configured_provider = os.getenv(POSE_PROVIDER_ENV, "").strip()
    if not configured_provider:
        return _PoseContext(
            bowler_pose_sequence=None,
            provenance={
                "name": None,
                "model": None,
                "schema": None,
                "status": "not_run",
                "evidence_real": False,
            },
            quality_flags=["pose_not_run"],
        )

    if configured_provider.lower() in TEST_ONLY_POSE_PROVIDERS:
        raise VideoReleasePointError(
            "FakePoseProvider is test-only and cannot be configured for production Release Point analysis.",
            status_code=400,
        )

    if configured_provider.lower() not in {"rtmpose", "rtmpose_mmpose"}:
        return _PoseContext(
            bowler_pose_sequence=None,
            provenance={
                "name": None,
                "model": None,
                "schema": None,
                "status": "unavailable",
                "configured_provider": configured_provider,
                "evidence_real": False,
            },
            quality_flags=["pose_provider_unknown", "pose_not_run"],
        )

    try:
        return _run_rtmpose_provider(
            release_input,
            calibration=calibration,
            calibration_v2=calibration_v2,
            tracking=tracking,
        )
    except RTMPoseProviderUnavailable as exc:
        return _PoseContext(
            bowler_pose_sequence=None,
            provenance={
                "name": None,
                "model": None,
                "schema": None,
                "status": "unavailable",
                "configured_provider": configured_provider,
                "failure": str(exc),
                "evidence_real": False,
            },
            quality_flags=["pose_provider_unavailable", "pose_not_run"],
        )


def _run_rtmpose_provider(
    release_input: ReleaseAnalysisInput,
    *,
    calibration: dict[str, Any],
    calibration_v2: dict[str, Any] | None,
    tracking: dict[str, Any],
) -> _PoseContext:
    config = RTMPoseProviderConfig(
        pose2d=os.getenv(POSE_MODEL_ENV, "human").strip() or "human",
        model_name=_model_name_from_env(),
        device=os.getenv(POSE_DEVICE_ENV, "").strip() or None,
    )
    provider = RTMPoseProvider(config)
    frame_window = _pose_frame_window(tracking, release_input.frame_count)
    if not frame_window:
        raise RTMPoseProviderUnavailable(
            "No primary-track frame window is available for pose inference."
        )
    pose_sequence = provider.estimate_sequence(
        release_input.raw_video_path,
        frame_window,
        fps=release_input.fps,
    )
    bowler = BowlerTracker().track(
        pose_sequence,
        scene_calibration=calibration,
        pitch_context=calibration_v2,
        ball_track=tracking.get("primary_track", []),
    )
    bowler_dict = bowler.to_dict()
    bowler_dict["provider"] = provider.provider_info.to_dict()
    arm = infer_bowling_arm(bowler.poses_by_frame)
    bowler_dict["bowling_arm"] = arm

    quality_flags = list(bowler.quality_flags)
    quality_flags.extend(arm.get("quality_flags", []))
    if not bowler.poses_by_frame:
        quality_flags.append("pose_insufficient")
        return _PoseContext(
            bowler_pose_sequence=None,
            provenance={
                **provider.provider_info.to_dict(),
                "status": "ran_insufficient",
                "evidence_real": False,
            },
            quality_flags=_unique_flags(quality_flags),
        )

    return _PoseContext(
        bowler_pose_sequence=bowler_dict,
        provenance={
            **provider.provider_info.to_dict(),
            "status": "ran",
            "evidence_real": True,
            "frame_window": {
                "start": min(frame_window),
                "end": max(frame_window),
                "count": len(frame_window),
            },
            "bowling_arm": arm,
        },
        quality_flags=_unique_flags(quality_flags),
    )


def _pose_frame_window(
    tracking: dict[str, Any],
    frame_count: int,
) -> list[int]:
    primary_track = tracking.get("primary_track", []) or []
    if not primary_track:
        return []
    start_frame = int(primary_track[0].get("frame_index", 0))
    start = max(0, start_frame - 12)
    end = min(max(0, frame_count - 1), start_frame + 10)
    return list(range(start, end + 1))


def _model_name_from_env() -> str:
    alias = os.getenv(POSE_MODEL_ENV, "human").strip() or "human"
    names = {
        "human": "rtmpose-m_8xb256-420e_body8-256x192",
        "body26": "rtmpose-m_8xb512-700e_body8-halpe26-256x192",
        "wholebody": "rtmpose-m_8xb64-270e_coco-wholebody-256x192",
    }
    return names.get(alias, alias)


def _unique_flags(flags: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw_flag in flags:
        flag = str(raw_flag)
        if flag and flag not in seen:
            result.append(flag)
            seen.add(flag)
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VideoReleasePointError(
            f"{label} is missing.",
            status_code=409,
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise VideoReleasePointError(
            f"{label} is unavailable or malformed.",
            status_code=400,
        ) from exc


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(path)
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        raise VideoReleasePointError(
            "release_point_v1.json could not be saved.",
            status_code=500,
        ) from exc


def _release_result_path(analysis_id: str) -> Path:
    return VIDEO_ANALYSIS_ROOT / analysis_id / "reports" / RELEASE_POINT_FILENAME


def _release_json_url(analysis_id: str) -> str:
    return f"/static/video-analysis/{analysis_id}/reports/{RELEASE_POINT_FILENAME}"


def _update_analysis_metadata(analysis_id: str, **updates: Any) -> None:
    metadata_path = (
        VIDEO_ANALYSIS_ROOT
        / analysis_id
        / "reports"
        / "analysis_metadata.json"
    )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(updates)
        temporary_path = metadata_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(metadata_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise VideoReleasePointError(
            "Analysis metadata could not be updated.",
            status_code=500,
        ) from exc


def _update_job(job_id: str, status: str, progress: int, message: str) -> None:
    video_release_point_job_store.update(
        job_id,
        status=status,
        progress=progress,
        message=message,
    )


def _mark_job_failed(analysis_id: str, job_id: str, message: str) -> None:
    for temporary_path in (VIDEO_ANALYSIS_ROOT / analysis_id / "reports").glob("*.tmp"):
        temporary_path.unlink(missing_ok=True)
    video_release_point_job_store.update(
        job_id,
        success=False,
        status="failed",
        error_message=message,
        message=message,
    )
    try:
        _update_analysis_metadata(
            analysis_id,
            release_point_status="release_point_failed",
            release_point_completed_at=_iso(utc_now()),
            updated_at=_iso(utc_now()),
        )
    except VideoReleasePointError:
        pass


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
