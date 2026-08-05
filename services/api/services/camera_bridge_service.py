"""Read-only adapters from stored OpenCV cameras to the renderer contract."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Sequence

from ..schemas.camera_bridge import (
    CameraBridgeDistortion,
    CameraBridgeInput,
    CameraBridgeResponse,
    CameraBridgeSetupFrame,
    CameraBridgeSource,
)
from ..schemas.real_pitch_registration import (
    CameraPoseCandidate,
    RealProjectedPitchGeometry,
)
from ..schemas.scene_calibration import AcceptedSceneCalibrationSnapshot
from ..schemas.virtual_pitch import ProjectedPitchGeometry, VirtualCamera
from ..schemas.wicket_observation import SetupFrameCandidate
from .real_pitch_registration_service import load_real_pitch_registration
from .scene_calibration_service import (
    load_active_accepted_scene_calibration,
    load_scene_calibration,
)
from .wicket_box_calibration_service import (
    load_active_accepted_wicket_box_calibration,
)
from .video_analysis_service import (
    ANALYSIS_ID_PATTERN,
    VIDEO_ANALYSIS_ROOT,
    VideoAnalysisServiceError,
    load_video_analysis,
)
from .frame_timing import resolve_frame_timestamp
from .virtual_pitch_service import build_synthetic_preview


DISTORTION_ZERO_TOLERANCE = 1e-12
DISTORTION_COEFFICIENT_ORDER = "OpenCV: k1,k2,p1,p2[,k3,k4,k5,k6,s1,s2,s3,s4,tauX,tauY]"
DEFAULT_NEAR_M = 0.01
DEFAULT_FAR_M = 250.0


def _distortion(
    coefficients: Sequence[float],
    *,
    frame_preundistorted: bool = False,
) -> CameraBridgeDistortion:
    values = [float(value) for value in coefficients]
    magnitude = max((abs(value) for value in values), default=0.0)
    if magnitude <= DISTORTION_ZERO_TOLERANCE:
        mode = "ZERO_DISTORTION"
        warning = None
    elif frame_preundistorted:
        mode = "PREUNDISTORTED_FRAME"
        warning = None
    else:
        mode = "NONZERO_DISTORTION_UNSUPPORTED"
        warning = (
            "Camera bridge requires a zero-distortion camera model or a "
            "pre-undistorted background frame."
        )
    return CameraBridgeDistortion(
        mode=mode,
        coefficients=values,
        coefficient_order=DISTORTION_COEFFICIENT_ORDER,
        maximum_absolute_coefficient=magnitude,
        frame_preundistorted=frame_preundistorted,
        exact_pinhole_rendering_supported=(
            mode in {"ZERO_DISTORTION", "PREUNDISTORTED_FRAME"}
        ),
        warning=warning,
    )


def _validated_setup_frame(
    analysis_id: str,
    frame: SetupFrameCandidate,
    image_url: str | None,
) -> CameraBridgeSetupFrame:
    if not image_url:
        raise VideoAnalysisServiceError(
            "Selected camera setup frame URL is unavailable.", status_code=404
        )
    prefix = f"/static/video-analysis/{analysis_id}/"
    if not image_url.startswith(prefix):
        raise VideoAnalysisServiceError(
            "Selected camera setup frame reference is invalid.", status_code=500
        )
    relative = PurePosixPath(image_url[len(prefix) :])
    if relative.is_absolute() or ".." in relative.parts:
        raise VideoAnalysisServiceError(
            "Selected camera setup frame reference is invalid.", status_code=500
        )
    analysis_root = (VIDEO_ANALYSIS_ROOT / analysis_id).resolve()
    image_path = (analysis_root / Path(*relative.parts)).resolve()
    try:
        image_path.relative_to(analysis_root)
    except ValueError as exc:
        raise VideoAnalysisServiceError(
            "Selected camera setup frame reference is invalid.", status_code=500
        ) from exc
    if not image_path.is_file():
        raise VideoAnalysisServiceError(
            "Selected camera setup frame is unavailable.", status_code=404
        )
    try:
        with image_path.open("rb") as image_file:
            signature = image_file.read(8)
    except OSError as exc:
        raise VideoAnalysisServiceError(
            "Selected camera setup frame is unavailable.", status_code=404
        ) from exc
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"} and signature.startswith(b"\xff\xd8\xff"):
        media_type = "image/jpeg"
    elif suffix == ".png" and signature == b"\x89PNG\r\n\x1a\n":
        media_type = "image/png"
    else:
        raise VideoAnalysisServiceError(
            "Selected camera setup frame is not a valid JPEG or PNG image.",
            status_code=422,
        )
    return CameraBridgeSetupFrame(
        frame_index=frame.frame_index,
        timestamp_seconds=frame.timestamp_seconds,
        image_width=frame.image_width,
        image_height=frame.image_height,
        image_url=image_url,
        media_type=media_type,
    )


def _matrix_values(camera_matrix: Sequence[Sequence[float]]):
    matrix = [[float(value) for value in row] for row in camera_matrix]
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise VideoAnalysisServiceError(
            "Selected camera matrix is invalid.", status_code=500
        )
    return matrix, matrix[0][0], matrix[1][1], matrix[0][2], matrix[1][2], matrix[0][1]


def _candidate_camera_matrix(candidate: CameraPoseCandidate) -> list[list[float]]:
    intrinsics = candidate.intrinsics
    return [
        [
            intrinsics.focal_length_x_px,
            0.0,
            intrinsics.principal_point_x_px,
        ],
        [0.0, intrinsics.focal_length_y_px, intrinsics.principal_point_y_px],
        [0.0, 0.0, 1.0],
    ]


def _candidate_bridge(
    *,
    source: CameraBridgeSource,
    source_version: str,
    analysis_id: str,
    candidate: CameraPoseCandidate,
    classification: str,
    accepted: bool,
    setup_frame: CameraBridgeSetupFrame,
    warnings: Sequence[str],
) -> CameraBridgeInput:
    if (
        candidate.rotation_vector is None
        or candidate.rotation_matrix is None
        or candidate.translation_vector is None
        or candidate.camera_world_position is None
    ):
        raise VideoAnalysisServiceError(
            "Selected camera candidate is incomplete.", status_code=422
        )
    matrix, fx, fy, cx, cy, skew = _matrix_values(_candidate_camera_matrix(candidate))
    distortion = _distortion(candidate.intrinsics.distortion_coefficients)
    camera_warnings = [*warnings]
    if distortion.warning:
        camera_warnings.append(distortion.warning)
    return CameraBridgeInput(
        source=source,
        source_version=source_version,
        analysis_id=analysis_id,
        candidate_id=candidate.candidate_id,
        accepted=accepted,
        classification=classification,
        image_width=setup_frame.image_width,
        image_height=setup_frame.image_height,
        camera_matrix=matrix,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        skew=skew,
        distortion=distortion,
        rotation_vector=[float(value) for value in candidate.rotation_vector],
        rotation_matrix=[
            [float(value) for value in row] for row in candidate.rotation_matrix
        ],
        translation_vector=[float(value) for value in candidate.translation_vector],
        camera_world_position=[
            float(value) for value in candidate.camera_world_position
        ],
        near_m=DEFAULT_NEAR_M,
        far_m=DEFAULT_FAR_M,
        setup_frame=setup_frame,
        warnings=camera_warnings,
    )


def _accepted_wicket_box_bridge(
    snapshot: dict[str, object],
    setup_frame: CameraBridgeSetupFrame,
) -> CameraBridgeInput:
    matrix, fx, fy, cx, cy, skew = _matrix_values(snapshot["camera_matrix"])
    distortion_coefficients = snapshot.get("distortion_coefficients") or [0.0] * 5
    distortion = _distortion(distortion_coefficients)
    warnings = [distortion.warning] if distortion.warning else []
    return CameraBridgeInput(
        source="ACCEPTED_WICKET_BOX_CALIBRATION",
        source_version="wicket_box_calibration_accepted_v1",
        analysis_id=str(snapshot["analysis_id"]),
        candidate_id="wicket-box-accepted",
        accepted=True,
        classification="METRIC_3D_READY",
        image_width=setup_frame.image_width,
        image_height=setup_frame.image_height,
        camera_matrix=matrix,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        skew=skew,
        distortion=distortion,
        rotation_vector=[float(value) for value in snapshot["rotation_vector"]],
        rotation_matrix=[
            [float(value) for value in row] for row in snapshot["rotation_matrix"]
        ],
        translation_vector=[
            float(value) for value in snapshot["translation_vector"]
        ],
        camera_world_position=[
            float(value) for value in snapshot["camera_world_position"]
        ],
        near_m=DEFAULT_NEAR_M,
        far_m=DEFAULT_FAR_M,
        setup_frame=setup_frame,
        warnings=warnings,
    )


def _wicket_box_setup_frame(
    analysis_id: str,
    snapshot: dict[str, object],
) -> tuple[SetupFrameCandidate, str, list[str]]:
    frame_index = int(snapshot["calibration_frame_index"])
    width = int(snapshot["source_image_width"])
    height = int(snapshot["source_image_height"])
    analysis = load_video_analysis(analysis_id)
    container_fps = getattr(analysis, "fps", None)
    timestamp_seconds, timestamp_method = resolve_frame_timestamp(
        frame_index,
        fps=float(container_fps) if container_fps is not None else None,
    )
    timing_warnings: list[str] = []
    if timestamp_method == "NOMINAL_FPS_FALLBACK":
        timing_warnings.append(f"timestamp_method:{timestamp_method}")
    setup = SetupFrameCandidate(
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        image_width=width,
        image_height=height,
        score=0.9,
        sharpness=100.0,
        brightness=120.0,
        wicket_detection_count=2,
        mean_detector_confidence=0.8,
        detection_stability=0.9,
        obstruction_score=0.0,
        selected=True,
    )
    analysis_root = (VIDEO_ANALYSIS_ROOT / analysis_id).resolve()
    relative_candidates = [
        "calibration/setup.jpg",
        f"calibration/wicket_observation_v1/setup_frame_{frame_index:06d}.jpg",
    ]
    calibration_dir = analysis_root / "calibration"
    if calibration_dir.is_dir():
        relative_candidates.extend(
            path.relative_to(analysis_root).as_posix()
            for path in sorted(calibration_dir.rglob(f"setup_frame_{frame_index:06d}.jpg"))
        )
    for relative in relative_candidates:
        image_url = f"/static/video-analysis/{analysis_id}/{relative}"
        try:
            _validated_setup_frame(analysis_id, setup, image_url)
            return setup, image_url, timing_warnings
        except VideoAnalysisServiceError:
            continue
    scene = load_scene_calibration(analysis_id)
    if scene.setup_frame is not None and scene.setup_frame_image_url:
        _validated_setup_frame(
            analysis_id, scene.setup_frame, scene.setup_frame_image_url
        )
        return scene.setup_frame, scene.setup_frame_image_url, timing_warnings
    raise VideoAnalysisServiceError(
        "Accepted wicket-box setup frame is unavailable.", status_code=404
    )


def _accepted_bridge(
    snapshot: AcceptedSceneCalibrationSnapshot,
    setup_frame: CameraBridgeSetupFrame,
) -> CameraBridgeInput:
    matrix, fx, fy, cx, cy, skew = _matrix_values(snapshot.camera_matrix)
    distortion = _distortion(snapshot.distortion_coefficients)
    warnings = [distortion.warning] if distortion.warning else []
    return CameraBridgeInput(
        source="ACCEPTED_SCENE_CALIBRATION",
        source_version=f"scene_calibration_v1_revision_{snapshot.revision}",
        analysis_id=snapshot.analysis_id,
        candidate_id=snapshot.candidate_id,
        accepted=True,
        classification=snapshot.calibration_level,
        image_width=setup_frame.image_width,
        image_height=setup_frame.image_height,
        camera_matrix=matrix,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        skew=skew,
        distortion=distortion,
        rotation_vector=[float(value) for value in snapshot.rotation_vector],
        rotation_matrix=[
            [float(value) for value in row] for row in snapshot.rotation_matrix
        ],
        translation_vector=[float(value) for value in snapshot.translation_vector],
        camera_world_position=[
            float(value) for value in snapshot.camera_world_position
        ],
        near_m=DEFAULT_NEAR_M,
        far_m=DEFAULT_FAR_M,
        setup_frame=setup_frame,
        warnings=warnings,
    )


def build_synthetic_camera_bridge(camera_name: str) -> CameraBridgeResponse:
    preview = build_synthetic_preview(camera_name)
    camera: VirtualCamera = preview.projection.source_camera
    matrix, fx, fy, cx, cy, skew = _matrix_values(camera.camera_matrix)
    distortion = _distortion(camera.distortion_coefficients)
    bridge = CameraBridgeInput(
        source="SYNTHETIC_VIRTUAL_PITCH",
        source_version="virtual_pitch_v1",
        candidate_id=camera.name,
        accepted=False,
        classification="SYNTHETIC_EXACT",
        image_width=camera.image_width,
        image_height=camera.image_height,
        camera_matrix=matrix,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        skew=skew,
        distortion=distortion,
        rotation_vector=camera.rotation_vector,
        rotation_matrix=camera.rotation_matrix,
        translation_vector=camera.translation_vector,
        camera_world_position=camera.camera_position_world,
        near_m=camera.near_m,
        far_m=camera.far_m,
    )
    return CameraBridgeResponse(
        status="AVAILABLE",
        camera=bridge,
        projected_pitch_geometry=preview.projection,
        message="Synthetic OpenCV camera normalized for bridge validation.",
    )


def _matching_scene_projection(
    candidate_id: str,
    scene_candidate_id: str | None,
    projection: RealProjectedPitchGeometry | None,
) -> RealProjectedPitchGeometry | None:
    return projection if candidate_id == scene_candidate_id else None


def load_analysis_camera_bridge(analysis_id: str) -> CameraBridgeResponse:
    if not ANALYSIS_ID_PATTERN.fullmatch(analysis_id):
        raise VideoAnalysisServiceError("Invalid analysis ID.", status_code=404)
    load_video_analysis(analysis_id)

    wicket_snapshot = load_active_accepted_wicket_box_calibration(analysis_id)
    if wicket_snapshot is not None:
        setup, image_url, timing_warnings = _wicket_box_setup_frame(analysis_id, wicket_snapshot)
        frame = _validated_setup_frame(analysis_id, setup, image_url)
        camera = _accepted_wicket_box_bridge(wicket_snapshot, frame)
        if timing_warnings:
            camera = camera.model_copy(
                update={"warnings": [*camera.warnings, *timing_warnings]}
            )
        return CameraBridgeResponse(
            status="AVAILABLE",
            camera=camera,
            projected_pitch_geometry=None,
            message="Active accepted wicket-box calibration loaded for rendering.",
        )

    scene = load_scene_calibration(analysis_id)

    if scene.accepted_calibration is not None:
        snapshot = load_active_accepted_scene_calibration(analysis_id)
        frame = _validated_setup_frame(
            analysis_id, snapshot.setup_frame, snapshot.setup_frame_image_url
        )
        camera = _accepted_bridge(snapshot, frame)
        projection = _matching_scene_projection(
            snapshot.candidate_id,
            scene.selected_candidate.candidate_id if scene.selected_candidate else None,
            scene.projected_pitch_geometry,
        )
        return CameraBridgeResponse(
            status="AVAILABLE",
            camera=camera,
            projected_pitch_geometry=projection,
            message="Active accepted scene calibration loaded for rendering.",
        )

    refined = scene.refined_registration_summary
    candidate = scene.selected_candidate
    if (
        refined is not None
        and candidate is not None
        and refined.selected_candidate_id == candidate.candidate_id
        and scene.setup_frame is not None
    ):
        frame = _validated_setup_frame(
            analysis_id, scene.setup_frame, scene.setup_frame_image_url
        )
        camera = _candidate_bridge(
            source="REFINED_SCENE_CALIBRATION_CANDIDATE",
            source_version="scene_calibration_v1",
            analysis_id=analysis_id,
            candidate=candidate,
            classification=refined.status,
            accepted=False,
            setup_frame=frame,
            warnings=[
                "Unaccepted camera candidate; visual validation only.",
                *scene.warnings,
            ],
        )
        return CameraBridgeResponse(
            status="AVAILABLE",
            camera=camera,
            projected_pitch_geometry=scene.projected_pitch_geometry,
            warnings=["Metric analytics remain locked."],
            message="Selected refined scene-calibration candidate loaded.",
        )

    try:
        registration = load_real_pitch_registration(analysis_id)
    except VideoAnalysisServiceError as exc:
        if exc.status_code != 404:
            raise
        return CameraBridgeResponse(
            status="UNAVAILABLE",
            warnings=["No selected camera candidate is available."],
            message="Run camera registration before using calibrated rendering.",
        )
    candidate = registration.selected_candidate
    if candidate is None or registration.setup_frame is None:
        return CameraBridgeResponse(
            status="UNAVAILABLE",
            warnings=["No selected camera candidate is available."],
            message="Run camera registration before using calibrated rendering.",
        )
    frame = _validated_setup_frame(
        analysis_id,
        registration.setup_frame,
        registration.diagnostics.setup_frame_image_url,
    )
    camera = _candidate_bridge(
        source="REAL_PITCH_REGISTRATION_CANDIDATE",
        source_version="real_pitch_registration_v1",
        analysis_id=analysis_id,
        candidate=candidate,
        classification=registration.status,
        accepted=False,
        setup_frame=frame,
        warnings=[
            "Unaccepted camera candidate; visual validation only.",
            *registration.warnings,
        ],
    )
    return CameraBridgeResponse(
        status="AVAILABLE",
        camera=camera,
        projected_pitch_geometry=registration.projected_pitch_geometry,
        warnings=["Metric analytics remain locked."],
        message="Selected real pitch-registration candidate loaded.",
    )
