from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from services.api.main import app
from services.api.schemas.real_pitch_registration import (
    CameraIntrinsicsCandidate,
    CameraPoseCandidate,
)
from services.api.schemas.wicket_observation import SetupFrameCandidate
from services.api.services import camera_bridge_service as bridge
from services.api.services.video_analysis_service import VideoAnalysisServiceError


ANALYSIS_ID = "analysis_20260728_120858_762989"


def _frame() -> SetupFrameCandidate:
    return SetupFrameCandidate(
        frame_index=10,
        timestamp_seconds=0.4,
        image_width=720,
        image_height=1280,
        score=0.9,
        sharpness=100,
        brightness=120,
        wicket_detection_count=2,
        mean_detector_confidence=0.8,
        detection_stability=0.9,
        obstruction_score=0.1,
        selected=True,
    )


def _candidate(candidate_id: str = "candidate-a") -> CameraPoseCandidate:
    intrinsics = CameraIntrinsicsCandidate.model_construct(
        candidate_id="intrinsics-a",
        focal_length_x_px=900.0,
        focal_length_y_px=880.0,
        principal_point_x_px=351.0,
        principal_point_y_px=626.0,
        distortion_coefficients=[0.0] * 5,
    )
    return CameraPoseCandidate.model_construct(
        candidate_id=candidate_id,
        intrinsics=intrinsics,
        rotation_vector=[0.1, 0.2, 0.3],
        rotation_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation_vector=[0.0, 1.0, 10.0],
        camera_world_position=[0.0, -1.0, 2.0],
    )


@pytest.fixture
def bridge_analysis_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "video_analysis"
    monkeypatch.setattr(bridge, "VIDEO_ANALYSIS_ROOT", root)
    monkeypatch.setattr(bridge, "load_video_analysis", lambda analysis_id: object())
    return root


def _write_setup_frame(root: Path, relative: str = "calibration/setup.jpg") -> str:
    path = root / ANALYSIS_ID / Path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8\xff\xe0test")
    return f"/static/video-analysis/{ANALYSIS_ID}/{relative}"


def test_synthetic_camera_bridge_api_normalizes_existing_camera() -> None:
    response = TestClient(app).get(
        "/video-analysis/virtual-pitch/camera-bridge",
        params={"camera_name": "centred_bowler_end"},
    )
    assert response.status_code == 200
    payload = response.json()
    camera = payload["camera"]
    assert payload["status"] == "AVAILABLE"
    assert payload["metrics_unlocked"] is False
    assert camera["source"] == "SYNTHETIC_VIRTUAL_PITCH"
    assert camera["distortion"]["mode"] == "ZERO_DISTORTION"
    assert camera["fx"] == camera["camera_matrix"][0][0]
    assert camera["extrinsic_convention"] == "X_CAMERA = R * X_CRICVISION_WORLD + T"


def test_active_accepted_wicket_box_has_first_precedence(
    bridge_analysis_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_url = _write_setup_frame(bridge_analysis_root)
    wicket_snapshot = {
        "schema_version": "wicket_box_calibration_accepted_v1",
        "analysis_id": ANALYSIS_ID,
        "frozen": True,
        "calibration_frame_index": 10,
        "source_image_width": 720,
        "source_image_height": 1280,
        "camera_matrix": [[900.0, 0.0, 351.0], [0.0, 880.0, 626.0], [0.0, 0.0, 1.0]],
        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "translation_vector": [0.0, 1.0, 10.0],
        "rotation_vector": [0.0, 0.0, 0.0],
        "camera_world_position": [0.0, -1.0, -10.0],
        "distortion_coefficients": [0.0] * 5,
    }
    monkeypatch.setattr(
        bridge,
        "load_active_accepted_wicket_box_calibration",
        lambda analysis_id: wicket_snapshot,
    )
    monkeypatch.setattr(
        bridge,
        "load_scene_calibration",
        lambda analysis_id: pytest.fail("scene calibration must not run"),
    )

    result = bridge.load_analysis_camera_bridge(ANALYSIS_ID)

    assert result.camera is not None
    assert result.camera.source == "ACCEPTED_WICKET_BOX_CALIBRATION"
    assert result.camera.accepted is True
    assert result.camera.setup_frame is not None
    assert result.camera.setup_frame.image_url == image_url


def test_active_accepted_calibration_has_first_precedence(
    bridge_analysis_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_url = _write_setup_frame(bridge_analysis_root)
    scene = SimpleNamespace(
        accepted_calibration=SimpleNamespace(),
        selected_candidate=None,
        projected_pitch_geometry=None,
    )
    snapshot = SimpleNamespace(
        revision=2,
        analysis_id=ANALYSIS_ID,
        candidate_id="accepted-a",
        calibration_level="GROUND_PLANE_READY",
        setup_frame=_frame(),
        setup_frame_image_url=image_url,
        camera_matrix=[[900.0, 3.0, 351.0], [0.0, 880.0, 626.0], [0.0, 0.0, 1.0]],
        distortion_coefficients=[0.0] * 5,
        rotation_vector=[0.1, 0.2, 0.3],
        rotation_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation_vector=[0.0, 1.0, 10.0],
        camera_world_position=[0.0, -1.0, 2.0],
    )
    monkeypatch.setattr(
        bridge, "load_active_accepted_wicket_box_calibration", lambda analysis_id: None
    )
    monkeypatch.setattr(bridge, "load_scene_calibration", lambda analysis_id: scene)
    monkeypatch.setattr(
        bridge, "load_active_accepted_scene_calibration", lambda analysis_id: snapshot
    )
    monkeypatch.setattr(
        bridge,
        "load_real_pitch_registration",
        lambda analysis_id: pytest.fail("real registration fallback must not run"),
    )

    result = bridge.load_analysis_camera_bridge(ANALYSIS_ID)

    assert result.camera is not None
    assert result.camera.source == "ACCEPTED_SCENE_CALIBRATION"
    assert result.camera.accepted is True
    assert result.camera.skew == 3.0


def test_refined_scene_candidate_is_unaccepted_and_precedes_registration(
    bridge_analysis_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_url = _write_setup_frame(bridge_analysis_root)
    candidate = _candidate("refined-a")
    scene = SimpleNamespace(
        accepted_calibration=None,
        refined_registration_summary=SimpleNamespace(
            selected_candidate_id="refined-a",
            status="GROUND_PLANE_CANDIDATE",
        ),
        selected_candidate=candidate,
        setup_frame=_frame(),
        setup_frame_image_url=image_url,
        projected_pitch_geometry=None,
        warnings=[],
    )
    monkeypatch.setattr(
        bridge, "load_active_accepted_wicket_box_calibration", lambda analysis_id: None
    )
    monkeypatch.setattr(bridge, "load_scene_calibration", lambda analysis_id: scene)
    monkeypatch.setattr(
        bridge,
        "load_real_pitch_registration",
        lambda analysis_id: pytest.fail("real registration fallback must not run"),
    )

    result = bridge.load_analysis_camera_bridge(ANALYSIS_ID)

    assert result.camera is not None
    assert result.camera.source == "REFINED_SCENE_CALIBRATION_CANDIDATE"
    assert result.camera.accepted is False
    assert result.metrics_unlocked is False


def test_selected_real_registration_is_last_camera_fallback(
    bridge_analysis_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_url = _write_setup_frame(bridge_analysis_root)
    scene = SimpleNamespace(
        accepted_calibration=None,
        refined_registration_summary=None,
        selected_candidate=None,
    )
    registration = SimpleNamespace(
        selected_candidate=_candidate("registration-a"),
        setup_frame=_frame(),
        diagnostics=SimpleNamespace(setup_frame_image_url=image_url),
        status="VISUAL_ONLY",
        warnings=[],
        projected_pitch_geometry=None,
    )
    monkeypatch.setattr(bridge, "load_scene_calibration", lambda analysis_id: scene)
    monkeypatch.setattr(
        bridge, "load_active_accepted_wicket_box_calibration", lambda analysis_id: None
    )
    monkeypatch.setattr(
        bridge, "load_real_pitch_registration", lambda analysis_id: registration
    )

    result = bridge.load_analysis_camera_bridge(ANALYSIS_ID)

    assert result.camera is not None
    assert result.camera.source == "REAL_PITCH_REGISTRATION_CANDIDATE"
    assert result.camera.accepted is False


def test_invalid_analysis_id_is_rejected_before_storage_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bridge,
        "load_video_analysis",
        lambda analysis_id: pytest.fail("invalid ID reached storage"),
    )
    with pytest.raises(VideoAnalysisServiceError) as error:
        bridge.load_analysis_camera_bridge("../../secrets")
    assert error.value.status_code == 404


def test_missing_setup_frame_is_reported(
    bridge_analysis_root: Path,
) -> None:
    with pytest.raises(VideoAnalysisServiceError) as error:
        bridge._validated_setup_frame(
            ANALYSIS_ID,
            _frame(),
            f"/static/video-analysis/{ANALYSIS_ID}/calibration/missing.jpg",
        )
    assert error.value.status_code == 404


def test_setup_frame_path_traversal_is_rejected(
    bridge_analysis_root: Path,
) -> None:
    with pytest.raises(VideoAnalysisServiceError) as error:
        bridge._validated_setup_frame(
            ANALYSIS_ID,
            _frame(),
            f"/static/video-analysis/{ANALYSIS_ID}/../other/setup.jpg",
        )
    assert error.value.status_code == 500


def test_non_image_setup_frame_is_rejected(
    bridge_analysis_root: Path,
) -> None:
    image_url = _write_setup_frame(bridge_analysis_root, "calibration/not-an-image.png")
    with pytest.raises(VideoAnalysisServiceError) as error:
        bridge._validated_setup_frame(ANALYSIS_ID, _frame(), image_url)
    assert error.value.status_code == 422


def test_distortion_policy_is_explicit() -> None:
    zero = bridge._distortion([0.0] * 5)
    unsupported = bridge._distortion([0.1, 0.0, 0.0, 0.0, 0.0])
    undistorted = bridge._distortion(
        [0.1, 0.0, 0.0, 0.0, 0.0], frame_preundistorted=True
    )
    assert zero.mode == "ZERO_DISTORTION"
    assert unsupported.mode == "NONZERO_DISTORTION_UNSUPPORTED"
    assert unsupported.exact_pinhole_rendering_supported is False
    assert undistorted.mode == "PREUNDISTORTED_FRAME"
    assert undistorted.exact_pinhole_rendering_supported is True
