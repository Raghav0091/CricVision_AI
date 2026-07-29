from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from services.api.main import app
from services.api.schemas.real_pitch_registration import (
    RealPitchRegistrationResult,
    RegistrationDiagnostics,
)
from services.api.schemas.scene_calibration import (
    SceneCalibrationActionRequest,
    SceneCalibrationAnchor,
    SceneCalibrationAnchorInput,
    SceneCalibrationAnchorUpdateRequest,
    SceneCalibrationRefineRequest,
    SceneCalibrationResult,
    SceneCalibrationValidation,
)
from services.api.schemas.wicket_observation import PixelPoint
from services.api.services.delivery_physics_service import (
    load_physics_calibration,
)
from services.api.services.scene_calibration_service import (
    METRIC_3D_METRICS,
    WICKET_ANCHOR_IDS,
    accept_scene_calibration,
    evaluate_calibration_candidate,
    initialise_scene_anchors,
    load_scene_calibration,
    refine_scene_calibration,
    run_scene_calibration,
    transition_scene_calibration,
    update_scene_calibration_anchors,
    use_visual_overlay_only,
    validate_scene_anchors,
)
from services.api.services.video_analysis_service import (
    VideoAnalysisServiceError,
)
from tests.test_real_pitch_registration import (
    _setup_frame,
    _solve,
    _synthetic_observation,
)


ANALYSIS_ID = "analysis_scene_calibration_test"


def _registration(observation=None) -> RealPitchRegistrationResult:
    observation = observation or _synthetic_observation(noise_px=0.2)
    candidate, correspondences = _solve(observation)
    return RealPitchRegistrationResult(
        analysis_id=ANALYSIS_ID,
        status=candidate.classification,
        attempted=True,
        setup_frame=observation.setup_frame,
        supporting_frames=observation.supporting_frames,
        wicket_observation_source="/static/wicket.json",
        correspondences=correspondences,
        candidates=[candidate],
        selected_candidate=candidate,
        ambiguity_score=0.1,
        diagnostics=RegistrationDiagnostics(),
        message="candidate",
    )


def _anchor(
    semantic_id: str,
    x: float,
    y: float,
    *,
    kind: str = "wicket",
    source: str = "automatic",
    original: PixelPoint | None = None,
    used_for_refinement: bool = True,
) -> SceneCalibrationAnchor:
    point = PixelPoint(x=x, y=y)
    return SceneCalibrationAnchor(
        semantic_id=semantic_id,
        kind=kind,
        wicket_role="near" if semantic_id.startswith("near") else "far",
        video_point=point,
        source=source,
        original_automatic_point=original if original is not None else point,
        confidence=0.9,
        uncertainty_px=2,
        adjustment_distance_px=0,
        frame_index=4,
        valid=True,
        used_for_refinement=used_for_refinement,
        used_for_validation=True,
    )


def _valid_anchors() -> list[SceneCalibrationAnchor]:
    return [
        _anchor("near_left_base", 200, 600),
        _anchor("near_right_base", 320, 600),
        _anchor("near_top_center", 260, 450),
        _anchor("far_left_base", 850, 300),
        _anchor("far_right_base", 900, 300),
        _anchor("far_top_center", 875, 235),
    ]


@pytest.fixture
def isolated_scene(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    root = tmp_path / "video_analysis"
    monkeypatch.setattr(
        "services.api.services.scene_calibration_service.VIDEO_ANALYSIS_ROOT",
        root,
    )
    monkeypatch.setattr(
        "services.api.services.scene_calibration_service.load_video_analysis",
        lambda analysis_id: SimpleNamespace(),
    )
    return root


def test_state_machine_rejects_invalid_transition() -> None:
    result = SceneCalibrationResult(
        analysis_id=ANALYSIS_ID,
        stage="NOT_STARTED",
        updated_at=datetime.now(timezone.utc),
        message="new",
    )
    detecting = transition_scene_calibration(
        result, "DETECTING_WICKETS", "detecting"
    )
    assert detecting.stage_history[-1].stage == "DETECTING_WICKETS"
    with pytest.raises(VideoAnalysisServiceError, match="Invalid"):
        transition_scene_calibration(
            result, "METRIC_3D_READY", "not allowed"
        )


def test_orchestration_runs_observation_once_then_registration_and_persists(
    isolated_scene: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = _synthetic_observation(noise_px=0.2)
    observation.diagnostics.setup_frame_image_url = "/static/setup.png"
    observation.diagnostics.raw_detection_overlay_url = "/static/raw.png"
    registration = _registration(observation)
    calls: list[str] = []
    monkeypatch.setattr(
        "services.api.services.scene_calibration_service.run_wicket_observation",
        lambda analysis_id: calls.append("observation") or observation,
    )
    monkeypatch.setattr(
        "services.api.services.scene_calibration_service.run_real_pitch_registration",
        lambda analysis_id: calls.append("registration") or registration,
    )

    result = run_scene_calibration(ANALYSIS_ID)

    assert calls == ["observation", "registration"]
    assert [event.stage for event in result.stage_history[-4:]] == [
        "DETECTING_WICKETS",
        "OBSERVING_WICKETS",
        "GENERATING_POSE",
        "NEEDS_ADJUSTMENT",
    ]
    assert result.raw_stump_detection_summary.raw_detection_count == 6
    restored = load_scene_calibration(ANALYSIS_ID)
    assert restored.setup_frame_image_url == "/static/setup.png"
    assert restored.raw_wicket_overlay_url == "/static/raw.png"
    assert restored.anchor_version == 1


def test_ineligible_observation_never_runs_registration(
    isolated_scene: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = _synthetic_observation()
    observation.near_wicket = None
    monkeypatch.setattr(
        "services.api.services.scene_calibration_service.run_wicket_observation",
        lambda analysis_id: observation,
    )
    monkeypatch.setattr(
        "services.api.services.scene_calibration_service.run_real_pitch_registration",
        lambda analysis_id: pytest.fail("registration must not run"),
    )
    result = run_scene_calibration(ANALYSIS_ID)
    assert result.stage == "INSUFFICIENT_EVIDENCE"
    assert result.selected_candidate is None


def test_failed_stage_preserves_existing_tracking(
    isolated_scene: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = isolated_scene / ANALYSIS_ID / "tracking" / "keep.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("unchanged", encoding="utf-8")
    monkeypatch.setattr(
        "services.api.services.scene_calibration_service.run_wicket_observation",
        lambda analysis_id: (_ for _ in ()).throw(RuntimeError("detector failed")),
    )
    result = run_scene_calibration(ANALYSIS_ID)
    assert result.stage == "FAILED"
    assert marker.read_text(encoding="utf-8") == "unchanged"


def test_missing_automatic_anchor_stays_unavailable() -> None:
    observation = _synthetic_observation()
    landmark = next(
        item
        for item in observation.near_wicket.coarse_landmarks
        if item.semantic_id == "wicket_top_center"
    )
    landmark.status = "UNAVAILABLE"
    anchors = initialise_scene_anchors(observation)
    missing = next(item for item in anchors if item.semantic_id == "near_top_center")
    assert missing.video_point is None
    assert not missing.valid
    assert missing.source == "automatic"


def test_stored_scene_hydrates_legacy_setup_urls_without_redetection(
    isolated_scene: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = SceneCalibrationResult(
        analysis_id=ANALYSIS_ID,
        stage="NEEDS_ADJUSTMENT",
        updated_at=datetime.now(timezone.utc),
        setup_frame=_setup_frame(selected=True),
        message="stored",
    )
    path = isolated_scene / ANALYSIS_ID / "reports" / "scene_calibration_v1.json"
    path.parent.mkdir(parents=True)
    path.write_text(result.model_dump_json(), encoding="utf-8")
    observation = _synthetic_observation()
    observation.diagnostics.setup_frame_image_url = "/static/setup.jpg"
    observation.diagnostics.raw_detection_overlay_url = "/static/raw.jpg"
    monkeypatch.setattr(
        "services.api.services.scene_calibration_service.load_wicket_observation",
        lambda analysis_id: observation,
    )
    restored = load_scene_calibration(ANALYSIS_ID)
    assert restored.setup_frame_image_url == "/static/setup.jpg"
    assert restored.raw_wicket_overlay_url == "/static/raw.jpg"


@pytest.mark.parametrize(
    ("semantic_id", "point", "message"),
    [
        ("near_left_base", PixelPoint(x=340, y=600), "reversed"),
        ("near_top_center", PixelPoint(x=260, y=650), "above"),
        ("far_right_base", PixelPoint(x=1400, y=300), "inside"),
    ],
)
def test_anchor_validation_rejects_invalid_geometry(
    semantic_id: str,
    point: PixelPoint,
    message: str,
) -> None:
    anchors = _valid_anchors()
    index = next(i for i, item in enumerate(anchors) if item.semantic_id == semantic_id)
    anchors[index] = anchors[index].model_copy(update={"video_point": point})
    validated = validate_scene_anchors(
        anchors, image_width=1280, image_height=720
    )
    target = next(item for item in validated if item.semantic_id == semantic_id)
    assert not target.valid
    assert any(message in reason for reason in target.validation_messages)


def test_anchor_edit_preserves_origin_and_manual_provenance(
    isolated_scene: Path,
) -> None:
    result = SceneCalibrationResult(
        analysis_id=ANALYSIS_ID,
        stage="NEEDS_ADJUSTMENT",
        updated_at=datetime.now(timezone.utc),
        setup_frame=_setup_frame(selected=True),
        current_anchor_set=_valid_anchors(),
        anchor_version=1,
        message="edit",
    )
    path = isolated_scene / ANALYSIS_ID / "reports" / "scene_calibration_v1.json"
    path.parent.mkdir(parents=True)
    path.write_text(result.model_dump_json(), encoding="utf-8")
    edited = update_scene_calibration_anchors(
        ANALYSIS_ID,
        SceneCalibrationAnchorUpdateRequest(
            anchor_version=1,
            anchors=[
                SceneCalibrationAnchorInput(
                    semantic_id="near_left_base",
                    video_point=PixelPoint(x=205, y=598),
                    source="manually_adjusted",
                )
            ],
        ),
    )
    anchor = next(
        item for item in edited.current_anchor_set
        if item.semantic_id == "near_left_base"
    )
    assert anchor.source == "manually_adjusted"
    assert anchor.original_automatic_point == PixelPoint(x=200, y=600)
    assert anchor.adjustment_distance_px > 0
    with pytest.raises(VideoAnalysisServiceError, match="stale"):
        update_scene_calibration_anchors(
            ANALYSIS_ID,
            SceneCalibrationAnchorUpdateRequest(
                anchor_version=1,
                anchors=[
                    SceneCalibrationAnchorInput(
                        semantic_id="near_left_base",
                        video_point=PixelPoint(x=210, y=598),
                        source="manually_adjusted",
                    )
                ],
            ),
        )


def test_refinement_passes_native_overrides_and_optional_usage(
    isolated_scene: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = _synthetic_observation(noise_px=0.2)
    registration = _registration(observation)
    crease = _anchor(
        "near_popping_crease_left",
        100,
        640,
        kind="crease",
        source="manually_added",
    )
    validation_only = _anchor(
        "near_popping_crease_right",
        400,
        640,
        kind="crease",
        source="manually_added",
        used_for_refinement=False,
    )
    result = SceneCalibrationResult(
        analysis_id=ANALYSIS_ID,
        stage="NEEDS_ADJUSTMENT",
        updated_at=datetime.now(timezone.utc),
        setup_frame=_setup_frame(selected=True),
        current_anchor_set=[
            item.model_copy(update={"source": "manually_adjusted"})
            for item in _valid_anchors()
        ],
        optional_crease_anchors=[crease, validation_only],
        anchor_version=2,
        selected_candidate=registration.selected_candidate,
        calibration_level="VISUAL_ONLY",
        message="refine",
    )
    path = isolated_scene / ANALYSIS_ID / "reports" / "scene_calibration_v1.json"
    path.parent.mkdir(parents=True)
    path.write_text(result.model_dump_json(), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_registration(analysis_id: str, **kwargs):
        captured.update(kwargs)
        return registration

    monkeypatch.setattr(
        "services.api.services.scene_calibration_service.run_real_pitch_registration",
        fake_registration,
    )
    refined = refine_scene_calibration(
        ANALYSIS_ID, SceneCalibrationRefineRequest(anchor_version=2)
    )
    assert set(captured["point_overrides"]) == set(WICKET_ANCHOR_IDS)
    assert set(captured["crease_overrides"]) == {
        "near_popping_crease_left"
    }
    assert set(captured["manual_override_ids"]) == set(WICKET_ANCHOR_IDS)
    assert captured["result_filename"] == "real_pitch_registration_v1_refined.json"
    assert refined.refined_registration_summary is not None


def test_visual_only_cannot_be_accepted_but_can_enable_overlay(
    isolated_scene: Path,
) -> None:
    registration = _registration()
    result = SceneCalibrationResult(
        analysis_id=ANALYSIS_ID,
        stage="NEEDS_ADJUSTMENT",
        updated_at=datetime.now(timezone.utc),
        setup_frame=_setup_frame(selected=True),
        current_anchor_set=_valid_anchors(),
        anchor_version=1,
        selected_candidate=registration.selected_candidate,
        calibration_level="VISUAL_ONLY",
        validation=evaluate_calibration_candidate(
            registration.selected_candidate,
            ambiguity_score=0.9,
            anchors=_valid_anchors(),
        ),
        message="visual",
    )
    path = isolated_scene / ANALYSIS_ID / "reports" / "scene_calibration_v1.json"
    path.parent.mkdir(parents=True)
    path.write_text(result.model_dump_json(), encoding="utf-8")
    request = SceneCalibrationActionRequest(anchor_version=1)
    with pytest.raises(VideoAnalysisServiceError, match="does not permit"):
        accept_scene_calibration(ANALYSIS_ID, request)
    visual = use_visual_overlay_only(ANALYSIS_ID, request)
    assert visual.visual_overlay_enabled
    assert visual.metrics_unlocked == []


def test_accepted_snapshots_are_revisioned_and_never_overwritten(
    isolated_scene: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = _registration()
    validation = SceneCalibrationValidation(
        eligible_level="METRIC_3D_READY",
        accepted_anchor_count=6,
        manually_adjusted_anchor_count=6,
        manually_added_anchor_count=0,
        all_required_checks_passed=True,
    )
    result = SceneCalibrationResult(
        analysis_id=ANALYSIS_ID,
        stage="METRIC_3D_READY",
        updated_at=datetime.now(timezone.utc),
        setup_frame=_setup_frame(selected=True),
        current_anchor_set=_valid_anchors(),
        anchor_version=3,
        selected_candidate=registration.selected_candidate,
        validation=validation,
        calibration_level="METRIC_3D_READY",
        message="ready",
    )
    path = isolated_scene / ANALYSIS_ID / "reports" / "scene_calibration_v1.json"
    path.parent.mkdir(parents=True)
    path.write_text(result.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        "services.api.services.scene_calibration_service._rerun_physics_if_available",
        lambda analysis_id: None,
    )
    request = SceneCalibrationActionRequest(
        anchor_version=3,
        candidate_id=registration.selected_candidate.candidate_id,
    )
    first = accept_scene_calibration(ANALYSIS_ID, request)
    first_snapshot = path.parent / "accepted_scene_calibration_v1.json"
    preserved = first_snapshot.read_bytes()
    second = accept_scene_calibration(ANALYSIS_ID, request)
    assert first.accepted_calibration.revision == 1
    assert second.accepted_calibration.revision == 2
    assert first_snapshot.read_bytes() == preserved
    assert (path.parent / "accepted_scene_calibration_v1_r2.json").is_file()
    assert second.metrics_unlocked == METRIC_3D_METRICS


@pytest.mark.parametrize(
    ("level", "expected_mode"),
    [
        ("METRIC_3D_READY", "METRIC_3D"),
        ("GROUND_PLANE_READY", "METRIC_GROUND_PLANE"),
    ],
)
def test_physics_uses_only_active_accepted_assisted_calibration(
    monkeypatch: pytest.MonkeyPatch,
    level: str,
    expected_mode: str,
) -> None:
    monkeypatch.setattr(
        "services.api.services.scene_calibration_service.load_scene_calibration",
        lambda analysis_id: SimpleNamespace(
            stage=level,
            accepted_calibration=SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(
        "services.api.services.scene_calibration_service.load_active_accepted_scene_calibration",
        lambda analysis_id: SimpleNamespace(
            calibration_level=level,
            reprojection_rmse_px=2,
            revision=1,
            camera_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            distortion_coefficients=[0] * 5,
            rotation_vector=[0, 0, 0],
            rotation_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            translation_vector=[0, 0, 1],
            projection_matrix=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]],
            image_to_pitch_homography=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            pitch_to_image_homography=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            correspondence_count=6,
        ),
    )
    calibration = load_physics_calibration(ANALYSIS_ID, 1280, 720)
    assert calibration.mode == expected_mode


def test_visual_assisted_state_keeps_physics_in_image_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "services.api.services.scene_calibration_service.load_scene_calibration",
        lambda analysis_id: SimpleNamespace(
            stage="NEEDS_ADJUSTMENT",
            accepted_calibration=None,
        ),
    )
    calibration = load_physics_calibration(ANALYSIS_ID, 1280, 720)
    assert calibration.mode == "IMAGE_SPACE_ONLY"
    assert "acceptance" in calibration.failure_reason


def test_unified_api_get_returns_backend_owned_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = SceneCalibrationResult(
        analysis_id=ANALYSIS_ID,
        stage="NOT_STARTED",
        updated_at=datetime.now(timezone.utc),
        message="Detect wickets to begin scene calibration.",
    )
    monkeypatch.setattr(
        "services.api.routes.video_analysis.load_scene_calibration",
        lambda analysis_id: payload,
    )
    response = TestClient(app).get(
        f"/video-analysis/{ANALYSIS_ID}/scene-calibration"
    )
    assert response.status_code == 200
    assert response.json()["workflow"] == "ASSISTED_SCENE_CALIBRATION_V1"
    assert response.json()["stage"] == "NOT_STARTED"


def test_ui_keeps_one_workspace_and_developer_panels_collapsed() -> None:
    page = Path("apps/web/app/video-analysis/page.tsx").read_text(
        encoding="utf-8"
    )
    panel = Path(
        "apps/web/components/video-analysis/AssistedSceneCalibrationPanel.tsx"
    ).read_text(encoding="utf-8")
    assert page.count("<AssistedSceneCalibrationPanel") == 1
    assert 'sceneCalibration.stage !== "NOT_STARTED"' in page
    assert "<summary" in page and "Developer Diagnostics" in page
    before_detection = panel.split('if (result.stage === "NOT_STARTED")')[1].split(
        "return (", 1
    )[1].split(");", 1)[0]
    assert "Detect Wickets" in before_detection
    assert "Fine Adjust" not in before_detection
    assert "Accept Calibration" not in before_detection


def test_scene_orchestrator_does_not_import_detector_or_pose_research() -> None:
    source = Path(
        "services/api/services/scene_calibration_service.py"
    ).read_text(encoding="utf-8")
    assert "detect_wickets_robust" not in source
    assert "release_point" not in source
    assert "mmpose" not in source.lower()
