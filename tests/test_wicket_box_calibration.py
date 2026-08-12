"""Wicket-box calibration Stages 1-3 tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from services.api.main import app
from services.api.schemas.replay_payload import build_stage0_replay_payload
from services.api.schemas.wicket_box_calibration import (
    PixelPoint,
    StumpLandmark,
    WicketBox,
    WicketBoxCalibrationAcceptRequest,
    WicketBoxCalibrationRegisterRequest,
    validate_wicket_box_pair,
)
from services.api.services.virtual_pitch_service import (
    build_synthetic_camera,
    project_virtual_pitch,
)
from services.api.services.wicket_box_calibration_service import (
    ACCEPTED_FILENAME,
    RESULT_FILENAME,
    _anchor_landmark,
    _observation_from_landmarks,
    _registration_rejection_reasons,
    _stump_landmarks_for_wicket,
    _validate_boxes,
    accept_wicket_box_calibration,
    load_active_accepted_wicket_box_calibration,
    load_wicket_box_calibration,
    register_wicket_box_calibration,
)
from services.api.schemas.wicket_box_calibration import (
    CalibrationResult,
    WicketBoxRegistrationSummary,
)
from services.api.schemas.real_pitch_registration import RegistrationDiagnostics
from services.api.schemas.real_pitch_registration import RealPitchRegistrationResult
from services.api.schemas.video_analysis import CricketPitchGeometry
from tests.test_real_pitch_registration import _solve, _synthetic_observation


def _empty_registration_summary() -> WicketBoxRegistrationSummary:
    return WicketBoxRegistrationSummary(
        recommended=None,
        alternative=None,
        rejected=[],
        auto_selected=False,
        orientation_ambiguous=False,
        user_message="",
    )


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


ANALYSIS_ID = "analysis_20260803_213802_c2d3e4"
client = TestClient(app)


def _wicket_box(
    role: str,
    *,
    x: float = 100.0,
    y: float = 400.0,
    width: float = 80.0,
    height: float = 120.0,
    frame_index: int = 4,
) -> WicketBox:
    return WicketBox(
        role=role,
        x=x,
        y=y,
        width=width,
        height=height,
        source_image_width=1280,
        source_image_height=720,
        calibration_frame_index=frame_index,
    )


def _register_request(
    *,
    near: WicketBox | None = None,
    far: WicketBox | None = None,
    landmarks: list[StumpLandmark] | None = None,
    pitch_geometry: CricketPitchGeometry | None = None,
) -> WicketBoxCalibrationRegisterRequest:
    return WicketBoxCalibrationRegisterRequest(
        analysis_id=ANALYSIS_ID,
        calibration_frame_index=4,
        source_image_width=1280,
        source_image_height=720,
        near_wicket_box=near or _wicket_box("NEAR", y=520.0, height=140.0, width=120.0),
        far_wicket_box=far or _wicket_box("FAR", y=120.0, height=70.0, width=50.0),
        stump_landmarks=landmarks or [],
        pitch_geometry=pitch_geometry,
    )


def _synthetic_landmarks() -> list[StumpLandmark]:
    projection = project_virtual_pitch(build_synthetic_camera("centred_bowler_end"))
    pixels = {
        item.semantic_id: item.pixel_point
        for item in projection.projected_landmarks
        if item.pixel_point is not None
    }
    landmarks: list[StumpLandmark] = []
    for role, end in (("NEAR", "bowler"), ("FAR", "striker")):
        for identity, side in (("LEFT", "left"), ("MIDDLE", "middle"), ("RIGHT", "right")):
            base = pixels[f"{end}_{side}_stump_base"]
            top = pixels[f"{end}_{side}_stump_top"]
            landmarks.append(
                StumpLandmark(
                    wicket_role=role,
                    stump_identity=identity,
                    base=PixelPoint(x=base.x, y=base.y),
                    top=PixelPoint(x=top.x, y=top.y),
                    centre=PixelPoint(x=(base.x + top.x) / 2, y=(base.y + top.y) / 2),
                    confidence=0.95,
                    provenance="AUTOMATIC",
                )
            )
    return landmarks


@pytest.fixture
def isolated_wicket_box(monkeypatch: pytest.MonkeyPatch) -> Path:
    root = Path(__file__).resolve().parents[1] / ".pytest_tmp" / "wicket_box_calibration"
    root.mkdir(parents=True, exist_ok=True)
    analysis_dir = root / ANALYSIS_ID
    for folder in ("raw", "reports", "calibration"):
        (analysis_dir / folder).mkdir(parents=True, exist_ok=True)
    (analysis_dir / "raw" / "original_video.mp4").write_bytes(b"")
    monkeypatch.setattr(
        "services.api.services.wicket_box_calibration_service.VIDEO_ANALYSIS_ROOT",
        root,
    )
    monkeypatch.setattr(
        "services.api.services.real_pitch_registration_service.VIDEO_ANALYSIS_ROOT",
        root,
    )
    monkeypatch.setattr(
        "services.api.services.wicket_box_calibration_service.load_video_analysis",
        lambda analysis_id: SimpleNamespace(
            analysis_id=analysis_id,
            stored_filename="original_video.mp4",
            fps=30.0,
            width=1280,
            height=720,
        ),
    )
    monkeypatch.setattr(
        "services.api.services.real_pitch_registration_service.load_video_analysis",
        lambda analysis_id: SimpleNamespace(
            analysis_id=analysis_id,
            stored_filename="original_video.mp4",
            fps=30.0,
            width=1280,
            height=720,
        ),
    )
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "services.api.services.wicket_box_calibration_service._read_setup_frame",
        lambda analysis_id, frame_index: (frame, SimpleNamespace(fps=30.0)),
    )
    return root


class TestBoxValidation:
    def test_overlap_rejected_without_role_swap(self):
        near = _wicket_box("NEAR", x=100.0, y=400.0)
        far = _wicket_box("FAR", x=110.0, y=410.0, width=60.0, height=80.0)
        result = validate_wicket_box_pair(near, far)
        assert result.valid is False
        assert near.role == "NEAR"
        assert far.role == "FAR"

    def test_suspicious_near_far_order_rejected(self):
        near = _wicket_box("NEAR", y=120.0, height=60.0, width=40.0)
        far = _wicket_box("FAR", y=520.0, height=140.0, width=120.0)
        result = validate_wicket_box_pair(near, far)
        assert result.valid is False
        assert result.role_order_valid is False

    def test_too_small_boxes_rejected(self):
        near = _wicket_box("NEAR", y=520.0, width=10.0, height=10.0)
        far = _wicket_box("FAR", y=120.0, width=40.0, height=50.0)
        result = _validate_boxes(near, far)
        assert result.valid is False


class TestLandmarkExtraction:
    def test_missing_bails_do_not_fail_landmark_build(self):
        landmarks = _stump_landmarks_for_wicket(
            "NEAR",
            [],
            yolo_stumps=[
                {
                    "name": "left",
                    "base": {"x": 100, "y": 500},
                    "top": {"x": 100, "y": 420},
                },
                {
                    "name": "middle",
                    "base": {"x": 140, "y": 500},
                    "top": {"x": 140, "y": 420},
                },
                {
                    "name": "right",
                    "base": {"x": 180, "y": 500},
                    "top": {"x": 180, "y": 420},
                },
            ],
        )
        assert len(landmarks) == 3
        assert landmarks[0].stump_identity == "LEFT"

    def test_user_corrected_provenance_preserved(self):
        landmark = _anchor_landmark(
            "left_stump_base",
            PixelPoint(x=10, y=20),
            0.9,
            "USER_CORRECTED",
        )
        assert landmark.extraction_method == "user_corrected_wicket_box"
        assert landmark.status == "AVAILABLE"

    def test_register_preserves_automatic_landmarks_after_user_correction(
        self,
        isolated_wicket_box: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        automatic = _synthetic_landmarks()
        detect_result = CalibrationResult(
            status="DETECTED",
            analysis_id=ANALYSIS_ID,
            calibration_frame_index=4,
            source_image_width=1280,
            source_image_height=720,
            stump_landmarks=automatic,
            automatic_stump_landmarks=automatic,
            validation_status="VALID",
            message="detected",
        )
        from services.api.services import wicket_box_calibration_service as service

        service._write_result(ANALYSIS_ID, detect_result)

        corrected = [
            item.model_copy(
                update={
                    "base": PixelPoint(x=item.base.x + 3, y=item.base.y + 2),
                    "provenance": "USER_CORRECTED",
                }
            )
            for item in automatic
        ]
        observation = _observation_from_landmarks(
            analysis_id=ANALYSIS_ID,
            frame_index=4,
            fps=30.0,
            near_box=_register_request().near_wicket_box,
            far_box=_register_request().far_wicket_box,
            landmarks=corrected,
        )
        candidate, _ = _solve(observation)
        registration = _registration(observation).model_copy(
            update={"selected_candidate": candidate, "status": candidate.classification}
        )
        monkeypatch.setattr(
            "services.api.services.wicket_box_calibration_service._register_from_observation",
            lambda *args, **kwargs: (registration, [], False, _empty_registration_summary()),
        )
        monkeypatch.setattr(
            "services.api.services.wicket_box_calibration_service._registration_rejection_reasons",
            lambda candidate, registration: [],
        )

        response = register_wicket_box_calibration(
            ANALYSIS_ID,
            _register_request(landmarks=corrected),
        )

        assert response.success is True
        assert response.calibration is not None
        assert any(item.provenance == "USER_CORRECTED" for item in response.calibration.stump_landmarks)
        assert len(response.calibration.automatic_stump_landmarks) == len(automatic)
        assert all(
            item.provenance == "AUTOMATIC"
            for item in response.calibration.automatic_stump_landmarks
        )


class TestRegistrationAndAcceptance:
    def test_synthetic_geometry_registers(
        self,
        isolated_wicket_box: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        observation = _observation_from_landmarks(
            analysis_id=ANALYSIS_ID,
            frame_index=4,
            fps=30.0,
            near_box=_register_request().near_wicket_box,
            far_box=_register_request().far_wicket_box,
            landmarks=_synthetic_landmarks(),
        )
        candidate, _ = _solve(observation)
        registration = _registration(observation)
        registration = registration.model_copy(
            update={"selected_candidate": candidate, "status": candidate.classification}
        )
        monkeypatch.setattr(
            "services.api.services.wicket_box_calibration_service._register_from_observation",
            lambda *args, **kwargs: (registration, [], False, _empty_registration_summary()),
        )
        monkeypatch.setattr(
            "services.api.services.wicket_box_calibration_service._registration_rejection_reasons",
            lambda candidate, registration: [],
        )
        response = register_wicket_box_calibration(
            ANALYSIS_ID,
            _register_request(landmarks=_synthetic_landmarks()),
        )
        assert response.success is True
        assert response.calibration is not None
        assert response.calibration.camera_matrix is not None
        assert response.calibration.reprojection_rmse_px is not None

    def test_implausible_pose_rejected(
        self,
        isolated_wicket_box: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        observation = _synthetic_observation(noise_px=0.2)
        candidate, _ = _solve(observation)
        candidate = candidate.model_copy(
            update={
                "plausibility_checks": [
                    item.model_copy(
                        update={"passed": item.check_id != "camera_above_pitch"}
                    )
                    for item in candidate.plausibility_checks
                ]
            }
        )
        registration = _registration(observation).model_copy(
            update={"selected_candidate": candidate}
        )
        reasons = _registration_rejection_reasons(candidate, registration)
        assert any("below ground" in reason for reason in reasons)

    def test_acceptance_persists_full_calibration_result(
        self,
        isolated_wicket_box: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        observation = _synthetic_observation(noise_px=0.2)
        candidate, _ = _solve(observation)
        registration = _registration(observation).model_copy(
            update={"selected_candidate": candidate, "status": candidate.classification}
        )
        monkeypatch.setattr(
            "services.api.services.wicket_box_calibration_service._register_from_observation",
            lambda *args, **kwargs: (registration, [], False, _empty_registration_summary()),
        )
        monkeypatch.setattr(
            "services.api.services.wicket_box_calibration_service._register_from_observation",
            lambda *args, **kwargs: (registration, [], False, _empty_registration_summary()),
        )
        monkeypatch.setattr(
            "services.api.services.wicket_box_calibration_service._registration_rejection_reasons",
            lambda candidate, registration: [],
        )
        registered = register_wicket_box_calibration(
            ANALYSIS_ID,
            _register_request(landmarks=_synthetic_landmarks()),
        )
        assert registered.calibration is not None
        accepted = accept_wicket_box_calibration(
            ANALYSIS_ID,
            WicketBoxCalibrationAcceptRequest(
                analysis_id=ANALYSIS_ID,
                accept_registered_calibration=True,
            ),
        )
        assert accepted.success is True
        assert accepted.calibration is not None
        assert accepted.calibration.status == "ACCEPTED"
        assert accepted.calibration.accepted_at is not None
        stored = load_wicket_box_calibration(ANALYSIS_ID)
        assert stored.status == "ACCEPTED"
        accepted_path = (
            isolated_wicket_box
            / ANALYSIS_ID
            / "reports"
            / ACCEPTED_FILENAME
        )
        assert accepted_path.is_file()
        payload = json.loads(accepted_path.read_text(encoding="utf-8"))
        assert payload["frozen"] is True
        assert payload["camera_matrix"] is not None
        snapshot = load_active_accepted_wicket_box_calibration(ANALYSIS_ID)
        assert snapshot is not None
        assert snapshot["rotation_vector"] is not None
        assert snapshot["camera_world_position"] is not None


class TestApiAndRegression:
    def test_register_validates_invalid_overlap(self):
        near = _wicket_box("NEAR", y=520.0, height=140.0, width=120.0)
        far = _wicket_box("FAR", y=500.0, height=70.0, width=50.0)
        response = client.post(
            f"/video-analysis/{ANALYSIS_ID}/wicket-box-calibration/register",
            json=_register_request(near=near, far=far).model_dump(mode="json"),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is False
        assert body["validation"]["valid"] is False

    def test_replay_payload_regression_still_honest(self):
        payload = build_stage0_replay_payload(ANALYSIS_ID)
        assert payload.metrics.estimated_lateral_deviation_m.value is None
        response = client.get(f"/video-analysis/{ANALYSIS_ID}/replay-payload")
        assert response.status_code == 200
        body = response.json()
        assert body["trajectory"] == []

    def test_detect_requires_request_body(self):
        response = client.post(
            f"/video-analysis/{ANALYSIS_ID}/wicket-box-calibration/detect"
        )
        assert response.status_code == 422

    def test_persisted_detect_report_filename(self, isolated_wicket_box: Path):
        path = isolated_wicket_box / ANALYSIS_ID / "reports" / RESULT_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "status": "DETECTED",
                    "analysis_id": ANALYSIS_ID,
                    "calibration_frame_index": 4,
                    "source_image_width": 1280,
                    "source_image_height": 720,
                    "stump_landmarks": [],
                    "reprojection_diagnostics": [],
                    "validation_status": "VALID",
                    "warnings": [],
                    "message": "saved",
                }
            ),
            encoding="utf-8",
        )
        loaded = load_wicket_box_calibration(ANALYSIS_ID)
        assert loaded.status == "DETECTED"
        assert loaded.message == "saved"


class TestStabilityAndAutomaticWorkflow:
    def test_temporal_zero_but_perturbation_stable_for_box_calibration(
        self,
        isolated_wicket_box: Path,
    ) -> None:
        from services.api.services.wicket_box_calibration_service import (
            _observation_from_landmarks,
            _perturbation_stability_score,
            _register_from_observation,
        )

        landmarks = _synthetic_landmarks()
        observation = _observation_from_landmarks(
            analysis_id=ANALYSIS_ID,
            frame_index=4,
            fps=30.0,
            near_box=_register_request().near_wicket_box,
            far_box=_register_request().far_wicket_box,
            landmarks=landmarks,
        )
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        registration, _, _, summary = _register_from_observation(
            ANALYSIS_ID,
            observation,
            frame,
            assignment_hypothesis=None,
        )
        selected = registration.selected_candidate
        assert selected is not None
        assert selected.temporal_validation is not None
        assert selected.temporal_validation.stability_score == 0.0
        assert selected.uncertainty is not None
        assert selected.uncertainty.perturbation_count > 0
        score = _perturbation_stability_score(selected.uncertainty)
        assert score is not None
        assert score > 0.0
        assert summary.recommended is not None
        assert summary.recommended.physically_valid is True
        assert summary.recommended.stability_score == score

    def test_missing_landmarks_return_structured_reason(self) -> None:
        from services.api.services.wicket_box_calibration_service import (
            _landmark_validation_reasons,
        )

        incomplete = _synthetic_landmarks()[:3]
        reasons = _landmark_validation_reasons(incomplete)
        assert any("Only" in reason and "FAR" in reason for reason in reasons)

    def test_low_confidence_centre_stump_reason(self) -> None:
        from services.api.services.wicket_box_calibration_service import (
            _landmark_validation_reasons,
        )

        landmarks = [
            item.model_copy(update={"confidence": 0.1})
            if item.stump_identity == "MIDDLE" and item.wicket_role == "FAR"
            else item
            for item in _synthetic_landmarks()
        ]
        reasons = _landmark_validation_reasons(landmarks)
        assert "FAR centre stump base has low confidence" in reasons

    def test_reversed_top_base_rejected(self) -> None:
        from services.api.services.wicket_box_calibration_service import (
            _landmark_validation_reasons,
        )

        landmarks = [
            item.model_copy(
                update={
                    "top": PixelPoint(x=item.base.x, y=item.base.y + 12),
                    "base": PixelPoint(x=item.top.x, y=item.top.y),
                }
            )
            for item in _synthetic_landmarks()
        ]
        reasons = _landmark_validation_reasons(landmarks)
        assert "Landmark correspondence is inconsistent" in reasons

    def test_failed_perturbation_trials_reported_not_silent_zero(
        self,
    ) -> None:
        from services.api.schemas.real_pitch_registration import PoseUncertainty
        from services.api.services.wicket_box_calibration_service import (
            _perturbation_stability_score,
            _stability_failure_reason,
        )

        uncertainty = PoseUncertainty(
            perturbation_count=0,
            deterministic_seed=42,
            stable_for_future_metric_use=False,
            warnings=["Pose perturbation was unavailable."],
        )
        assert _perturbation_stability_score(uncertainty) == 0.0
        candidate, _ = _solve(_synthetic_observation(noise_px=0.2))
        candidate = candidate.model_copy(update={"uncertainty": uncertainty})
        assert _stability_failure_reason(candidate) == "PnP perturbation trials failed"

    def test_non_finite_spread_reports_explicit_reason(self) -> None:
        from services.api.schemas.real_pitch_registration import PoseUncertainty
        from services.api.services.wicket_box_calibration_service import (
            _stability_failure_reason,
        )

        uncertainty = PoseUncertainty.model_construct(
            perturbation_count=5,
            deterministic_seed=42,
            camera_position_spread_m=float("nan"),
            rotation_spread_degrees=1.0,
            maximum_overlay_movement_px=2.0,
            stable_for_future_metric_use=False,
            warnings=[],
        )
        candidate, _ = _solve(_synthetic_observation(noise_px=0.2))
        candidate = candidate.model_copy(update={"uncertainty": uncertainty})
        assert (
            _stability_failure_reason(candidate)
            == "Stability calculation produced a non-finite value"
        )

    def test_automatic_landmarks_register_without_user_correction(
        self,
        isolated_wicket_box: Path,
    ) -> None:
        landmarks = _synthetic_landmarks()
        response = register_wicket_box_calibration(
            ANALYSIS_ID,
            _register_request(landmarks=landmarks),
        )
        assert response.success is True
        assert response.calibration is not None
        assert all(item.provenance == "AUTOMATIC" for item in response.calibration.stump_landmarks)
        assert response.calibration.registration_summary is not None
        assert response.calibration.registration_summary.recommended is not None
        assert response.calibration.registration_summary.recommended.physically_valid is True

    def test_user_corrected_only_after_actual_edit(self) -> None:
        automatic = _synthetic_landmarks()
        assert all(item.provenance == "AUTOMATIC" for item in automatic)
        corrected = [
            item.model_copy(
                update={
                    "base": PixelPoint(x=item.base.x + 1, y=item.base.y),
                    "provenance": "USER_CORRECTED",
                }
            )
            for item in automatic
        ]
        assert any(item.provenance == "USER_CORRECTED" for item in corrected)
        assert corrected[0].provenance == "USER_CORRECTED"
        assert automatic[0].provenance == "AUTOMATIC"

    def test_debug_report_written_on_register(
        self,
        isolated_wicket_box: Path,
    ) -> None:
        from services.api.services.wicket_box_calibration_service import DEBUG_REPORT_FILENAME

        register_wicket_box_calibration(
            ANALYSIS_ID,
            _register_request(landmarks=_synthetic_landmarks()),
        )
        debug_path = (
            isolated_wicket_box / ANALYSIS_ID / "reports" / DEBUG_REPORT_FILENAME
        )
        assert debug_path.is_file()
        payload = json.loads(debug_path.read_text(encoding="utf-8"))
        assert len(payload["endpoints"]) == 12
        assert payload["stability"]["perturbation"]["perturbation_count"] > 0


SHORT_PITCH_GEOMETRY = CricketPitchGeometry(
    pitch_length_m=4.0,
    popping_crease_distance_m=1.0,
)


class TestDeclaredPitchGeometry:
    def test_declared_geometry_persists_to_accepted_snapshot(
        self,
        isolated_wicket_box: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A declared 4 m pitch must survive accept and reload unchanged."""
        observation = _observation_from_landmarks(
            analysis_id=ANALYSIS_ID,
            frame_index=4,
            fps=30.0,
            near_box=_register_request().near_wicket_box,
            far_box=_register_request().far_wicket_box,
            landmarks=_synthetic_landmarks(),
        )
        registration = _registration(observation)
        monkeypatch.setattr(
            "services.api.services.wicket_box_calibration_service._register_from_observation",
            lambda *args, **kwargs: (
                registration,
                [],
                False,
                _empty_registration_summary(),
            ),
        )
        monkeypatch.setattr(
            "services.api.services.wicket_box_calibration_service._registration_rejection_reasons",
            lambda candidate, registration: [],
        )
        monkeypatch.setattr(
            "services.api.services.wicket_box_calibration_service._acceptance_rejection_reasons",
            lambda calibration: [],
        )

        registered = register_wicket_box_calibration(
            ANALYSIS_ID,
            _register_request(
                landmarks=_synthetic_landmarks(),
                pitch_geometry=SHORT_PITCH_GEOMETRY,
            ),
        )
        assert registered.calibration is not None
        assert registered.calibration.pitch_geometry is not None
        assert registered.calibration.pitch_geometry.pitch_length_m == 4.0

        accepted = accept_wicket_box_calibration(
            ANALYSIS_ID,
            WicketBoxCalibrationAcceptRequest(
                analysis_id=ANALYSIS_ID,
                accept_registered_calibration=True,
            ),
        )
        assert accepted.success is True
        assert accepted.calibration is not None
        assert accepted.calibration.pitch_geometry.pitch_length_m == 4.0

        # Survives the round trip through the frozen snapshot on disk.
        snapshot = load_active_accepted_wicket_box_calibration(ANALYSIS_ID)
        assert snapshot is not None
        assert snapshot["pitch_geometry"]["pitch_length_m"] == 4.0
        assert load_wicket_box_calibration(ANALYSIS_ID).pitch_geometry.pitch_length_m == 4.0

    def test_absent_geometry_stays_regulation(
        self,
        isolated_wicket_box: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Omitting the field must keep every existing full-size flow unchanged."""
        observation = _observation_from_landmarks(
            analysis_id=ANALYSIS_ID,
            frame_index=4,
            fps=30.0,
            near_box=_register_request().near_wicket_box,
            far_box=_register_request().far_wicket_box,
            landmarks=_synthetic_landmarks(),
        )
        registration = _registration(observation)
        monkeypatch.setattr(
            "services.api.services.wicket_box_calibration_service._register_from_observation",
            lambda *args, **kwargs: (
                registration,
                [],
                False,
                _empty_registration_summary(),
            ),
        )
        monkeypatch.setattr(
            "services.api.services.wicket_box_calibration_service._registration_rejection_reasons",
            lambda candidate, registration: [],
        )

        registered = register_wicket_box_calibration(
            ANALYSIS_ID,
            _register_request(landmarks=_synthetic_landmarks()),
        )

        assert registered.calibration is not None
        assert registered.calibration.pitch_geometry is None

    def test_declared_geometry_reaches_the_solver(
        self,
        isolated_wicket_box: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The register request's geometry must arrive at the registration call."""
        seen: dict[str, object] = {}

        def _capture(*args, **kwargs):
            seen["dimensions"] = kwargs.get("dimensions")
            observation = _observation_from_landmarks(
                analysis_id=ANALYSIS_ID,
                frame_index=4,
                fps=30.0,
                near_box=_register_request().near_wicket_box,
                far_box=_register_request().far_wicket_box,
                landmarks=_synthetic_landmarks(),
            )
            return (
                _registration(observation),
                [],
                False,
                _empty_registration_summary(),
            )

        monkeypatch.setattr(
            "services.api.services.wicket_box_calibration_service._register_from_observation",
            _capture,
        )
        monkeypatch.setattr(
            "services.api.services.wicket_box_calibration_service._registration_rejection_reasons",
            lambda candidate, registration: [],
        )

        register_wicket_box_calibration(
            ANALYSIS_ID,
            _register_request(
                landmarks=_synthetic_landmarks(),
                pitch_geometry=SHORT_PITCH_GEOMETRY,
            ),
        )

        assert seen["dimensions"] == SHORT_PITCH_GEOMETRY.to_dimensions()
        assert seen["dimensions"].pitch_length_m == 4.0
