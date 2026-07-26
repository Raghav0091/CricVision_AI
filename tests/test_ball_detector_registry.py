from pathlib import Path

from services.api.routes.video_analysis import get_detector_models
from services.api.schemas.video_analysis import VideoBallDetectionStartRequest
from services.api.services.ball_detector_registry import (
    AUTOMATIC_MODEL_KEY,
    BALL_DETECTOR_MODELS,
    E2_PATH,
    E3_PATH,
    E4C_PATH,
    resolve_ball_detector_model,
)
from services.api.services.video_ball_detection_job_store import (
    VideoBallDetectionJobStore,
)


def test_start_request_defaults_to_automatic() -> None:
    request = VideoBallDetectionStartRequest()

    assert request.ball_detector_model_key == AUTOMATIC_MODEL_KEY


def test_start_request_keeps_unknown_key_for_safe_registry_fallback() -> None:
    request = VideoBallDetectionStartRequest(
        ball_detector_model_key="not-approved"
    )

    assert request.ball_detector_model_key == "not-approved"
    assert resolve_ball_detector_model(request.ball_detector_model_key).model_key == (
        AUTOMATIC_MODEL_KEY
    )


def test_explicit_model_keys_resolve_to_distinct_approved_paths() -> None:
    resolved = {
        key: resolve_ball_detector_model(key).path
        for key in ("e2", "e3", "e4c")
    }

    assert resolved == {
        "e2": E2_PATH,
        "e3": E3_PATH,
        "e4c": E4C_PATH,
    }
    assert len(set(resolved.values())) == 3


def test_automatic_preserves_original_e2_first_behavior() -> None:
    resolved = resolve_ball_detector_model(AUTOMATIC_MODEL_KEY)

    assert resolved.model_key == AUTOMATIC_MODEL_KEY
    assert resolved.selected_key == "e2"
    assert resolved.path == E2_PATH


def test_model_endpoint_reports_registry_availability() -> None:
    response = get_detector_models()

    assert response.default_key == AUTOMATIC_MODEL_KEY
    assert [option.key for option in response.models] == list(
        BALL_DETECTOR_MODELS
    )
    assert all(
        option.available
        == BALL_DETECTOR_MODELS[option.key].available
        for option in response.models
    )


def test_detection_job_preserves_selected_model() -> None:
    store = VideoBallDetectionJobStore()

    job = store.create("analysis_test", 12, "e3", "E3 - Motion Blur")

    assert job is not None
    assert job["ball_detector_model_key"] == "e3"
    assert job["ball_detector_model_name"] == "E3 - Motion Blur"


def test_registry_contains_only_repository_local_paths() -> None:
    project_root = Path(__file__).resolve().parents[1]

    for model in BALL_DETECTOR_MODELS.values():
        assert model.paths
        for path in model.paths:
            path.relative_to(project_root)
