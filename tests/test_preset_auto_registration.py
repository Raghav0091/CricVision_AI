"""Contract and architecture tests for preset-constrained auto-registration V1.

The implementation is developed in a separate worktree.  Until those modules are
merged, this file is skipped as one unit; once present, every behavioral assertion
becomes mandatory.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_NAME = "services.api.schemas.preset_auto_registration"
SERVICE_NAME = "services.api.services.preset_auto_registration"

if importlib.util.find_spec(SCHEMA_NAME) is None or importlib.util.find_spec(SERVICE_NAME) is None:
    pytest.skip(
        "preset auto-registration implementation is not merged into this worktree",
        allow_module_level=True,
    )

schemas = __import__(SCHEMA_NAME, fromlist=["*"])
service = __import__(SERVICE_NAME, fromlist=["*"])


def _public(module: Any, *names: str) -> Any:
    for name in names:
        value = getattr(module, name, None)
        if value is not None:
            return value
    pytest.fail(f"{module.__name__} must expose one of: {', '.join(names)}")


def _dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    pytest.fail(f"expected a Pydantic model or mapping, got {type(value)!r}")


def _preset_payload() -> dict[str, Any]:
    return {
        "preset_id": "STANDARD_REAR_WICKET_NET_V1",
        "preset_name": "Standard rear-wicket net",
        "version": "v1",
        "description": "Broad development assumptions for a fixed rear-wicket camera.",
        "intended_use": "rear_wicket_practice_net",
        "camera_end": "bowler",
        "pitch_profile": "full_pitch",
        "native_orientation": "PORTRAIT_OR_LANDSCAPE",
        "expected_aspect_ratio_range": {
            "minimum_long_edge_to_short_edge_ratio": 1.25,
            "maximum_long_edge_to_short_edge_ratio": 2.25,
        },
        "nominal_camera_height_m": 1.8,
        "camera_height_bounds_m": {"minimum_m": 0.8, "maximum_m": 3.5},
        "nominal_distance_behind_wicket_m": 5.0,
        "distance_bounds_m": {"minimum_m": 1.0, "maximum_m": 12.0},
        "nominal_lateral_offset_m": 0.0,
        "lateral_offset_bounds_m": {"minimum_m": -4.0, "maximum_m": 4.0},
        "nominal_yaw_deg": 0.0,
        "yaw_bounds_deg": {"minimum_deg": -20.0, "maximum_deg": 20.0},
        "nominal_pitch_deg": -4.0,
        "pitch_bounds_deg": {"minimum_deg": -25.0, "maximum_deg": 12.0},
        "nominal_roll_deg": 0.0,
        "roll_bounds_deg": {"minimum_deg": -10.0, "maximum_deg": 10.0},
        "nominal_horizontal_fov_deg": 45.0,
        "horizontal_fov_bounds_deg": {"minimum_deg": 28.0, "maximum_deg": 80.0},
        "image_left_mapping": "IMAGE_LEFT_IS_PITCH_LEFT",
        "distortion_policy": "ZERO_DISTORTION",
        "both_wickets_required": True,
        "minimum_frame_support": 3,
        "minimum_wicket_confidence": 0.45,
        "source": "development_assumption",
        "development_only": True,
        "warnings": ["Preset values are bounded assumptions, not measured camera metadata."],
    }


def _preset() -> Any:
    model = _public(schemas, "CameraSetupPreset")
    return model(**_preset_payload())


def _observation_payload(
    *,
    width: int = 720,
    height: int = 1280,
    rotation_degrees: int = 0,
    frame_count: int = 5,
    both_wickets: bool = True,
    clipped: bool = False,
    distortion_mode: str = "ZERO_DISTORTION",
) -> dict[str, Any]:
    return {
        "image_width": width,
        "image_height": height,
        "rotation_degrees": rotation_degrees,
        "distortion_mode": distortion_mode,
        "camera_end": "bowler",
        "setup_frame_available": True,
        "supporting_frame_count": frame_count,
        "near_wicket_available": both_wickets,
        "far_wicket_available": both_wickets,
        "minimum_observed_confidence": 0.8,
        "severe_clipping": clipped,
        "nested_false_wicket_evidence": False,
    }


def _compatibility(preset: Any, **changes: Any) -> dict[str, Any]:
    payload = _observation_payload(**changes)
    check = _public(service, "check_preset_compatibility", "evaluate_preset_compatibility")
    return _dump(check(preset, payload))


def _synthetic_observations(
    *, noise_px: float = 0.0, outlier: bool = False, weak: bool = False
) -> dict[str, Any]:
    """Two wicket envelopes across five frames from one fixed synthetic camera."""
    rng = np.random.default_rng(20260801)
    frames: list[dict[str, Any]] = []
    for frame_index in range(5):
        jitter = rng.normal(0.0, noise_px, size=8)
        near = np.array([252.0, 707.0, 216.0, 344.0]) + jitter[:4]
        far = np.array([326.0, 265.0, 69.0, 112.0]) + jitter[4:]
        if outlier and frame_index == 3:
            near += np.array([140.0, -90.0, 80.0, 70.0])
        frames.append(
            {
                "frame_index": frame_index * 3,
                "near_wicket_bbox": near.tolist(),
                "far_wicket_bbox": far.tolist(),
                "near_confidence": 0.52 if weak else 0.92,
                "far_confidence": 0.48 if weak else 0.88,
                "near_clipped": False,
                "far_clipped": False,
            }
        )
    return {
        "image_width": 720,
        "image_height": 1280,
        "setup_frame_index": 0,
        "frames": frames,
        "point_anchors": [] if weak else [
            {"semantic_id": "near_base_center", "x": 360.0, "y": 1051.0, "uncertainty_px": 2.0},
            {"semantic_id": "far_base_center", "x": 360.0, "y": 377.0, "uncertainty_px": 3.0},
        ],
    }


def _fit(*, noise_px: float = 0.0, outlier: bool = False, weak: bool = False) -> dict[str, Any]:
    fit = _public(service, "fit_bounded_camera", "fit_preset_registration")
    return _dump(
        fit(
            preset=_preset(),
            observations=_synthetic_observations(noise_px=noise_px, outlier=outlier, weak=weak),
            deterministic_seed=20260801,
        )
    )


def _parameters(result: dict[str, Any]) -> dict[str, float]:
    return result.get("fitted_parameters") or result.get("parameters") or {}


def test_rear_wicket_preset_is_versioned_explicit_and_bounded() -> None:
    data = _dump(_preset())
    assert data["preset_id"] == "STANDARD_REAR_WICKET_NET_V1"
    assert data["version"] == "v1"
    assert data["development_only"] is True
    for field in (
        "camera_height_bounds_m", "distance_bounds_m", "lateral_offset_bounds_m",
        "yaw_bounds_deg", "pitch_bounds_deg", "roll_bounds_deg", "horizontal_fov_bounds_deg",
    ):
        bounds = data[field]
        low = next(value for key, value in bounds.items() if key.startswith("minimum_"))
        high = next(value for key, value in bounds.items() if key.startswith("maximum_"))
        assert low < high
    assert all(key.endswith(("_m", "_deg", "_degrees", "_range", "_policy", "_only", "_required", "_support", "_confidence", "_mapping", "_use", "_end", "_profile", "_orientation", "_id", "_name", "version", "description", "source", "warnings")) for key in data)


@pytest.mark.parametrize("field", [
    "camera_height_bounds_m", "distance_bounds_m", "lateral_offset_bounds_m",
    "yaw_bounds_deg", "pitch_bounds_deg", "roll_bounds_deg", "horizontal_fov_bounds_deg",
])
def test_preset_rejects_inverted_bounds(field: str) -> None:
    payload = _preset_payload()
    keys = list(payload[field])
    payload[field] = {
        keys[0]: payload[field][keys[1]],
        keys[1]: payload[field][keys[0]],
    }
    with pytest.raises(ValidationError):
        _public(schemas, "CameraSetupPreset")(**payload)


def test_preset_rejects_missing_orientation_and_unsupported_distortion() -> None:
    model = _public(schemas, "CameraSetupPreset")
    missing = _preset_payload()
    missing.pop("native_orientation")
    with pytest.raises(ValidationError):
        model(**missing)
    unsupported = _preset_payload()
    unsupported["distortion_policy"] = "OPTIMISE_ARBITRARY_DISTORTION"
    with pytest.raises(ValidationError):
        model(**unsupported)


@pytest.mark.parametrize("width,height", [(720, 1280), (1920, 1080)])
def test_portrait_and_landscape_are_compatible(width: int, height: int) -> None:
    result = _compatibility(_preset(), width=width, height=height)
    assert result["status"] in {"COMPATIBLE", "COMPATIBLE_WITH_WARNINGS"}


@pytest.mark.parametrize(
    "changes,reason_fragment",
    [
        ({"rotation_degrees": 90}, "rotation"),
        ({"width": 4096, "height": 320}, "aspect"),
        ({"frame_count": 0}, "frame"),
        ({"both_wickets": False}, "wicket"),
        ({"clipped": True}, "clip"),
        ({"distortion_mode": "FISHEYE"}, "distortion"),
    ],
)
def test_incompatible_inputs_stop_before_fitting(changes: dict[str, Any], reason_fragment: str) -> None:
    result = _compatibility(_preset(), **changes)
    assert result["status"] == "INCOMPATIBLE"
    reasons = " ".join(
        reason.get("reason_code", "") if isinstance(reason, dict) else str(reason)
        for reason in result.get("reasons", [])
    ).lower()
    assert reason_fragment in reasons


def test_exact_and_noisy_synthetic_recovery_stays_inside_every_bound() -> None:
    for result in (_fit(), _fit(noise_px=2.5)):
        params = _parameters(result)
        preset = _preset_payload()
        mapping = {
            "camera_height_m": "camera_height_bounds_m",
            "distance_behind_wicket_m": "distance_bounds_m",
            "lateral_offset_m": "lateral_offset_bounds_m",
            "yaw_deg": "yaw_bounds_deg", "pitch_deg": "pitch_bounds_deg",
            "roll_deg": "roll_bounds_deg", "horizontal_fov_deg": "horizontal_fov_bounds_deg",
        }
        for parameter, bound_name in mapping.items():
            assert parameter in params
            bounds = preset[bound_name]
            low = next(value for key, value in bounds.items() if key.startswith("minimum_"))
            high = next(value for key, value in bounds.items() if key.startswith("maximum_"))
            assert low <= params[parameter] <= high
        assert result.get("robust_loss") in {"soft_l1", "huber"}


def test_weak_evidence_uses_priors_and_strong_evidence_can_correct_nominal() -> None:
    weak = _fit(weak=True, noise_px=3.0)
    strong = _fit(noise_px=0.5)
    assert weak.get("preset_prior_applied") is True
    assert strong.get("strong_evidence_overrode_nominal") is True
    assert strong.get("anchor_metrics", {}).get("rmse_px", float("inf")) < 5.0


def test_robust_outlier_does_not_move_one_fixed_camera_to_bad_frame() -> None:
    clean = _fit(noise_px=1.0)
    contaminated = _fit(noise_px=1.0, outlier=True)
    assert contaminated.get("outlier_frame_ids") == [9]
    assert contaminated.get("temporal_metrics", {}).get("successful_frame_count", 0) >= 4
    assert abs(_parameters(clean)["camera_height_m"] - _parameters(contaminated)["camera_height_m"]) < 0.25
    assert contaminated.get("camera_pose_count", 1) == 1


def test_candidate_order_and_uncertainty_are_deterministic() -> None:
    first = _fit(noise_px=1.5)
    second = _fit(noise_px=1.5)
    assert first.get("attempted_candidate_ids") == second.get("attempted_candidate_ids")
    assert first.get("selected_candidate_id") == second.get("selected_candidate_id")
    assert first.get("uncertainty") == second.get("uncertainty")


def test_temporal_instability_physical_invalidity_and_uncertainty_downgrade_status() -> None:
    classify = _public(service, "classify_auto_registration", "classify_registration_result")
    base = {
        "compatibility_status": "COMPATIBLE",
        "both_wickets_available": True,
        "fit_valid": True,
        "physical_checks_passed": True,
        "temporal_stability_score": 0.9,
        "uncertainty_acceptable": True,
        "overlay_usable": True,
    }
    assert classify(**base) in {"AUTO_REGISTRATION_READY", "VISUAL_OVERLAY_READY"}
    assert classify(**(base | {"temporal_stability_score": 0.1})) == "NEEDS_ASSISTANCE"
    assert classify(**(base | {"physical_checks_passed": False})) in {"NEEDS_ASSISTANCE", "FAILED"}
    assert classify(**(base | {"uncertainty_acceptable": False})) != "AUTO_REGISTRATION_READY"
    assert classify(**(base | {"both_wickets_available": False})) == "INSUFFICIENT_WICKETS"
    assert classify(**(base | {"compatibility_status": "INCOMPATIBLE"})) == "PRESET_INCOMPATIBLE"


def test_result_contract_never_accepts_or_unlocks_metrics() -> None:
    result_model = _public(schemas, "PresetAutoRegistrationResult")
    fields = result_model.model_fields
    assert "production_accepted" in fields
    assert "metrics_unlocked" in fields
    assert fields["production_accepted"].default is False
    assert fields["metrics_unlocked"].default_factory() == []


def test_registration_service_reuses_observations_without_importing_detector() -> None:
    source_path = ROOT / "services/api/services/preset_auto_registration.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    source = source_path.read_text(encoding="utf-8")
    assert not any("stump_detector" in name or "ball_detector" in name for name in imports)
    assert "detect_wickets_robust" not in source
    assert "load_wicket_observation" in source
    assert "run_wicket_observation" in source


def test_registration_service_does_not_duplicate_geometry_pnp_or_bridge() -> None:
    source = (ROOT / "services/api/services/preset_auto_registration.py").read_text(encoding="utf-8")
    forbidden = (
        "cv2.solvePnP", "cv2.solvePnPRansac", "solvePnPRefineLM",
        "PITCH_LENGTH", "WICKET_WIDTH", "STUMP_HEIGHT",
        "opencvToThree", "projectionMatrix.elements",
    )
    assert not [token for token in forbidden if token in source]
    assert "real_pitch_registration_service" in source
    assert "virtual_pitch_service" in source
    assert "camera_bridge_service" in source


def test_run_defaults_reuse_observations_and_never_redetects_implicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    run = _public(service, "run_preset_auto_registration")
    signature = inspect.signature(run)
    assert signature.parameters["reuse_existing_observations"].default is True
    assert signature.parameters["force_redetect"].default is False
    calls = {"load": 0, "run": 0}
    captured: dict[str, Any] = {}
    monkeypatch.setattr(service, "load_wicket_observation", lambda _analysis_id: calls.__setitem__("load", calls["load"] + 1) or object())
    monkeypatch.setattr(service, "run_wicket_observation", lambda _analysis_id: calls.__setitem__("run", calls["run"] + 1) or object())
    monkeypatch.setattr(service, "_auto_register_from_observation", lambda **kwargs: captured.update(kwargs) or {"status": "NEEDS_ASSISTANCE", "production_accepted": False, "metrics_unlocked": []})
    run("analysis_test", preset_id="STANDARD_REAR_WICKET_NET_V1")
    assert calls == {"load": 1, "run": 0}
    assert captured["detection_reused"] is True
    assert captured["observation_source"] == "PERSISTED_WICKET_OBSERVATION_V1"


def test_explicit_redetect_runs_observation_once(monkeypatch: pytest.MonkeyPatch) -> None:
    run = _public(service, "run_preset_auto_registration")
    calls = {"run": 0}
    captured: dict[str, Any] = {}
    monkeypatch.setattr(service, "load_wicket_observation", lambda _analysis_id: pytest.fail("forced redetection must not load stale evidence first"))
    monkeypatch.setattr(service, "run_wicket_observation", lambda _analysis_id: calls.__setitem__("run", calls["run"] + 1) or object())
    monkeypatch.setattr(service, "_auto_register_from_observation", lambda **kwargs: captured.update(kwargs) or {"status": "NEEDS_ASSISTANCE", "production_accepted": False, "metrics_unlocked": []})
    run("analysis_test", preset_id="STANDARD_REAR_WICKET_NET_V1", force_redetect=True)
    assert calls["run"] == 1
    assert captured["detection_reused"] is False
    assert captured["observation_source"] == "NEW_WICKET_OBSERVATION_V1"


def test_persistence_writes_only_unaccepted_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    persist = _public(service, "persist_preset_auto_registration", "_persist_result")
    result = {"preset_auto_registration_version": "v1", "production_accepted": False, "metrics_unlocked": []}
    path = persist("analysis_test", result, reports_directory=tmp_path)
    assert Path(path).name == "preset_auto_registration_v1.json"
    assert list(tmp_path.glob("*accepted*")) == []
    assert list(tmp_path.glob("*snapshot*")) == []
