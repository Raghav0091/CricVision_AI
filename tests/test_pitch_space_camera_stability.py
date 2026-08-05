from __future__ import annotations

import ast
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from services.api.schemas.pitch_space_analysis import (
    CameraStabilityResult,
    StableWicketBox,
)
from services.api.services.wicket_box_stabilization_service import (
    assess_camera_stability,
)


ROOT = Path(__file__).resolve().parents[1]


def _stable_box(
    role: str,
    *,
    centre_spread_px: float,
    size_spread_ratio: float,
    width: float = 80,
    height: float = 160,
) -> StableWicketBox:
    return StableWicketBox(
        perspective_role=role,
        x=400,
        y=350,
        width=width,
        height=height,
        confidence=0.85,
        frame_support=4,
        supporting_frame_indices=[0, 5, 10, 15],
        centre_spread_px=centre_spread_px,
        size_spread_ratio=size_spread_ratio,
        clipped=False,
        source="persisted",
    )


def test_camera_status_contract_covers_fixed_drift_unstable_and_unavailable() -> None:
    statuses = set(get_args(CameraStabilityResult.model_fields["status"].annotation))
    assert statuses == {
        "FIXED_CAMERA",
        "MINOR_DRIFT",
        "UNSTABLE_CAMERA",
        "UNAVAILABLE",
    }


@pytest.mark.parametrize(
    ("status", "centre", "scale", "confidence", "reliable_until"),
    [
        ("FIXED_CAMERA", 0.004, 0.006, 0.95, None),
        ("MINOR_DRIFT", 0.025, 0.035, 0.65, None),
        ("UNSTABLE_CAMERA", 0.18, 0.22, 0.2, 75),
        ("UNAVAILABLE", None, None, 0.0, None),
    ],
)
def test_camera_stability_evidence_round_trips_without_inventing_values(
    status: str,
    centre: float | None,
    scale: float | None,
    confidence: float,
    reliable_until: int | None,
) -> None:
    result = CameraStabilityResult(
        status=status,
        frames_checked=[0, 25, 50] if status != "UNAVAILABLE" else [],
        maximum_centre_drift_ratio=centre,
        maximum_scale_change_ratio=scale,
        confidence=confidence,
        reliable_until_frame=reliable_until,
        warnings=["wicket evidence unavailable"] if status == "UNAVAILABLE" else [],
    )
    restored = CameraStabilityResult.model_validate_json(result.model_dump_json())
    assert restored == result
    assert restored.maximum_centre_drift_ratio == centre
    assert restored.maximum_scale_change_ratio == scale


def test_invalid_camera_status_and_non_finite_drift_are_rejected() -> None:
    with pytest.raises(ValidationError):
        CameraStabilityResult(status="RECALIBRATE", confidence=0.5)
    with pytest.raises(ValidationError):
        CameraStabilityResult(
            status="UNSTABLE_CAMERA",
            maximum_centre_drift_ratio=float("nan"),
            confidence=0.2,
        )


def test_camera_stability_uses_deterministic_checks_and_never_recalibrates() -> None:
    path = ROOT / "services/api/services/wicket_box_stabilization_service.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "random" not in imported_roots
    lowered = source.lower()
    assert "manual_recalibration" not in lowered
    assert "recalibrate_every_frame" not in lowered


@pytest.mark.parametrize(
    ("centre_spread_px", "scale_change", "expected"),
    [
        (2.0, 0.02, "FIXED_CAMERA"),
        (24.0, 0.20, "MINOR_DRIFT"),
        (50.0, 0.10, "UNSTABLE_CAMERA"),
        (2.0, 0.40, "UNSTABLE_CAMERA"),
    ],
)
def test_camera_stability_classifies_fixed_drift_translation_and_zoom(
    centre_spread_px: float,
    scale_change: float,
    expected: str,
) -> None:
    result = assess_camera_stability(
        _stable_box(
            "NEAR",
            centre_spread_px=centre_spread_px,
            size_spread_ratio=scale_change,
        ),
        _stable_box(
            "FAR",
            centre_spread_px=centre_spread_px / 2,
            size_spread_ratio=scale_change,
            width=40,
            height=80,
        ),
    )
    assert result.status == expected
    assert result.frames_checked == [0, 5, 10, 15]


def test_missing_wicket_evidence_is_unavailable_not_assumed_fixed() -> None:
    result = assess_camera_stability(
        _stable_box("NEAR", centre_spread_px=1, size_spread_ratio=0.01),
        None,
    )
    assert result.status == "UNAVAILABLE"
    assert result.confidence == 0
    assert result.maximum_centre_drift_ratio is None
    assert result.maximum_scale_change_ratio is None
    assert result.warnings


def test_unstable_camera_contract_exposes_metric_cutoff() -> None:
    result = CameraStabilityResult(
        status="UNSTABLE_CAMERA",
        frames_checked=[0, 25, 50, 75],
        maximum_centre_drift_ratio=0.2,
        maximum_scale_change_ratio=0.15,
        confidence=0.25,
        reliable_until_frame=50,
        warnings=["pitch-space metrics unavailable after frame 50"],
    )
    assert result.reliable_until_frame == 50
    assert any("unavailable after frame 50" in item for item in result.warnings)
