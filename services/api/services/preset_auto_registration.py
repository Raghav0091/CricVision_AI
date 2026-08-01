"""Preset-bounded automatic registration from persisted wicket evidence.

This service deliberately owns orchestration only.  Wicket extraction, pitch
geometry, projection, and the existing PnP camera candidate machinery remain
owned by their V1 services.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from ..schemas.camera_bridge import CameraBridgeInput
from ..schemas.real_pitch_registration import (
    CameraIntrinsicsCandidate,
    CameraPoseCandidate,
    PlausibilityCheck,
    RealProjectedPitchGeometry,
    RefinementDiagnostics,
)
from ..schemas.wicket_observation import (
    PixelBox,
    RawWicketDetection,
    WicketObservationResult,
)
from .real_pitch_registration_service import (
    _box_iou,
    _camera_position,
    _line_residuals,
    _project,
    _projected_wicket_box,
    _virtual_camera,
    build_registration_correspondences,
    load_real_pitch_registration,
)
from .camera_bridge_service import DEFAULT_FAR_M, DEFAULT_NEAR_M, _distortion
from .camera_preset_parameterization import (
    PresetCameraParameters,
    build_opencv_camera_from_preset_parameters,
    decompose_opencv_camera_to_preset_parameters,
    denormalize_parameters,
    normalize_parameters,
)
from .video_analysis_service import (
    VIDEO_ANALYSIS_ROOT,
    VideoAnalysisServiceError,
    load_video_analysis,
)
from .virtual_pitch_service import (
    build_virtual_pitch_specification,
    project_virtual_pitch,
)
from .wicket_observation_service import (
    RESULT_FILENAME as WICKET_RESULT_FILENAME,
    load_wicket_observation,
    run_wicket_observation,
)


RESULT_FILENAME = "preset_auto_registration_v1.json"
RESULT_VERSION = "v1"
OPTIMISATION_SEED = 2718
UNCERTAINTY_PERTURBATIONS = 8
MAX_CANDIDATES = 18


@dataclass(frozen=True)
class _Preset:
    preset_id: str
    version: str
    camera_end: str
    image_left_mapping: str
    nominal: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    minimum_frame_support: int
    minimum_wicket_confidence: float
    both_wickets_required: bool
    distortion_policy: str


@dataclass(frozen=True)
class _Evidence:
    observation: WicketObservationResult
    assignment: str
    lateral_mapping: str
    correspondences: list[Any]
    frame_boxes: dict[tuple[int, str], PixelBox]
    image_width: int
    image_height: int


@dataclass(frozen=True)
class _Fit:
    candidate_id: str
    seed_source: str
    initial: np.ndarray
    parameters: np.ndarray
    converged: bool
    iterations: int
    initial_cost: float
    final_cost: float
    active_bounds: list[str]
    anchor_rmse_px: float | None
    median_anchor_error_px: float | None
    near_iou: float | None
    far_iou: float | None
    temporal: dict[str, Any]
    physical_checks: list[dict[str, Any]]
    score: float
    eligible: bool
    rejection_reasons: list[str]
    objective_components: dict[str, Any]


@dataclass(frozen=True)
class _ObjectiveProfile:
    point_weight: float = 1.0
    coarse_point_weight: float = 1.0
    line_weight: float = 1.0
    envelope_weight: float = 1.0
    temporal_weight: float = 1.0
    prior_weight: float = 1.0
    coarse_uncertainty_floor_ratio: float = 0.35
    correlated_evidence_normalization: bool = True


_LEGACY_OBJECTIVE = _ObjectiveProfile(
    coarse_uncertainty_floor_ratio=0.0,
    correlated_evidence_normalization=False,
)
_CORRELATED_SOFT_OBJECTIVE = _ObjectiveProfile(prior_weight=2.0)
# One detector/Hough region produces six coarse points, four lines, and repeated
# frame boxes. Treating those as independent 1.66 px measurements caused 99.4%
# of the assisted-camera data loss and pulled height to its ceiling. The normal
# profile preserves the evidence but applies semantic uncertainty, correlation
# normalization, and enough preset regularization to keep weak geometry bounded.
_NORMAL_OBJECTIVE = _CORRELATED_SOFT_OBJECTIVE


_PARAMETER_NAMES = (
    "lateral_offset_m",
    "distance_behind_wicket_m",
    "camera_height_m",
    "yaw_deg",
    "pitch_deg",
    "roll_deg",
    "horizontal_fov_deg",
)


def _get(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _number(source: Any, name: str, default: float) -> float:
    value = _get(source, name, default)
    if hasattr(value, "value"):
        value = value.value
    return float(value)


def _bounds(source: Any, name: str, nominal: float) -> tuple[float, float]:
    value = _get(source, name)
    if value is None:
        raise ValueError(f"Preset field {name!r} is required.")
    if isinstance(value, Mapping):
        low = value.get("minimum", value.get("min", value.get("lower")))
        high = value.get("maximum", value.get("max", value.get("upper")))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        low, high = value[0], value[1]
    else:
        low = _get(
            value,
            "minimum",
            _get(
                value,
                "minimum_m",
                _get(value, "minimum_deg", _get(value, "lower", _get(value, "min"))),
            ),
        )
        high = _get(
            value,
            "maximum",
            _get(
                value,
                "maximum_m",
                _get(value, "maximum_deg", _get(value, "upper", _get(value, "max"))),
            ),
        )
    if low is None or high is None:
        raise ValueError(f"Preset bound {name!r} must expose lower and upper values.")
    low_value, high_value = float(low), float(high)
    if not math.isfinite(low_value) or not math.isfinite(high_value):
        raise ValueError(f"Preset bound {name!r} must be finite.")
    if low_value >= high_value or not low_value <= nominal <= high_value:
        raise ValueError(f"Preset nominal value is outside {name!r}.")
    return low_value, high_value


def _normalise_preset(preset: Any) -> _Preset:
    fields = (
        ("nominal_lateral_offset_m", "lateral_offset_bounds_m"),
        ("nominal_distance_behind_wicket_m", "distance_bounds_m"),
        ("nominal_camera_height_m", "camera_height_bounds_m"),
        ("nominal_yaw_deg", "yaw_bounds_deg"),
        ("nominal_pitch_deg", "pitch_bounds_deg"),
        ("nominal_roll_deg", "roll_bounds_deg"),
        ("nominal_horizontal_fov_deg", "horizontal_fov_bounds_deg"),
    )
    nominal = np.asarray([_number(preset, item[0], 0.0) for item in fields])
    limits = [
        _bounds(preset, bound, float(value))
        for value, (_, bound) in zip(nominal, fields, strict=True)
    ]
    camera_end = str(_get(preset, "camera_end", "bowler")).lower()
    if camera_end not in {"bowler", "striker"}:
        raise ValueError("Preset camera_end must be bowler or striker.")
    mapping = str(
        _get(preset, "image_left_mapping", "image_left_to_world_left")
    ).lower()
    aliases = {
        "world_left": "image_left_to_world_left",
        "world_right": "image_left_to_world_right",
        "image_left_is_pitch_left": "image_left_to_world_left",
        "image_left_is_pitch_right": "image_left_to_world_right",
    }
    mapping = aliases.get(mapping, mapping)
    if mapping not in {
        "image_left_to_world_left",
        "image_left_to_world_right",
    }:
        raise ValueError("Preset image_left_mapping is unsupported.")
    return _Preset(
        preset_id=str(_get(preset, "preset_id", "unknown")),
        version=str(_get(preset, "version", "v1")),
        camera_end=camera_end,
        image_left_mapping=mapping,
        nominal=nominal,
        lower=np.asarray([item[0] for item in limits], dtype=np.float64),
        upper=np.asarray([item[1] for item in limits], dtype=np.float64),
        minimum_frame_support=int(_get(preset, "minimum_frame_support", 2)),
        minimum_wicket_confidence=float(
            _get(preset, "minimum_wicket_confidence", 0.35)
        ),
        both_wickets_required=bool(_get(preset, "both_wickets_required", True)),
        distortion_policy=str(_get(preset, "distortion_policy", "ZERO")),
    )


def _compatibility(
    observation: WicketObservationResult, preset: _Preset
) -> dict[str, Any]:
    reasons: list[dict[str, str]] = []

    def reject(code: str, message: str) -> None:
        reasons.append({"reason_code": code, "severity": "ERROR", "message": message})

    if observation.setup_frame is None:
        reject("SETUP_FRAME_UNAVAILABLE", "A persisted setup frame is required.")
    if preset.both_wickets_required and (
        observation.near_wicket is None or observation.far_wicket is None
    ):
        reject("BOTH_WICKETS_REQUIRED", "The preset requires two independent wickets.")
    if len(observation.supporting_frames) < preset.minimum_frame_support:
        reject(
            "INSUFFICIENT_FRAME_SUPPORT",
            "Supporting-frame evidence is below the preset minimum.",
        )
    for role, wicket in (
        ("near", observation.near_wicket),
        ("far", observation.far_wicket),
    ):
        if wicket is None:
            continue
        if wicket.quality_score < preset.minimum_wicket_confidence:
            reject(
                "INVALID_WICKET_OBSERVATIONS",
                f"The {role} wicket confidence is below the preset minimum.",
            )
        if wicket.region.rejection_reason:
            reject(
                "INVALID_WICKET_OBSERVATIONS", f"The persisted {role} wicket was rejected."
            )
        if wicket.region.stability not in {"STABLE", "PARTIALLY_STABLE"}:
            reject(
                "INVALID_WICKET_OBSERVATIONS",
                f"The {role} wicket lacks temporal stability.",
            )
        if wicket.quality_factors.get("frame_edge_clipping", 1.0) < 0.7:
            reject(
                "SEVERE_WICKET_CLIPPING",
                f"The {role} wicket is severely clipped.",
            )
    if preset.distortion_policy.upper() not in {
        "ZERO",
        "ZERO_DISTORTION",
        "ASSUME_ZERO",
        "NONE",
        "ZERO_OR_PREUNDISTORTED",
    }:
        reject(
            "UNSUPPORTED_DISTORTION",
            "The preset distortion policy is unsupported in V1.",
        )
    setup = observation.setup_frame
    width = setup.image_width if setup else 1
    height = setup.image_height if setup else 1
    orientation = (
        "LANDSCAPE" if width > height else "PORTRAIT" if height > width else "SQUARE"
    )
    return {
        "status": "INCOMPATIBLE" if reasons else "COMPATIBLE",
        "native_video_width_px": width,
        "native_video_height_px": height,
        "detected_orientation": orientation,
        "long_edge_to_short_edge_aspect_ratio": max(width, height)
        / max(min(width, height), 1),
        "rotation_metadata_deg": None,
        "distortion_mode": "ZERO_DISTORTION",
        "camera_end": preset.camera_end,
        "both_wickets_present": observation.near_wicket is not None
        and observation.far_wicket is not None,
        "setup_frame_available": setup is not None,
        "supporting_frame_count": len(observation.supporting_frames),
        "wicket_observations_valid": observation.status
        == "READY_FOR_REGISTRATION_EXPERIMENT",
        "severe_clipping_detected": any(
            item["reason_code"] == "SEVERE_WICKET_CLIPPING" for item in reasons
        ),
        "nested_false_wicket_evidence_detected": False,
        "unsupported_crop_or_rotation_detected": False,
        "reasons": reasons,
    }


def check_preset_compatibility(preset: Any, evidence: Any) -> dict[str, Any]:
    """Check cheap input compatibility before loading or fitting camera evidence."""
    normalised = _normalise_preset(preset)
    width = int(_get(evidence, "image_width", 0))
    height = int(_get(evidence, "image_height", 0))
    reasons: list[dict[str, str]] = []

    def reject(code: str, message: str) -> None:
        reasons.append({"reason_code": code, "severity": "ERROR", "message": message})

    if width <= 0 or height <= 0:
        reject(
            "INVALID_NATIVE_DIMENSIONS",
            "Native video dimensions must be positive.",
        )
    else:
        expected = _get(preset, "expected_aspect_ratio_range", (1.0, 4.0))
        if isinstance(expected, Sequence):
            low, high = float(expected[0]), float(expected[1])
        else:
            low = float(_get(expected, "minimum_long_edge_to_short_edge_ratio"))
            high = float(_get(expected, "maximum_long_edge_to_short_edge_ratio"))
        aspect = max(width, height) / max(min(width, height), 1)
        if not low <= aspect <= high:
            reject(
                "ASPECT_RATIO_OUT_OF_RANGE",
                "Native aspect ratio is outside the preset range.",
            )
    if int(_get(evidence, "rotation_degrees", 0)) not in {0, 180}:
        reject(
            "UNSUPPORTED_ROTATION",
            "Rotation metadata requires an unsupported transform.",
        )
    if not bool(_get(evidence, "setup_frame_available", True)):
        reject("SETUP_FRAME_UNAVAILABLE", "A setup frame is required.")
    if (
        int(_get(evidence, "supporting_frame_count", 0))
        < normalised.minimum_frame_support
    ):
        reject(
            "INSUFFICIENT_FRAME_SUPPORT",
            "Supporting-frame evidence is below the preset minimum.",
        )
    if normalised.both_wickets_required and not (
        bool(_get(evidence, "near_wicket_available", False))
        and bool(_get(evidence, "far_wicket_available", False))
    ):
        reject("BOTH_WICKETS_REQUIRED", "Both wickets are required.")
    if (
        float(_get(evidence, "minimum_observed_confidence", 0.0))
        < normalised.minimum_wicket_confidence
    ):
        reject(
            "INVALID_WICKET_OBSERVATIONS",
            "Wicket confidence is below the preset minimum.",
        )
    if bool(_get(evidence, "severe_clipping", False)):
        reject("SEVERE_WICKET_CLIPPING", "Severe wicket clipping is incompatible.")
    if bool(_get(evidence, "nested_false_wicket_evidence", False)):
        reject(
            "NESTED_FALSE_WICKET_EVIDENCE",
            "Nested false-wicket evidence is unresolved.",
        )
    distortion = str(_get(evidence, "distortion_mode", "ZERO_DISTORTION")).upper()
    if distortion not in {"ZERO", "ZERO_DISTORTION", "NONE", "PREUNDISTORTED_FRAME"}:
        reject("UNSUPPORTED_DISTORTION", "The distortion mode is unsupported.")
    evidence_end = str(_get(evidence, "camera_end", "unknown")).lower()
    if evidence_end not in {"unknown", normalised.camera_end}:
        reject(
            "CAMERA_END_MISMATCH",
            "Camera-end evidence conflicts with the preset.",
        )
    orientation = (
        "LANDSCAPE" if width > height else "PORTRAIT" if height > width else "SQUARE"
    )
    return {
        "status": "INCOMPATIBLE" if reasons else "COMPATIBLE",
        "native_video_width_px": max(width, 1),
        "native_video_height_px": max(height, 1),
        "detected_orientation": orientation,
        "long_edge_to_short_edge_aspect_ratio": (
            max(width, height) / max(min(width, height), 1)
            if width > 0 and height > 0
            else 1.0
        ),
        "rotation_metadata_deg": int(_get(evidence, "rotation_degrees", 0)),
        "distortion_mode": distortion,
        "camera_end": None if evidence_end == "unknown" else evidence_end,
        "both_wickets_present": bool(_get(evidence, "near_wicket_available", False))
        and bool(_get(evidence, "far_wicket_available", False)),
        "setup_frame_available": bool(_get(evidence, "setup_frame_available", True)),
        "supporting_frame_count": int(_get(evidence, "supporting_frame_count", 0)),
        "wicket_observations_valid": not reasons,
        "severe_clipping_detected": bool(_get(evidence, "severe_clipping", False)),
        "nested_false_wicket_evidence_detected": bool(
            _get(evidence, "nested_false_wicket_evidence", False)
        ),
        "unsupported_crop_or_rotation_detected": any(
            item["reason_code"] == "unsupported_rotation_metadata" for item in reasons
        ),
        "reasons": reasons,
    }


evaluate_preset_compatibility = check_preset_compatibility


def _select_frame_boxes(
    observation: WicketObservationResult,
) -> dict[tuple[int, str], PixelBox]:
    selected: dict[tuple[int, str], PixelBox] = {}
    support = {item.frame_index for item in observation.supporting_frames}
    for role, wicket in (
        ("near", observation.near_wicket),
        ("far", observation.far_wicket),
    ):
        if wicket is None:
            continue
        by_frame: dict[int, list[RawWicketDetection]] = {}
        for detection in observation.diagnostics.raw_detections:
            if detection.frame_index in support:
                by_frame.setdefault(detection.frame_index, []).append(detection)
        for frame_index, detections in by_frame.items():
            ranked = sorted(
                detections,
                key=lambda item: (
                    -_box_iou(wicket.region.bbox, item.bbox),
                    -item.confidence,
                    item.source,
                ),
            )
            if ranked and _box_iou(wicket.region.bbox, ranked[0].bbox) >= 0.05:
                selected[(frame_index, role)] = ranked[0].bbox
    return selected


def _evidence(observation: WicketObservationResult, preset: _Preset) -> _Evidence:
    assert observation.setup_frame is not None
    assignment = "A" if preset.camera_end == "bowler" else "B"
    correspondences = build_registration_correspondences(
        observation,
        assignment_hypothesis=assignment,
        lateral_mapping=preset.image_left_mapping,
    )
    return _Evidence(
        observation=observation,
        assignment=assignment,
        lateral_mapping=preset.image_left_mapping,
        correspondences=correspondences,
        frame_boxes=_select_frame_boxes(observation),
        image_width=observation.setup_frame.image_width,
        image_height=observation.setup_frame.image_height,
    )


def _pose(parameters: np.ndarray, preset: _Preset, pitch_length_m: float):
    del pitch_length_m  # Geometry ownership remains in the central conversion utility.
    camera = build_opencv_camera_from_preset_parameters(
        PresetCameraParameters(
            lateral_offset_m=float(parameters[0]),
            distance_behind_wicket_m=float(parameters[1]),
            camera_height_m=float(parameters[2]),
            yaw_deg=float(parameters[3]),
            pitch_deg=float(parameters[4]),
            roll_deg=float(parameters[5]),
            horizontal_fov_deg=float(parameters[6]),
        ),
        image_width=2,
        image_height=2,
        camera_end=preset.camera_end,
        image_left_mapping=preset.image_left_mapping,
    )
    focal_ratio = float(camera.camera_matrix[0, 0] / 2.0)
    return (
        camera.rotation_vector.reshape(3, 1),
        camera.translation_vector.reshape(3, 1),
        camera.rotation_matrix,
        camera.camera_position_world,
        focal_ratio,
    )


def _camera_arrays(parameters: np.ndarray, preset: _Preset, evidence: _Evidence):
    camera = build_opencv_camera_from_preset_parameters(
        PresetCameraParameters(
            lateral_offset_m=float(parameters[0]),
            distance_behind_wicket_m=float(parameters[1]),
            camera_height_m=float(parameters[2]),
            yaw_deg=float(parameters[3]),
            pitch_deg=float(parameters[4]),
            roll_deg=float(parameters[5]),
            horizontal_fov_deg=float(parameters[6]),
        ),
        image_width=evidence.image_width,
        image_height=evidence.image_height,
        camera_end=preset.camera_end,
        image_left_mapping=preset.image_left_mapping,
    )
    return (
        camera.rotation_vector.reshape(3, 1),
        camera.translation_vector.reshape(3, 1),
        camera.rotation_matrix,
        camera.camera_position_world,
        float(camera.camera_matrix[0, 0]),
        camera.camera_matrix,
    )


def _box_residuals(
    projected: PixelBox | None,
    observed: PixelBox,
    weight: float,
    *,
    uncertainty_scale_ratio: float = 0.08,
) -> list[float]:
    if projected is None:
        return [25.0 * weight] * 4
    scale = max(
        4.0,
        math.hypot(observed.width, observed.height) * uncertainty_scale_ratio,
    )
    return [
        ((projected.x + projected.width / 2) - (observed.x + observed.width / 2))
        / scale
        * weight,
        ((projected.y + projected.height / 2) - (observed.y + observed.height / 2))
        / scale
        * weight,
        (projected.width - observed.width) / scale * weight,
        (projected.height - observed.height) / scale * weight,
    ]


def _wicket_for_role(evidence: _Evidence, role: str) -> Any:
    return (
        evidence.observation.near_wicket
        if role == "near"
        else evidence.observation.far_wicket
    )


def _objective_component_vectors(
    parameters: np.ndarray,
    preset: _Preset,
    evidence: _Evidence,
    profile: _ObjectiveProfile = _NORMAL_OBJECTIVE,
) -> dict[str, np.ndarray]:
    rotation, translation, _, _, _, camera_matrix = _camera_arrays(
        parameters, preset, evidence
    )
    components: dict[str, list[float]] = {
        "exact_points": [],
        "coarse_points": [],
        "lines": [],
        "setup_envelopes": [],
        "temporal_envelopes": [],
        "preset_priors": [],
    }
    coarse_counts = {
        role: max(
            1,
            sum(
                item.status == "USED"
                and item.mapping_type == "COARSE_POINTLIKE"
                and item.correspondence_id.startswith(f"{role}:")
                for item in evidence.correspondences
            ),
        )
        for role in ("near", "far")
    }
    line_counts = {
        role: max(
            1,
            sum(
                item.status == "SOFT_ONLY"
                and item.mapping_type in {"TOP_LINE", "BASE_LINE", "OUTER_AXIS"}
                and item.correspondence_id.startswith(f"{role}:")
                for item in evidence.correspondences
            ),
        )
        for role in ("near", "far")
    }
    for item in evidence.correspondences:
        if (
            item.status == "USED"
            and item.observed_pixel is not None
            and item.world_point is not None
        ):
            world = np.asarray(
                [[item.world_point.x, item.world_point.y, item.world_point.z]]
            )
            projected, depths = _project(world, rotation, translation, camera_matrix)
            if depths[0] <= 0:
                values = [25.0, 25.0]
            else:
                uncertainty = max(item.uncertainty_px, 1.0)
                weight = math.sqrt(max(item.registration_weight, 1e-6))
                role = item.correspondence_id.split(":", 1)[0]
                if item.mapping_type == "COARSE_POINTLIKE":
                    wicket = _wicket_for_role(evidence, role)
                    if wicket is not None:
                        semantic_floor = (
                            max(wicket.region.bbox.width, wicket.region.bbox.height)
                            * profile.coarse_uncertainty_floor_ratio
                        )
                        uncertainty = max(uncertainty, semantic_floor)
                    if profile.correlated_evidence_normalization:
                        weight /= math.sqrt(coarse_counts[role])
                values = (
                    (projected[0] - [item.observed_pixel.x, item.observed_pixel.y])
                    / uncertainty
                    * weight
                ).tolist()
            key = (
                "coarse_points"
                if item.mapping_type == "COARSE_POINTLIKE"
                else "exact_points"
            )
            scale = (
                profile.coarse_point_weight
                if key == "coarse_points"
                else profile.point_weight
            )
            components[key].extend((np.asarray(values) * scale).tolist())
        elif item.status == "SOFT_ONLY" and item.mapping_type in {
            "TOP_LINE",
            "BASE_LINE",
            "OUTER_AXIS",
        }:
            values = _line_residuals(item, rotation, translation, camera_matrix)
            role = item.correspondence_id.split(":", 1)[0]
            if profile.correlated_evidence_normalization:
                wicket = _wicket_for_role(evidence, role)
                floor = 1.0
                if wicket is not None:
                    floor = max(
                        floor,
                        max(wicket.region.bbox.width, wicket.region.bbox.height)
                        * profile.coarse_uncertainty_floor_ratio,
                    )
                original_uncertainty = max(item.uncertainty_px, 1.0)
                values = [
                    value
                    * original_uncertainty
                    / floor
                    / math.sqrt(line_counts[role])
                    for value in values
                ]
            components["lines"].extend(
                (np.asarray(values) * profile.line_weight).tolist()
            )
    ends = {
        "near": "bowler" if evidence.assignment == "A" else "striker",
        "far": "striker" if evidence.assignment == "A" else "bowler",
    }
    for role, wicket in (
        ("near", evidence.observation.near_wicket),
        ("far", evidence.observation.far_wicket),
    ):
        if wicket is None:
            continue
        projected = _projected_wicket_box(
            ends[role], rotation, translation, camera_matrix
        )
        components["setup_envelopes"].extend(
            _box_residuals(
                projected,
                wicket.region.bbox,
                0.7
                * math.sqrt(max(wicket.quality_score, 0.05))
                * profile.envelope_weight,
                uncertainty_scale_ratio=(
                    0.35 if profile.correlated_evidence_normalization else 0.08
                ),
            )
        )
        role_frame_count = max(
            1,
            sum(frame_role == role for _, frame_role in evidence.frame_boxes),
        )
        for (frame_index, frame_role), observed in evidence.frame_boxes.items():
            if frame_role == role:
                correlation_scale = (
                    math.sqrt(role_frame_count)
                    if profile.correlated_evidence_normalization
                    else 1.0
                )
                components["temporal_envelopes"].extend(
                    _box_residuals(
                        projected,
                        observed,
                        0.18 * profile.temporal_weight / correlation_scale,
                        uncertainty_scale_ratio=(
                            0.35
                            if profile.correlated_evidence_normalization
                            else 0.08
                        ),
                    )
                )
    half_ranges = np.maximum((preset.upper - preset.lower) / 2.0, 1e-6)
    components["preset_priors"].extend(
        (
            (parameters - preset.nominal)
            / half_ranges
            * 0.22
            * profile.prior_weight
        ).tolist()
    )
    return {
        name: np.asarray(values, dtype=np.float64)
        for name, values in components.items()
    }


def _objective(
    parameters: np.ndarray,
    preset: _Preset,
    evidence: _Evidence,
    profile: _ObjectiveProfile = _NORMAL_OBJECTIVE,
) -> np.ndarray:
    components = _objective_component_vectors(parameters, preset, evidence, profile)
    return np.concatenate(list(components.values()))


def _objective_diagnostics(
    parameters: np.ndarray,
    preset: _Preset,
    evidence: _Evidence,
    profile: _ObjectiveProfile = _NORMAL_OBJECTIVE,
) -> dict[str, Any]:
    components = _objective_component_vectors(parameters, preset, evidence, profile)
    result = {
        name: {
            "residual_count": int(values.size),
            "squared_loss": float(np.sum(values**2) / 2.0),
            "rms_residual": (
                float(np.sqrt(np.mean(values**2))) if values.size else None
            ),
            "maximum_absolute_residual": (
                float(np.max(np.abs(values))) if values.size else None
            ),
        }
        for name, values in components.items()
    }
    data_names = (
        "exact_points",
        "coarse_points",
        "lines",
        "setup_envelopes",
        "temporal_envelopes",
    )
    result["raw_data_loss"] = sum(result[name]["squared_loss"] for name in data_names)
    result["prior_loss"] = result["preset_priors"]["squared_loss"]
    result["physical_loss"] = 0.0
    result["final_weighted_loss"] = result["raw_data_loss"] + result["prior_loss"]
    result["profile"] = {
        name: getattr(profile, name)
        for name in profile.__dataclass_fields__
    }
    result["semantic_evidence"] = _semantic_objective_diagnostics(
        parameters, preset, evidence, profile
    )
    return result


def _semantic_objective_diagnostics(
    parameters: np.ndarray,
    preset: _Preset,
    evidence: _Evidence,
    profile: _ObjectiveProfile,
) -> dict[str, Any]:
    """Describe physical evidence without changing the residual vector."""
    rotation, translation, _, _, _, camera_matrix = _camera_arrays(
        parameters, preset, evidence
    )
    points: list[dict[str, Any]] = []
    lines: list[dict[str, Any]] = []
    for item in evidence.correspondences:
        if (
            item.status == "USED"
            and item.observed_pixel is not None
            and item.world_point is not None
        ):
            world = np.asarray(
                [[item.world_point.x, item.world_point.y, item.world_point.z]]
            )
            projected, depths = _project(world, rotation, translation, camera_matrix)
            observed = np.asarray([item.observed_pixel.x, item.observed_pixel.y])
            points.append(
                {
                    "correspondence_id": item.correspondence_id,
                    "mapping_type": item.mapping_type,
                    "projected_pixel": projected[0].tolist(),
                    "observed_pixel": observed.tolist(),
                    "pixel_residual": (
                        float(np.linalg.norm(projected[0] - observed))
                        if depths[0] > 0
                        else None
                    ),
                    "positive_depth": bool(depths[0] > 0),
                    "uncertainty_px": float(item.uncertainty_px),
                    "registration_weight": float(item.registration_weight),
                }
            )
        elif item.status == "SOFT_ONLY" and item.mapping_type in {
            "TOP_LINE",
            "BASE_LINE",
            "OUTER_AXIS",
        }:
            residuals = _line_residuals(item, rotation, translation, camera_matrix)
            lines.append(
                {
                    "correspondence_id": item.correspondence_id,
                    "mapping_type": item.mapping_type,
                    "normalised_residuals": [float(value) for value in residuals],
                    "uncertainty_px": float(item.uncertainty_px),
                    "registration_weight": float(item.registration_weight),
                }
            )

    ends = {
        "near": "bowler" if evidence.assignment == "A" else "striker",
        "far": "striker" if evidence.assignment == "A" else "bowler",
    }
    envelopes: dict[str, Any] = {}
    for role, wicket in (
        ("near", evidence.observation.near_wicket),
        ("far", evidence.observation.far_wicket),
    ):
        if wicket is None:
            envelopes[role] = None
            continue
        projected = _projected_wicket_box(
            ends[role], rotation, translation, camera_matrix
        )
        observed = wicket.region.bbox
        envelopes[role] = {
            "centre_residual_px": (
                math.hypot(
                    projected.x + projected.width / 2
                    - observed.x
                    - observed.width / 2,
                    projected.y + projected.height / 2
                    - observed.y
                    - observed.height / 2,
                )
                if projected is not None
                else None
            ),
            "width_residual_px": (
                float(projected.width - observed.width)
                if projected is not None
                else None
            ),
            "height_residual_px": (
                float(projected.height - observed.height)
                if projected is not None
                else None
            ),
            "intersection_over_union": (
                _box_iou(projected, observed) if projected is not None else None
            ),
            "quality_score": float(wicket.quality_score),
        }

    half_ranges = np.maximum((preset.upper - preset.lower) / 2.0, 1e-6)
    prior_residuals = (
        (parameters - preset.nominal)
        / half_ranges
        * 0.22
        * profile.prior_weight
    )
    return {
        "points": points,
        "lines": lines,
        "setup_envelopes": envelopes,
        "temporal": _temporal_metrics(parameters, preset, evidence),
        "preset_priors": {
            name: {
                "value": float(parameters[index]),
                "nominal": float(preset.nominal[index]),
                "normalised_residual": float(prior_residuals[index]),
                "squared_loss": float(prior_residuals[index] ** 2 / 2.0),
            }
            for index, name in enumerate(_PARAMETER_NAMES)
        },
        "physical_checks": _physical_checks(parameters, preset, evidence),
    }


def _normalised_objective(
    normalised_parameters: np.ndarray,
    preset: _Preset,
    evidence: _Evidence,
    profile: _ObjectiveProfile = _NORMAL_OBJECTIVE,
) -> np.ndarray:
    parameters = denormalize_parameters(
        normalised_parameters, preset.lower, preset.upper
    )
    return _objective(parameters, preset, evidence, profile)


def _anchor_metrics(parameters: np.ndarray, preset: _Preset, evidence: _Evidence):
    rotation, translation, _, _, _, camera_matrix = _camera_arrays(
        parameters, preset, evidence
    )
    errors: list[float] = []
    for item in evidence.correspondences:
        if (
            item.status != "USED"
            or item.observed_pixel is None
            or item.world_point is None
        ):
            continue
        world = np.asarray(
            [[item.world_point.x, item.world_point.y, item.world_point.z]]
        )
        projected, depths = _project(world, rotation, translation, camera_matrix)
        if depths[0] > 0:
            errors.append(
                float(
                    np.linalg.norm(
                        projected[0] - [item.observed_pixel.x, item.observed_pixel.y]
                    )
                )
            )
    if not errors:
        return None, None
    return float(np.sqrt(np.mean(np.square(errors)))), float(np.median(errors))


def _temporal_metrics(
    parameters: np.ndarray, preset: _Preset, evidence: _Evidence
) -> dict[str, Any]:
    rotation, translation, _, _, _, camera_matrix = _camera_arrays(
        parameters, preset, evidence
    )
    ends = {
        "near": "bowler" if evidence.assignment == "A" else "striker",
        "far": "striker" if evidence.assignment == "A" else "bowler",
    }
    scores: dict[str, list[float]] = {"near": [], "far": []}
    frame_scores: dict[int, list[float]] = {}
    centre_residuals: list[float] = []
    width_residuals: list[float] = []
    height_residuals: list[float] = []
    for role, end in ends.items():
        projected = _projected_wicket_box(end, rotation, translation, camera_matrix)
        if projected is None:
            continue
        for (frame_index, frame_role), observed in evidence.frame_boxes.items():
            if frame_role != role:
                continue
            iou = _box_iou(projected, observed)
            scores[role].append(iou)
            frame_scores.setdefault(frame_index, []).append(iou)
            centre_residuals.append(
                math.hypot(
                    (projected.x + projected.width / 2)
                    - (observed.x + observed.width / 2),
                    (projected.y + projected.height / 2)
                    - (observed.y + observed.height / 2),
                )
            )
            width_residuals.append(abs(projected.width - observed.width))
            height_residuals.append(abs(projected.height - observed.height))
    medians = {
        role: (float(np.median(values)) if values else None)
        for role, values in scores.items()
    }
    per_frame = {
        frame: float(np.mean(values)) for frame, values in frame_scores.items()
    }
    stability = float(np.median(list(per_frame.values()))) if per_frame else 0.0
    worst = (
        min(per_frame, key=lambda frame: (per_frame[frame], frame))
        if per_frame
        else None
    )
    all_heights = height_residuals
    return {
        "frame_count": len(evidence.observation.supporting_frames),
        "successful_frame_count": len(per_frame),
        "evaluated_frame_ids": sorted(per_frame),
        "median_near_iou": medians["near"],
        "median_far_iou": medians["far"],
        "median_centre_residual_px": (
            float(np.median(centre_residuals)) if centre_residuals else None
        ),
        "median_width_residual_px": (
            float(np.median(width_residuals)) if width_residuals else None
        ),
        "median_height_residual_px": (
            float(np.median(height_residuals)) if height_residuals else None
        ),
        "scale_consistency": (
            max(
                0.0,
                1.0
                - float(np.median(all_heights)) / max(evidence.image_height * 0.1, 1.0),
            )
            if all_heights
            else 0.0
        ),
        "worst_supporting_frame": worst,
        "temporal_stability_score": stability
        * min(1.0, len(per_frame) / max(preset.minimum_frame_support, 1)),
    }


def _physical_checks(
    parameters: np.ndarray, preset: _Preset, evidence: _Evidence
) -> list[dict[str, Any]]:
    rotation, translation, matrix, position, focal, camera_matrix = _camera_arrays(
        parameters, preset, evidence
    )
    specification = build_virtual_pitch_specification()
    world = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, specification.dimensions.pitch_length_m, 0.0]]
    )
    _, depths = _project(world, rotation, translation, camera_matrix)
    near_end = "bowler" if evidence.assignment == "A" else "striker"
    far_end = "striker" if evidence.assignment == "A" else "bowler"
    near_box = _projected_wicket_box(near_end, rotation, translation, camera_matrix)
    far_box = _projected_wicket_box(far_end, rotation, translation, camera_matrix)
    pitch_centre = np.asarray([0.0, specification.dimensions.pitch_length_m / 2.0, 0.0])
    forward = matrix.T @ np.asarray([0.0, 0.0, 1.0])
    direction = pitch_centre - position
    direction /= max(np.linalg.norm(direction), 1e-9)
    margin = np.maximum((preset.upper - preset.lower) * 1e-4, 1e-6)
    critical_indices = (1, 2, 6)  # distance, height, and focal/FOV only
    critical_bound = any(
        parameters[index] <= preset.lower[index] + margin[index]
        or parameters[index] >= preset.upper[index] - margin[index]
        for index in critical_indices
    )
    fov = float(parameters[6])
    checks = [
        (
            "finite_pose",
            bool(np.isfinite(matrix).all() and np.isfinite(position).all()),
            None,
        ),
        ("camera_above_pitch", float(position[2]) > 0.0, float(position[2])),
        (
            "preset_bounds",
            bool(
                np.all(parameters >= preset.lower)
                and np.all(parameters <= preset.upper)
            ),
            None,
        ),
        (
            "camera_height_bounds",
            bool(preset.lower[2] <= parameters[2] <= preset.upper[2]),
            float(parameters[2]),
        ),
        (
            "distance_bounds",
            bool(preset.lower[1] <= parameters[1] <= preset.upper[1]),
            float(parameters[1]),
        ),
        (
            "lateral_bounds",
            bool(preset.lower[0] <= parameters[0] <= preset.upper[0]),
            float(parameters[0]),
        ),
        (
            "yaw_bounds",
            bool(preset.lower[3] <= parameters[3] <= preset.upper[3]),
            float(parameters[3]),
        ),
        (
            "pitch_bounds",
            bool(preset.lower[4] <= parameters[4] <= preset.upper[4]),
            float(parameters[4]),
        ),
        (
            "roll_bounds",
            bool(preset.lower[5] <= parameters[5] <= preset.upper[5]),
            float(parameters[5]),
        ),
        (
            "fov_bounds",
            bool(preset.lower[6] <= parameters[6] <= preset.upper[6]),
            float(parameters[6]),
        ),
        ("wickets_in_front", bool(np.all(depths > 0.05)), float(np.min(depths))),
        (
            "camera_faces_pitch",
            float(forward @ direction) > 0.25,
            float(forward @ direction),
        ),
        (
            "near_far_perspective_order",
            bool(near_box and far_box and near_box.height >= far_box.height * 0.85),
            (
                None
                if not near_box or not far_box
                else near_box.height / max(far_box.height, 1e-9)
            ),
        ),
        (
            "projected_wickets_scene_sized",
            bool(
                near_box
                and far_box
                and near_box.width < evidence.image_width * 0.8
                and near_box.height < evidence.image_height * 0.8
                and far_box.width < evidence.image_width * 0.8
                and far_box.height < evidence.image_height * 0.8
            ),
            None,
        ),
        ("focal_not_extreme", 5.0 < fov < 150.0 and focal > 0.0, fov),
        ("critical_bounds_clear", not critical_bound, critical_bound),
    ]
    return [
        {"check_id": name, "passed": passed, "value": value}
        for name, passed, value in checks
    ]


def _active_bounds(parameters: np.ndarray, preset: _Preset) -> list[str]:
    tolerance = np.maximum((preset.upper - preset.lower) * 1e-4, 1e-6)
    reached: list[str] = []
    for index, name in enumerate(_PARAMETER_NAMES):
        if parameters[index] <= preset.lower[index] + tolerance[index]:
            reached.append(f"{name}:lower")
        elif parameters[index] >= preset.upper[index] - tolerance[index]:
            reached.append(f"{name}:upper")
    return reached


def _score_fit(
    rmse: float | None,
    near_iou: float | None,
    far_iou: float | None,
    temporal: dict[str, Any],
    checks: list[dict[str, Any]],
    active_bounds: list[str],
) -> float:
    anchor = math.exp(-float(rmse) / 10.0) if rmse is not None else 0.0
    envelopes = (
        float(np.mean([value for value in (near_iou, far_iou) if value is not None]))
        if near_iou is not None or far_iou is not None
        else 0.0
    )
    physical = sum(item["passed"] for item in checks) / max(len(checks), 1)
    score = (
        0.38 * anchor
        + 0.28 * envelopes
        + 0.22 * temporal["temporal_stability_score"]
        + 0.12 * physical
    )
    score -= min(0.15, len(active_bounds) * 0.05)
    return max(0.0, min(1.0, score))


def _fit_candidate(
    candidate_id: str,
    source: str,
    initial: np.ndarray,
    preset: _Preset,
    evidence: _Evidence,
    profile: _ObjectiveProfile = _NORMAL_OBJECTIVE,
) -> _Fit:
    initial = np.clip(initial, preset.lower + 1e-8, preset.upper - 1e-8)
    initial_normalised = normalize_parameters(initial, preset.lower, preset.upper)
    initial_residual = _objective(initial, preset, evidence, profile)
    result = least_squares(
        _normalised_objective,
        initial_normalised,
        args=(preset, evidence, profile),
        bounds=(-np.ones_like(initial_normalised), np.ones_like(initial_normalised)),
        loss="soft_l1",
        f_scale=1.0,
        x_scale=np.ones_like(initial_normalised),
        max_nfev=160,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )
    fitted = denormalize_parameters(
        result.x.astype(np.float64), preset.lower, preset.upper
    )
    rmse, median = _anchor_metrics(fitted, preset, evidence)
    rotation, translation, _, _, _, camera_matrix = _camera_arrays(
        fitted, preset, evidence
    )
    near_end = "bowler" if evidence.assignment == "A" else "striker"
    far_end = "striker" if evidence.assignment == "A" else "bowler"
    near_box = _projected_wicket_box(near_end, rotation, translation, camera_matrix)
    far_box = _projected_wicket_box(far_end, rotation, translation, camera_matrix)
    near_iou = (
        _box_iou(near_box, evidence.observation.near_wicket.region.bbox)
        if near_box and evidence.observation.near_wicket
        else None
    )
    far_iou = (
        _box_iou(far_box, evidence.observation.far_wicket.region.bbox)
        if far_box and evidence.observation.far_wicket
        else None
    )
    temporal = _temporal_metrics(fitted, preset, evidence)
    checks = _physical_checks(fitted, preset, evidence)
    active = _active_bounds(fitted, preset)
    rejected = [item["check_id"] for item in checks if not item["passed"]]
    score = _score_fit(rmse, near_iou, far_iou, temporal, checks, active)
    return _Fit(
        candidate_id=candidate_id,
        seed_source=source,
        initial=initial,
        parameters=fitted,
        converged=bool(result.success),
        iterations=int(result.nfev),
        initial_cost=float(np.sum(initial_residual**2) / 2.0),
        final_cost=float(result.cost),
        active_bounds=active,
        anchor_rmse_px=rmse,
        median_anchor_error_px=median,
        near_iou=near_iou,
        far_iou=far_iou,
        temporal=temporal,
        physical_checks=checks,
        score=score,
        eligible=bool(result.success and not rejected),
        rejection_reasons=rejected
        or ([] if result.success else ["optimisation_failed"]),
        objective_components=_objective_diagnostics(fitted, preset, evidence, profile),
    )


def diagnose_objective_components(
    *,
    parameters: Sequence[float],
    preset: Any,
    observation: WicketObservationResult,
    legacy: bool = False,
) -> dict[str, Any]:
    """Expose every objective family without changing or persisting a fit."""
    normalised = _normalise_preset(preset)
    evidence = _evidence(observation, normalised)
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (len(_PARAMETER_NAMES),) or not np.isfinite(values).all():
        raise ValueError("Expected seven finite preset parameters.")
    return _objective_diagnostics(
        values,
        normalised,
        evidence,
        _LEGACY_OBJECTIVE if legacy else _NORMAL_OBJECTIVE,
    )


def replay_physical_eligibility(
    *,
    parameters: Sequence[float],
    preset: Any,
    observation: WicketObservationResult,
) -> list[dict[str, Any]]:
    """Replay deterministic geometric gates for diagnostic initial candidates."""
    normalised = _normalise_preset(preset)
    evidence = _evidence(observation, normalised)
    values = np.asarray(parameters, dtype=np.float64)
    checks = _physical_checks(values, normalised, evidence)
    thresholds = {
        "camera_above_pitch": "> 0 m",
        "camera_height_bounds": f"[{normalised.lower[2]}, {normalised.upper[2]}] m",
        "distance_bounds": f"[{normalised.lower[1]}, {normalised.upper[1]}] m",
        "lateral_bounds": f"[{normalised.lower[0]}, {normalised.upper[0]}] m",
        "yaw_bounds": f"[{normalised.lower[3]}, {normalised.upper[3]}] deg",
        "pitch_bounds": f"[{normalised.lower[4]}, {normalised.upper[4]}] deg",
        "roll_bounds": f"[{normalised.lower[5]}, {normalised.upper[5]}] deg",
        "fov_bounds": f"[{normalised.lower[6]}, {normalised.upper[6]}] deg",
        "wickets_in_front": "> 0.05 m minimum depth",
        "camera_faces_pitch": "> 0.25 forward dot product",
        "near_far_perspective_order": ">= 0.85 near/far height ratio",
        "projected_wickets_scene_sized": "each width/height < 80% of frame",
        "focal_not_extreme": "5 < HFOV < 150 deg and focal > 0",
        "critical_bounds_clear": "distance, height, and HFOV not at a bound",
    }
    return [
        {
            **check,
            "threshold": thresholds.get(check["check_id"]),
            "severity": "ERROR" if not check["passed"] else "INFO",
            "reason": (
                f"{check['check_id']} passed."
                if check["passed"]
                else f"{check['check_id']} failed."
            ),
        }
        for check in checks
    ]


def run_objective_ablation_matrix(
    *,
    preset: Any,
    observation: WicketObservationResult,
    initial_candidates: Mapping[str, Sequence[float]] | None = None,
) -> list[dict[str, Any]]:
    """Run deterministic diagnostic fits; this never persists or accepts a camera."""
    normalised = _normalise_preset(preset)
    evidence = _evidence(observation, normalised)
    profiles = {
        "A_POINT_EVIDENCE_ONLY": _ObjectiveProfile(
            line_weight=0.0,
            envelope_weight=0.0,
            temporal_weight=0.0,
            prior_weight=0.0,
        ),
        "B_COARSE_POINT_ONLY": _ObjectiveProfile(
            point_weight=0.0,
            line_weight=0.0,
            envelope_weight=0.0,
            temporal_weight=0.0,
            prior_weight=0.0,
        ),
        "C_ENVELOPE_ONLY": _ObjectiveProfile(
            point_weight=0.0,
            coarse_point_weight=0.0,
            line_weight=0.0,
            temporal_weight=0.0,
            prior_weight=0.0,
        ),
        "D_POINT_PLUS_ENVELOPE": _ObjectiveProfile(
            line_weight=0.0,
            temporal_weight=0.0,
            prior_weight=0.0,
        ),
        "E_POINT_ENVELOPE_TEMPORAL": _ObjectiveProfile(
            line_weight=0.0,
            prior_weight=0.0,
        ),
        "F_DATA_WITHOUT_PRIORS": _ObjectiveProfile(prior_weight=0.0),
        "G_NORMAL_PRIORS": _NORMAL_OBJECTIVE,
        "H_STRONGER_PRIORS": _ObjectiveProfile(prior_weight=4.0),
    }
    seeds: dict[str, np.ndarray] = {"nominal": normalised.nominal.copy()}
    for name, values in (initial_candidates or {}).items():
        array = np.asarray(values, dtype=np.float64)
        if array.shape != normalised.nominal.shape or not np.isfinite(array).all():
            raise ValueError(f"Initial candidate {name!r} must contain seven values.")
        seeds[str(name)] = array
    matrix: list[dict[str, Any]] = []
    for ablation_id, profile in profiles.items():
        fit = _fit_candidate(
            ablation_id,
            "diagnostic_ablation",
            seeds["nominal"],
            normalised,
            evidence,
            profile,
        )
        matrix.append(_ablation_payload(ablation_id, "nominal", fit))
    for seed_name in sorted(name for name in seeds if name != "nominal"):
        fit = _fit_candidate(
            f"NORMAL_{seed_name}",
            "diagnostic_initial_candidate",
            seeds[seed_name],
            normalised,
            evidence,
            _NORMAL_OBJECTIVE,
        )
        matrix.append(_ablation_payload(f"NORMAL_{seed_name}", seed_name, fit))
    return matrix


def _ablation_payload(
    ablation_id: str, seed_name: str, fit: _Fit
) -> dict[str, Any]:
    temporal_score = float(fit.temporal["temporal_stability_score"])
    if fit.eligible and fit.score >= 0.48 and temporal_score >= 0.25:
        status = "VISUAL_OVERLAY_READY"
    else:
        status = "NEEDS_ASSISTANCE"
    return {
        "ablation_id": ablation_id,
        "seed_name": seed_name,
        "initial_parameters": _parameters_payload(fit.initial),
        "final_parameters": _parameters_payload(fit.parameters),
        "height_trajectory_m": [float(fit.initial[2]), float(fit.parameters[2])],
        "objective_components": fit.objective_components,
        "initial_cost": fit.initial_cost,
        "final_cost": fit.final_cost,
        "active_bounds": fit.active_bounds,
        "physical_checks": fit.physical_checks,
        "eligible": fit.eligible,
        "registration_classification": (
            "GROUND_PLANE_CANDIDATE" if fit.score >= 0.55 else "VISUAL_ONLY"
        ),
        "automation_status": status,
        "score": fit.score,
        "anchor_rmse_px": fit.anchor_rmse_px,
        "temporal_stability_score": temporal_score,
    }


def _parameters_from_candidate(
    candidate: Any,
    preset: _Preset,
    pitch_length_m: float,
    *,
    image_width: int | None = None,
    image_height: int | None = None,
) -> np.ndarray | None:
    del pitch_length_m
    if (
        not candidate
        or not candidate.rotation_matrix
        or not candidate.camera_world_position
        or not candidate.translation_vector
        or not candidate.intrinsics
    ):
        return None
    intrinsics = candidate.intrinsics
    width = int(image_width or round(float(intrinsics.principal_point_x_px) * 2.0))
    height = int(image_height or round(float(intrinsics.principal_point_y_px) * 2.0))
    if width <= 0 or height <= 0:
        return None
    camera_matrix = np.asarray(
        [
            [
                float(intrinsics.focal_length_x_px),
                0.0,
                float(intrinsics.principal_point_x_px),
            ],
            [
                0.0,
                float(intrinsics.focal_length_y_px),
                float(intrinsics.principal_point_y_px),
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    decomposition = decompose_opencv_camera_to_preset_parameters(
        camera_matrix=camera_matrix,
        rotation_matrix=candidate.rotation_matrix,
        translation_vector=candidate.translation_vector,
        image_width=width,
        image_height=height,
        camera_end=preset.camera_end,
        image_left_mapping=preset.image_left_mapping,
    )
    values = np.asarray(
        [
            decomposition.parameters.lateral_offset_m,
            decomposition.parameters.distance_behind_wicket_m,
            decomposition.parameters.camera_height_m,
            decomposition.parameters.yaw_deg,
            decomposition.parameters.pitch_deg,
            decomposition.parameters.roll_deg,
            decomposition.parameters.horizontal_fov_deg,
        ],
        dtype=np.float64,
    )
    return values if np.isfinite(values).all() else None


def _candidate_seeds(
    analysis_id: str, preset: _Preset
) -> list[tuple[str, str, np.ndarray]]:
    seeds: list[tuple[str, str, np.ndarray]] = [
        ("nominal", "preset_nominal", preset.nominal.copy())
    ]
    spans = preset.upper - preset.lower
    for index, name in enumerate(_PARAMETER_NAMES):
        for sign, label in ((-1.0, "low"), (1.0, "high")):
            value = preset.nominal.copy()
            value[index] += sign * spans[index] * 0.22
            seeds.append((f"preset_{name}_{label}", "preset_perturbation", value))
    try:
        registration = load_real_pitch_registration(analysis_id)
    except VideoAnalysisServiceError:
        registration = None
    if registration is not None:
        pitch_length = build_virtual_pitch_specification().dimensions.pitch_length_m
        for candidate in (
            registration.selected_candidate,
            registration.competing_candidate,
        ):
            values = _parameters_from_candidate(
                candidate,
                preset,
                pitch_length,
                image_width=(
                    registration.setup_frame.image_width
                    if registration.setup_frame is not None
                    else None
                ),
                image_height=(
                    registration.setup_frame.image_height
                    if registration.setup_frame is not None
                    else None
                ),
            )
            if values is not None:
                seeds.append(
                    (
                        f"persisted_{candidate.candidate_id}",
                        "persisted_registration",
                        values,
                    )
                )
    unique: list[tuple[str, str, np.ndarray]] = []
    seen: set[tuple[float, ...]] = set()
    for item in seeds:
        clipped = np.clip(item[2], preset.lower, preset.upper)
        key = tuple(np.round(clipped, 8))
        if key not in seen:
            seen.add(key)
            unique.append((item[0], item[1], clipped))
    return unique[:MAX_CANDIDATES]


def _parameters_payload(values: np.ndarray) -> dict[str, float]:
    return {
        **dict(zip(_PARAMETER_NAMES, values.tolist(), strict=True)),
        "principal_point_offset_x_px": 0.0,
        "principal_point_offset_y_px": 0.0,
    }


def _candidate_attempt(fit: _Fit, order: int) -> dict[str, Any]:
    source = {
        "preset_nominal": "PRESET_NOMINAL",
        "preset_perturbation": "PRESET_PERTURBATION",
        "persisted_registration": "EXISTING_PNP_CANDIDATE",
    }.get(fit.seed_source, "PRESET_PERTURBATION")
    return {
        "candidate_id": fit.candidate_id,
        "source": source,
        "deterministic_order": order,
        "initial_parameters": _parameters_payload(fit.initial),
        "attempted": True,
        "converged": fit.converged,
        "eligible_for_selection": fit.eligible,
        "robust_loss": "soft_l1",
        "final_cost": fit.final_cost,
        "score": fit.score,
        "rejection_reasons": fit.rejection_reasons,
    }


def _pose_candidate(
    fit: _Fit, preset: _Preset, evidence: _Evidence
) -> CameraPoseCandidate:
    rotation, translation, matrix, position, focal, _ = _camera_arrays(
        fit.parameters, preset, evidence
    )
    lower_focal = evidence.image_width / (
        2 * math.tan(math.radians(preset.upper[6]) / 2)
    )
    upper_focal = evidence.image_width / (
        2 * math.tan(math.radians(preset.lower[6]) / 2)
    )
    intrinsics = CameraIntrinsicsCandidate(
        candidate_id=f"{fit.candidate_id}:preset_fov",
        focal_length_x_px=focal,
        focal_length_y_px=focal,
        principal_point_x_px=evidence.image_width / 2,
        principal_point_y_px=evidence.image_height / 2,
        distortion_coefficients=[0.0] * 5,
        source="bounded_image_hypothesis",
        confidence="MEDIUM",
        horizontal_fov_degrees=float(fit.parameters[6]),
        lower_focal_bound_px=lower_focal,
        upper_focal_bound_px=upper_focal,
        focal_bound_reached=any("horizontal_fov" in item for item in fit.active_bounds),
        distortion_assumption="Zero distortion or pre-undistorted frame required by preset.",
    )
    checks = [
        PlausibilityCheck(
            check_id=item["check_id"],
            passed=item["passed"],
            value=item["value"],
            reason=f"Preset physical check: {item['check_id']}.",
        )
        for item in fit.physical_checks
    ]
    return CameraPoseCandidate(
        candidate_id=fit.candidate_id,
        assignment_hypothesis=evidence.assignment,
        near_semantic_end="bowler" if evidence.assignment == "A" else "striker",
        far_semantic_end="striker" if evidence.assignment == "A" else "bowler",
        lateral_mapping=evidence.lateral_mapping,
        setup_frame_index=evidence.observation.setup_frame.frame_index,
        intrinsics=intrinsics,
        anchor_subset_id="persisted_wicket_evidence",
        attempted=True,
        solver_success=fit.converged,
        pnp_method="preset_bounded_scipy_existing_math",
        refinement=RefinementDiagnostics(
            attempted=True,
            converged=fit.converged,
            method="scipy_least_squares_preset_parameters",
            robust_loss="soft_l1",
            initial_cost=fit.initial_cost,
            final_cost=fit.final_cost,
            iterations=fit.iterations,
            parameters_changed=list(_PARAMETER_NAMES),
            parameters_reaching_bounds=fit.active_bounds,
        ),
        rotation_vector=rotation.reshape(-1).tolist(),
        translation_vector=translation.reshape(-1).tolist(),
        rotation_matrix=matrix.tolist(),
        camera_world_position=position.tolist(),
        reprojection_rmse_px=fit.anchor_rmse_px,
        median_reprojection_error_px=fit.median_anchor_error_px,
        plausibility_checks=checks,
        score=fit.score,
        classification="GROUND_PLANE_CANDIDATE" if fit.score >= 0.55 else "VISUAL_ONLY",
        eligible_for_selection=fit.eligible,
        failure_reasons=fit.rejection_reasons,
        warnings=["Development candidate only; metric analytics remain locked."],
    )


def _standalone_arrays(
    parameters: np.ndarray, preset: _Preset, width: int, height: int
):
    pitch_length = build_virtual_pitch_specification().dimensions.pitch_length_m
    rotation, translation, matrix, position, focal_ratio = _pose(
        parameters, preset, pitch_length
    )
    focal = width * focal_ratio
    camera_matrix = np.asarray(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rotation, translation, matrix, position, focal, camera_matrix


def _box_frames(observations: Any) -> tuple[list[dict[str, Any]], list[int]]:
    frames = [
        dict(item) if isinstance(item, Mapping) else item.model_dump()
        for item in _get(observations, "frames", [])
    ]
    if len(frames) < 4:
        return frames, []
    vectors = np.asarray(
        [
            list(item["near_wicket_bbox"]) + list(item["far_wicket_bbox"])
            for item in frames
        ],
        dtype=np.float64,
    )
    median = np.median(vectors, axis=0)
    mad = np.median(np.abs(vectors - median), axis=0)
    robust_distance = np.max(
        np.abs(vectors - median) / np.maximum(1.4826 * mad, 2.0), axis=1
    )
    outliers = [
        int(frames[index]["frame_index"])
        for index, value in enumerate(robust_distance)
        if value > 8.0
    ]
    return [
        item for item in frames if int(item["frame_index"]) not in outliers
    ], sorted(outliers)


def _standalone_objective(
    parameters: np.ndarray,
    preset: _Preset,
    width: int,
    height: int,
    frames: Sequence[dict[str, Any]],
    anchors: Sequence[dict[str, Any]],
) -> np.ndarray:
    rotation, translation, _, _, _, camera_matrix = _standalone_arrays(
        parameters, preset, width, height
    )
    near_end = "bowler" if preset.camera_end == "bowler" else "striker"
    far_end = "striker" if near_end == "bowler" else "bowler"
    near_projected = _projected_wicket_box(
        near_end, rotation, translation, camera_matrix
    )
    far_projected = _projected_wicket_box(far_end, rotation, translation, camera_matrix)
    residuals: list[float] = []
    for frame in frames:
        for role, projected in (("near", near_projected), ("far", far_projected)):
            values = frame[f"{role}_wicket_bbox"]
            observed = PixelBox(
                x=values[0], y=values[1], width=values[2], height=values[3]
            )
            confidence = float(frame.get(f"{role}_confidence", 0.5))
            residuals.extend(
                _box_residuals(projected, observed, math.sqrt(max(confidence, 0.05)))
            )
    pitch_length = build_virtual_pitch_specification().dimensions.pitch_length_m
    anchor_world = {
        "near_base_center": [0.0, 0.0 if near_end == "bowler" else pitch_length, 0.0],
        "far_base_center": [0.0, pitch_length if far_end == "striker" else 0.0, 0.0],
    }
    for anchor in anchors:
        world_value = anchor_world.get(str(anchor.get("semantic_id")))
        if world_value is None:
            continue
        projected, depths = _project(
            np.asarray([world_value]), rotation, translation, camera_matrix
        )
        if depths[0] <= 0:
            residuals.extend([25.0, 25.0])
        else:
            uncertainty = max(float(anchor.get("uncertainty_px", 3.0)), 1.0)
            residuals.extend(
                (
                    (projected[0] - [float(anchor["x"]), float(anchor["y"])])
                    / uncertainty
                    * 5.0
                ).tolist()
            )
    half_ranges = np.maximum((preset.upper - preset.lower) / 2.0, 1e-6)
    evidence_strength = min(1.0, (len(frames) / 5.0) * (1.0 if anchors else 0.45))
    prior_weight = 0.12 + 0.3 * (1.0 - evidence_strength)
    residuals.extend(
        ((parameters - preset.nominal) / half_ranges * prior_weight).tolist()
    )
    return np.asarray(residuals, dtype=np.float64)


def _standalone_normalised_objective(
    values: np.ndarray,
    preset: _Preset,
    width: int,
    height: int,
    frames: Sequence[dict[str, Any]],
    anchors: Sequence[dict[str, Any]],
) -> np.ndarray:
    parameters = denormalize_parameters(values, preset.lower, preset.upper)
    return _standalone_objective(
        parameters, preset, width, height, frames, anchors
    )


def fit_bounded_camera(
    *,
    preset: Any,
    observations: Any,
    deterministic_seed: int = OPTIMISATION_SEED,
) -> dict[str, Any]:
    """Bounded deterministic fitting kernel for contract tests and service reuse."""
    normalised = _normalise_preset(preset)
    width = int(_get(observations, "image_width", 0))
    height = int(_get(observations, "image_height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("Observation image dimensions must be positive.")
    frames, outlier_ids = _box_frames(observations)
    anchors = [
        dict(item) if isinstance(item, Mapping) else item.model_dump()
        for item in _get(observations, "point_anchors", [])
    ]
    spans = normalised.upper - normalised.lower
    seeds: list[tuple[str, np.ndarray]] = [("nominal", normalised.nominal.copy())]
    for index, name in enumerate(_PARAMETER_NAMES):
        for sign, label in ((-1.0, "low"), (1.0, "high")):
            seed = normalised.nominal.copy()
            seed[index] += sign * spans[index] * 0.2
            seeds.append((f"{name}_{label}", seed))
    attempts: list[tuple[str, Any]] = []
    for candidate_id, seed in seeds:
        clipped = np.clip(seed, normalised.lower + 1e-8, normalised.upper - 1e-8)
        scaled = normalize_parameters(clipped, normalised.lower, normalised.upper)
        result = least_squares(
            _standalone_normalised_objective,
            scaled,
            args=(normalised, width, height, frames, anchors),
            bounds=(-np.ones_like(scaled), np.ones_like(scaled)),
            loss="soft_l1",
            x_scale=np.ones_like(scaled),
            max_nfev=160,
        )
        attempts.append((candidate_id, result))
    attempts.sort(key=lambda item: (float(item[1].cost), item[0]))
    selected_id, selected = attempts[0]
    parameters = denormalize_parameters(
        selected.x.astype(np.float64), normalised.lower, normalised.upper
    )
    rotation, translation, _, _, _, camera_matrix = _standalone_arrays(
        parameters, normalised, width, height
    )
    pitch_length = build_virtual_pitch_specification().dimensions.pitch_length_m
    anchor_errors: list[float] = []
    for anchor in anchors:
        semantic_id = str(anchor.get("semantic_id"))
        y = (
            0.0
            if semantic_id == "near_base_center"
            else pitch_length if semantic_id == "far_base_center" else None
        )
        if y is None:
            continue
        projected, depths = _project(
            np.asarray([[0.0, y, 0.0]]), rotation, translation, camera_matrix
        )
        if depths[0] > 0:
            anchor_errors.append(
                float(np.linalg.norm(projected[0] - [anchor["x"], anchor["y"]]))
            )
    frame_ious: list[tuple[int, float]] = []
    near_box = _projected_wicket_box(
        "bowler" if normalised.camera_end == "bowler" else "striker",
        rotation,
        translation,
        camera_matrix,
    )
    far_box = _projected_wicket_box(
        "striker" if normalised.camera_end == "bowler" else "bowler",
        rotation,
        translation,
        camera_matrix,
    )
    for frame in frames:
        values: list[float] = []
        for role, projected in (("near", near_box), ("far", far_box)):
            raw = frame[f"{role}_wicket_bbox"]
            observed = PixelBox(x=raw[0], y=raw[1], width=raw[2], height=raw[3])
            values.append(_box_iou(projected, observed) if projected else 0.0)
        frame_ious.append((int(frame["frame_index"]), float(np.mean(values))))
    rng = np.random.default_rng(deterministic_seed)
    probes: list[np.ndarray] = []
    for _ in range(UNCERTAINTY_PERTURBATIONS):
        seed = np.clip(
            parameters + rng.normal(0.0, 0.01, len(parameters)) * spans,
            normalised.lower,
            normalised.upper,
        )
        scaled = normalize_parameters(seed, normalised.lower, normalised.upper)
        probe = least_squares(
            _standalone_normalised_objective,
            scaled,
            args=(normalised, width, height, frames, anchors),
            bounds=(-np.ones_like(scaled), np.ones_like(scaled)),
            loss="soft_l1",
            x_scale=np.ones_like(scaled),
            max_nfev=100,
        )
        if probe.success:
            probes.append(
                denormalize_parameters(
                    probe.x, normalised.lower, normalised.upper
                )
            )
    spread = (
        np.max(np.abs(np.asarray(probes) - parameters), axis=0)
        if probes
        else np.full(7, math.inf)
    )
    confidences = [float(frame.get("near_confidence", 0.0)) for frame in frames] + [
        float(frame.get("far_confidence", 0.0)) for frame in frames
    ]
    strong = bool(anchors and confidences and float(np.mean(confidences)) >= 0.7)
    rmse = float(np.sqrt(np.mean(np.square(anchor_errors)))) if anchor_errors else None
    return {
        "fitted_parameters": dict(
            zip(_PARAMETER_NAMES, parameters.tolist(), strict=True)
        ),
        "robust_loss": "soft_l1",
        "preset_prior_applied": True,
        "strong_evidence_overrode_nominal": strong
        and bool(
            np.linalg.norm((parameters - normalised.nominal) / np.maximum(spans, 1e-9))
            > 0.005
        ),
        "anchor_metrics": {
            "rmse_px": rmse,
            "median_error_px": (
                float(np.median(anchor_errors)) if anchor_errors else None
            ),
        },
        "temporal_metrics": {
            "frame_count": len(frames) + len(outlier_ids),
            "successful_frame_count": len(frames),
            "temporal_stability_score": (
                float(np.median([item[1] for item in frame_ious]))
                if frame_ious
                else 0.0
            ),
        },
        "outlier_frame_ids": outlier_ids,
        "camera_pose_count": 1,
        "attempted_candidate_ids": [item[0] for item in attempts],
        "selected_candidate_id": selected_id,
        "uncertainty": {
            "deterministic_seed": deterministic_seed,
            "perturbation_count": len(probes),
            "parameter_spread": dict(
                zip(_PARAMETER_NAMES, spread.tolist(), strict=True)
            ),
        },
    }


fit_preset_registration = fit_bounded_camera


def classify_auto_registration(
    *,
    compatibility_status: str,
    both_wickets_available: bool,
    fit_valid: bool,
    physical_checks_passed: bool,
    temporal_stability_score: float,
    uncertainty_acceptable: bool,
    overlay_usable: bool,
) -> str:
    if compatibility_status == "INCOMPATIBLE":
        return "PRESET_INCOMPATIBLE"
    if not both_wickets_available:
        return "INSUFFICIENT_WICKETS"
    if not fit_valid:
        return "FAILED"
    if not physical_checks_passed or temporal_stability_score < 0.25:
        return "NEEDS_ASSISTANCE"
    if uncertainty_acceptable and temporal_stability_score >= 0.5:
        return "AUTO_REGISTRATION_READY"
    return "VISUAL_OVERLAY_READY" if overlay_usable else "NEEDS_ASSISTANCE"


classify_registration_result = classify_auto_registration


def _uncertainty(
    selected: _Fit, preset: _Preset, evidence: _Evidence
) -> dict[str, Any]:
    rng = np.random.default_rng(OPTIMISATION_SEED)
    spans = preset.upper - preset.lower
    fitted: list[_Fit] = []
    for index in range(UNCERTAINTY_PERTURBATIONS):
        perturbation = rng.normal(0.0, 0.015, size=len(_PARAMETER_NAMES)) * spans
        initial = np.clip(
            selected.parameters + perturbation, preset.lower, preset.upper
        )
        try:
            fitted.append(
                _fit_candidate(
                    f"uncertainty_{index:02d}",
                    "deterministic_perturbation",
                    initial,
                    preset,
                    evidence,
                )
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
    valid = [item for item in fitted if item.converged]
    if len(valid) < 3:
        return {
            "deterministic_seed": OPTIMISATION_SEED,
            "perturbation_count": len(valid),
            "stable": False,
            "warnings": ["Too few deterministic perturbation fits converged."],
        }
    values = np.asarray([item.parameters for item in valid])
    medians = np.median(values, axis=0)
    spread = np.max(np.abs(values - medians), axis=0)
    score_order_stability = sum(
        item.score <= selected.score + 0.03 for item in valid
    ) / len(valid)
    stable = bool(
        spread[0] <= 0.5
        and spread[1] <= 1.0
        and spread[2] <= 0.5
        and np.max(spread[3:6]) <= 2.0
        and spread[6] <= 3.0
        and score_order_stability >= 0.75
    )
    return {
        "deterministic_seed": OPTIMISATION_SEED,
        "perturbation_count": len(valid),
        "camera_position_spread_m": float(np.linalg.norm(spread[:3])),
        "rotation_spread_deg": float(np.max(spread[3:6])),
        "horizontal_fov_spread_deg": float(spread[6]),
        "candidate_ordering_stability": score_order_stability,
        "stable": stable,
        "warnings": (
            []
            if stable
            else ["Preset fit is sensitive to deterministic initial perturbations."]
        ),
    }


def _projection_payload(fit: _Fit, preset: _Preset, evidence: _Evidence):
    rotation, translation, rotation_matrix, position, focal, _ = _camera_arrays(
        fit.parameters, preset, evidence
    )
    fov = float(fit.parameters[6])
    intrinsics = type(
        "_Intrinsics",
        (),
        {
            "focal_length_x_px": focal,
            "principal_point_x_px": evidence.image_width / 2.0,
            "principal_point_y_px": evidence.image_height / 2.0,
            "distortion_coefficients": [0.0] * 5,
        },
    )()
    camera = _virtual_camera(
        f"preset_auto:{fit.candidate_id}",
        evidence.image_width,
        evidence.image_height,
        intrinsics,
        rotation,
        translation,
        rotation_matrix,
        position,
    )
    camera = camera.model_copy(update={"horizontal_fov_degrees": fov})
    projected = project_virtual_pitch(camera)
    geometry = RealProjectedPitchGeometry(
        source_camera=camera,
        projected_landmarks=projected.projected_landmarks,
        projected_line_segments=projected.projected_line_segments,
        projected_stumps=projected.projected_stumps,
        projected_polygons=projected.projected_polygons,
        projected_bails=projected.projected_bails,
        diagnostics=projected.diagnostics,
    )
    bridge = CameraBridgeInput(
        source="REAL_PITCH_REGISTRATION_CANDIDATE",
        source_version="preset_auto_registration_v1",
        analysis_id=evidence.observation.analysis_id,
        candidate_id=fit.candidate_id,
        accepted=False,
        classification="PRESET_AUTO_REGISTRATION_CANDIDATE",
        image_width=evidence.image_width,
        image_height=evidence.image_height,
        camera_matrix=camera.camera_matrix,
        fx=focal,
        fy=focal,
        cx=evidence.image_width / 2.0,
        cy=evidence.image_height / 2.0,
        skew=0.0,
        distortion=_distortion([0.0] * 5),
        rotation_vector=rotation.reshape(-1).tolist(),
        rotation_matrix=rotation_matrix.tolist(),
        translation_vector=translation.reshape(-1).tolist(),
        camera_world_position=position.tolist(),
        near_m=DEFAULT_NEAR_M,
        far_m=DEFAULT_FAR_M,
        setup_frame=None,
        warnings=["Development candidate only; production acceptance is false."],
    )
    return bridge.model_dump(mode="json"), geometry.model_dump(mode="json")


def _schema_result(payload: dict[str, Any]) -> Any:
    """Validate through Agent 1's contract when present; dict is the isolated adapter."""
    try:
        module = importlib.import_module(
            "services.api.schemas.preset_auto_registration"
        )
    except ModuleNotFoundError as exc:
        if exc.name != "services.api.schemas.preset_auto_registration":
            raise
        return payload
    result_type = getattr(module, "PresetAutoRegistrationResult", None)
    return result_type.model_validate(payload) if result_type is not None else payload


def persist_preset_auto_registration(
    analysis_id: str,
    result: Any,
    *,
    reports_directory: Path | None = None,
) -> Path:
    payload = (
        result.model_dump(mode="json")
        if hasattr(result, "model_dump")
        else dict(result)
    )
    if payload.get("production_accepted") is not False or payload.get(
        "metrics_unlocked"
    ) not in (False, []):
        raise ValueError(
            "Automatic registration persistence cannot accept calibration or unlock metrics."
        )
    destination = (
        reports_directory or (VIDEO_ANALYSIS_ROOT / analysis_id / "reports")
    ) / RESULT_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


_persist_result = persist_preset_auto_registration


def _persist(analysis_id: str, payload: dict[str, Any]) -> None:
    persist_preset_auto_registration(analysis_id, payload)


def _auto_register_from_observation(
    *,
    analysis_id: str,
    preset: Any,
    observation: WicketObservationResult,
    observation_source: str = "PERSISTED_WICKET_OBSERVATION_V1",
    detection_reused: bool = True,
    observation_load_ms: float | None = None,
    workflow_started: float | None = None,
    development_diagnostics: bool = False,
) -> Any:
    """Fit one preset-constrained camera from already materialised evidence."""
    started = workflow_started if workflow_started is not None else perf_counter()
    timings: dict[str, float] = {}
    normalised = _normalise_preset(preset)
    compatibility = _compatibility(observation, normalised)
    if compatibility["status"] == "INCOMPATIBLE":
        reasons = compatibility["reasons"]
        reason_codes = [item["reason_code"] for item in reasons]
        status = (
            "INSUFFICIENT_WICKETS"
            if "BOTH_WICKETS_REQUIRED" in reason_codes
            else "PRESET_INCOMPATIBLE"
        )
        payload = {
            "preset_auto_registration_version": RESULT_VERSION,
            "analysis_id": analysis_id,
            "status": status,
            "geometric_classification": "REGISTRATION_FAILED",
            "preset": (
                preset.model_dump(mode="json")
                if hasattr(preset, "model_dump")
                else preset
            ),
            "preset_compatibility": compatibility,
            "setup_frame": (
                observation.setup_frame.model_dump(mode="json")
                if observation.setup_frame
                else None
            ),
            "supporting_frames": [
                item.model_dump(mode="json") for item in observation.supporting_frames
            ],
            "observation_source": observation_source,
            "detection_reused": detection_reused,
            "candidates_attempted": [],
            "selected_candidate": None,
            "competing_candidate": None,
            "warnings": [],
            "failure_reasons": reason_codes,
            "manual_assistance_available": True,
            "production_accepted": False,
            "metrics_unlocked": [],
            "stage_timings": {
                "observation_load_ms": observation_load_ms,
                "total_ms": (perf_counter() - started) * 1000.0,
            },
        }
        _persist(analysis_id, payload)
        return _schema_result(payload)
    evidence_started = perf_counter()
    evidence = _evidence(observation, normalised)
    used_points = sum(item.status == "USED" for item in evidence.correspondences)
    if used_points < 6:
        failure_evidence_reason = "insufficient_pointlike_evidence"
    timings["prepare_evidence_ms"] = (perf_counter() - evidence_started) * 1000.0
    fitting_started = perf_counter()
    fits: list[_Fit] = []
    rejected_attempts: list[dict[str, Any]] = []
    if used_points >= 6:
        for candidate_id, source, initial in _candidate_seeds(analysis_id, normalised):
            try:
                fits.append(
                    _fit_candidate(candidate_id, source, initial, normalised, evidence)
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                rejected_attempts.append(
                    {
                        "candidate_id": candidate_id,
                        "seed_source": source,
                        "rejection_reason": type(exc).__name__,
                    }
                )
    timings["bounded_fitting_ms"] = (perf_counter() - fitting_started) * 1000.0
    ranked = sorted(
        (item for item in fits if item.eligible),
        key=lambda item: (-item.score, item.final_cost, item.candidate_id),
    )
    selected = ranked[0] if ranked else None
    competing = ranked[1] if len(ranked) > 1 else None
    uncertainty = None
    bridge_camera = None
    projected_pitch = None
    status = "FAILED"
    classification = "REGISTRATION_FAILED"
    ambiguity = 0.0
    failure_reasons: list[str] = []
    if used_points < 6:
        status = "INSUFFICIENT_EVIDENCE"
        failure_reasons.append("insufficient_pointlike_evidence")
    elif selected is None:
        status = "NEEDS_ASSISTANCE" if fits else "FAILED"
        failure_reasons.append("no_physically_valid_bounded_candidate")
    else:
        uncertainty_started = perf_counter()
        uncertainty = _uncertainty(selected, normalised, evidence)
        timings["uncertainty_ms"] = (perf_counter() - uncertainty_started) * 1000.0
        score_gap = selected.score - competing.score if competing else 1.0
        ambiguity = max(0.0, min(1.0, 1.0 - score_gap / 0.08)) if competing else 0.0
        stable = bool(uncertainty["stable"])
        temporal_score = selected.temporal["temporal_stability_score"]
        if (
            selected.score >= 0.72
            and temporal_score >= 0.5
            and stable
            and ambiguity <= 0.5
            and (selected.anchor_rmse_px or math.inf) <= 8.0
        ):
            status = "AUTO_REGISTRATION_READY"
            classification = "GROUND_PLANE_CANDIDATE"
        elif selected.score >= 0.48 and temporal_score >= 0.25:
            status = "VISUAL_OVERLAY_READY"
            classification = "VISUAL_ONLY"
        else:
            status = "NEEDS_ASSISTANCE"
            classification = "VISUAL_ONLY"
        bridge_camera, projected_pitch = _projection_payload(
            selected, normalised, evidence
        )
    timings["total_ms"] = (perf_counter() - started) * 1000.0
    selected_pose = (
        _pose_candidate(selected, normalised, evidence) if selected else None
    )
    competing_pose = (
        _pose_candidate(competing, normalised, evidence) if competing else None
    )
    parameter_changes = []
    active_bounds = []
    if selected:
        for index, name in enumerate(_PARAMETER_NAMES):
            unit = "m" if name.endswith("_m") else "deg"
            parameter_changes.append(
                {
                    "parameter_name": name,
                    "unit": unit,
                    "initial_value": float(selected.initial[index]),
                    "fitted_value": float(selected.parameters[index]),
                    "delta": float(
                        selected.parameters[index] - selected.initial[index]
                    ),
                }
            )
        for item in selected.active_bounds:
            name, side = item.split(":", 1)
            index = _PARAMETER_NAMES.index(name)
            active_bounds.append(
                {
                    "parameter_name": name,
                    "bound": "MINIMUM" if side == "lower" else "MAXIMUM",
                    "value": float(selected.parameters[index]),
                    "unit": "m" if name.endswith("_m") else "deg",
                    "critical": name
                    in {
                        "camera_height_m",
                        "distance_behind_wicket_m",
                        "horizontal_fov_deg",
                    },
                }
            )
    anchor_items = [item for item in evidence.correspondences if item.status == "USED"]
    soft_items = [
        item for item in evidence.correspondences if item.status == "SOFT_ONLY"
    ]
    typed_uncertainty = (
        None
        if uncertainty is None
        else {
            "perturbation_count": uncertainty["perturbation_count"],
            "deterministic_seed": uncertainty["deterministic_seed"],
            "camera_position_spread_m": uncertainty.get("camera_position_spread_m"),
            "camera_rotation_spread_deg": uncertainty.get("rotation_spread_degrees"),
            "horizontal_fov_spread_deg": uncertainty.get("horizontal_fov_spread_deg"),
            "candidate_ordering_stability": uncertainty.get(
                "candidate_ordering_stability"
            ),
            "stable": uncertainty["stable"],
            "warnings": uncertainty.get("warnings", []),
        }
    )
    payload = {
        "preset_auto_registration_version": RESULT_VERSION,
        "analysis_id": analysis_id,
        "status": status,
        "geometric_classification": classification,
        "preset": (
            preset.model_dump(mode="json") if hasattr(preset, "model_dump") else preset
        ),
        "preset_compatibility": compatibility,
        "setup_frame": (
            observation.setup_frame.model_dump(mode="json")
            if observation.setup_frame
            else None
        ),
        "supporting_frames": [
            item.model_dump(mode="json") for item in observation.supporting_frames
        ],
        "observation_source": observation_source,
        "detection_reused": detection_reused,
        "candidates_attempted": [
            _candidate_attempt(item, index) for index, item in enumerate(fits)
        ],
        "selected_candidate": (
            selected_pose.model_dump(mode="json") if selected_pose else None
        ),
        "competing_candidate": (
            competing_pose.model_dump(mode="json") if competing_pose else None
        ),
        "fitted_parameters": (
            _parameters_payload(selected.parameters) if selected else None
        ),
        "initial_parameters": (
            _parameters_payload(selected.initial) if selected else None
        ),
        "parameter_changes": parameter_changes,
        "active_bounds": active_bounds,
        "anchor_metrics": (
            {
                "exact_anchor_count": sum(
                    item.exactness == "EXACT" for item in anchor_items
                ),
                "pointlike_anchor_count": sum(
                    item.exactness == "POINTLIKE" for item in anchor_items
                ),
                "soft_constraint_count": len(soft_items),
                "inlier_count": len(anchor_items),
                "outlier_count": 0,
                "reprojection_rmse_px": selected.anchor_rmse_px,
                "median_reprojection_error_px": selected.median_anchor_error_px,
                "maximum_inlier_error_px": None,
            }
            if selected
            else None
        ),
        "envelope_metrics": (
            {"near_wicket_iou": selected.near_iou, "far_wicket_iou": selected.far_iou}
            if selected
            else None
        ),
        "temporal_metrics": (
            {
                "frame_count": selected.temporal["frame_count"],
                "successful_frame_count": selected.temporal["successful_frame_count"],
                "median_near_wicket_iou": selected.temporal["median_near_iou"],
                "median_far_wicket_iou": selected.temporal["median_far_iou"],
                "median_centre_residual_px": selected.temporal[
                    "median_centre_residual_px"
                ],
                "median_width_residual_px": selected.temporal[
                    "median_width_residual_px"
                ],
                "median_height_residual_px": selected.temporal[
                    "median_height_residual_px"
                ],
                "scale_consistency_score": selected.temporal["scale_consistency"],
                "temporal_stability_score": selected.temporal[
                    "temporal_stability_score"
                ],
                "worst_supporting_frame_index": selected.temporal[
                    "worst_supporting_frame"
                ],
            }
            if selected
            else None
        ),
        "physical_checks": (
            [
                {**item, "reason": f"Preset physical check: {item['check_id']}."}
                for item in selected.physical_checks
            ]
            if selected
            else []
        ),
        "uncertainty": typed_uncertainty,
        "ambiguity": {
            "score": ambiguity,
            "competing_solution_plausible": competing is not None,
            "selected_candidate_id": selected.candidate_id if selected else None,
            "competing_candidate_id": competing.candidate_id if competing else None,
            "reasons": (
                ["Competing bounded candidates score similarly."]
                if ambiguity > 0.5
                else []
            ),
        },
        "projected_pitch": projected_pitch,
        "bridge_camera": bridge_camera,
        "warnings": [
            "Automatic result is a development candidate; production acceptance and metric analytics remain locked."
        ],
        "failure_reasons": failure_reasons,
        "manual_assistance_available": status
        not in {"AUTO_REGISTRATION_READY", "VISUAL_OVERLAY_READY"},
        "production_accepted": False,
        "metrics_unlocked": [],
        "stage_timings": {
            "observation_load_ms": observation_load_ms,
            "candidate_generation_ms": timings.get("prepare_evidence_ms"),
            "optimisation_ms": timings.get("bounded_fitting_ms", 0.0)
            + timings.get("uncertainty_ms", 0.0),
            "temporal_validation_ms": None,
            "total_ms": timings["total_ms"],
        },
    }
    _persist(analysis_id, payload)
    return _schema_result(payload)


def _preset_by_id(preset_id: str) -> Any:
    try:
        module = importlib.import_module(
            "services.api.schemas.preset_auto_registration"
        )
    except ModuleNotFoundError as exc:
        if exc.name == "services.api.schemas.preset_auto_registration":
            raise VideoAnalysisServiceError(
                "Preset schema integration is not available in this worktree.",
                status_code=503,
            ) from exc
        raise
    for name in (
        "get_camera_setup_preset",
        "get_preset",
        "load_camera_setup_preset",
    ):
        getter = getattr(module, name, None)
        if getter is not None:
            preset = getter(preset_id)
            if preset is not None:
                return preset
    for name in ("CAMERA_SETUP_PRESETS_BY_ID", "CAMERA_SETUP_PRESETS", "PRESETS"):
        presets = getattr(module, name, None)
        if isinstance(presets, Mapping) and preset_id in presets:
            return presets[preset_id]
        if presets:
            for preset in presets:
                if str(_get(preset, "preset_id", "")) == preset_id:
                    return preset
    value = getattr(module, preset_id, None)
    if value is not None:
        return value
    raise VideoAnalysisServiceError("Unknown camera setup preset.", status_code=404)


def run_preset_auto_registration(
    analysis_id: str,
    preset_id: str = "STANDARD_REAR_WICKET_NET_V1",
    *,
    preset: Any | None = None,
    reuse_existing_observations: bool = True,
    force_redetect: bool = False,
    development_diagnostics: bool = False,
) -> Any:
    """Run with persisted evidence by default; redetection requires explicit opt-in."""
    workflow_started = perf_counter()
    selected_preset = preset if preset is not None else _preset_by_id(preset_id)
    observation_started = perf_counter()
    if force_redetect:
        observation = run_wicket_observation(analysis_id)
        observation_source = "NEW_WICKET_OBSERVATION_V1"
        detection_reused = False
    elif reuse_existing_observations:
        observation = load_wicket_observation(analysis_id)
        observation_source = "PERSISTED_WICKET_OBSERVATION_V1"
        detection_reused = True
    else:
        raise VideoAnalysisServiceError(
            "No wicket evidence source was requested.", status_code=422
        )
    observation_load_ms = (perf_counter() - observation_started) * 1000.0
    return _auto_register_from_observation(
        analysis_id=analysis_id,
        preset=selected_preset,
        observation=observation,
        observation_source=observation_source,
        detection_reused=detection_reused,
        observation_load_ms=observation_load_ms,
        workflow_started=workflow_started,
        development_diagnostics=development_diagnostics,
    )


def load_preset_auto_registration(analysis_id: str) -> Any:
    load_video_analysis(analysis_id)
    path = VIDEO_ANALYSIS_ROOT / analysis_id / "reports" / RESULT_FILENAME
    if not path.is_file():
        raise VideoAnalysisServiceError(
            "Preset auto-registration has not been generated for this analysis.",
            status_code=404,
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VideoAnalysisServiceError(
            "Stored preset auto-registration is unavailable.", status_code=500
        ) from exc
    return _schema_result(payload)


def clear_preset_auto_registration(analysis_id: str) -> bool:
    load_video_analysis(analysis_id)
    path = VIDEO_ANALYSIS_ROOT / analysis_id / "reports" / RESULT_FILENAME
    if not path.exists():
        return False
    path.unlink()
    return True
