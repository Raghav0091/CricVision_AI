"""Candidate-only real camera pose registration from wicket observations."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares

from ..schemas.real_pitch_registration import (
    CameraIntrinsicsCandidate,
    CameraPoseCandidate,
    IndependentValidation,
    PlausibilityCheck,
    PoseUncertainty,
    RealPitchRegistrationResult,
    RealProjectedPitchGeometry,
    RefinementDiagnostics,
    RegistrationCorrespondence,
    RegistrationDiagnostics,
    ReprojectionResidual,
    TemporalValidation,
)
from ..schemas.virtual_pitch import VirtualCamera, WorldPoint3D
from ..schemas.wicket_observation import (
    PixelBox,
    PixelPoint,
    WicketLandmarkObservation,
    WicketObservation,
    WicketObservationResult,
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
)


RESULT_FILENAME = "real_pitch_registration_v1.json"
DEBUG_DIRECTORY = "real_pitch_registration_v1"
FOCAL_FOV_HYPOTHESES = (45.0, 60.0, 75.0)
PNP_RANSAC_ITERATIONS = 500
PNP_CONFIDENCE = 0.999
MIN_POINTLIKE_ANCHORS = 8
MIN_ASSISTED_POINTLIKE_ANCHORS = 6
MIN_ANCHORS_PER_WICKET = 4
AMBIGUITY_SCORE_GAP = 0.08
PERTURBATION_SEED = 1729
PERTURBATION_COUNT = 8


@dataclass(frozen=True)
class Eligibility:
    eligible: bool
    reasons: list[str]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _world(point: Sequence[float]) -> WorldPoint3D:
    return WorldPoint3D(x=float(point[0]), y=float(point[1]), z=float(point[2]))


def _point_from_landmark(item: WicketLandmarkObservation) -> PixelPoint | None:
    if item.pixel_x is None or item.pixel_y is None:
        return None
    return PixelPoint(x=item.pixel_x, y=item.pixel_y)


def _available_landmark(
    observation: WicketObservation,
    semantic_id: str,
) -> WicketLandmarkObservation | None:
    for item in observation.coarse_landmarks + observation.detailed_landmarks:
        if item.semantic_id == semantic_id and item.status == "AVAILABLE":
            return item
    return None


def check_registration_eligibility(
    observation: WicketObservationResult,
) -> Eligibility:
    reasons: list[str] = []
    if observation.status != "READY_FOR_REGISTRATION_EXPERIMENT":
        reasons.append(f"wicket_observation_status:{observation.status}")
    if observation.setup_frame is None:
        reasons.append("setup_frame_unavailable")
    if not observation.supporting_frames:
        reasons.append("supporting_setup_frames_unavailable")
    if observation.near_wicket is None or observation.far_wicket is None:
        reasons.append("two_independent_wickets_required")
        return Eligibility(False, reasons)
    for role, wicket in (
        ("near", observation.near_wicket),
        ("far", observation.far_wicket),
    ):
        if observation.setup_frame is not None and (
            wicket.roi.source_frame_width != observation.setup_frame.image_width
            or wicket.roi.source_frame_height != observation.setup_frame.image_height
        ):
            reasons.append(f"{role}_wicket_source_dimensions_mismatch")
        if wicket.region.stability not in {"STABLE", "PARTIALLY_STABLE"}:
            reasons.append(f"{role}_wicket_not_temporally_stable")
        if wicket.region.rejection_reason:
            reasons.append(f"{role}_wicket_rejected:{wicket.region.rejection_reason}")
        if wicket.quality_factors.get("frame_edge_clipping", 1.0) < 0.7:
            reasons.append(f"{role}_wicket_severely_clipped")
        if not math.isfinite(wicket.region.uncertainty_px):
            reasons.append(f"{role}_wicket_uncertainty_invalid")
        usable = [
            item
            for item in wicket.coarse_landmarks + wicket.detailed_landmarks
            if item.status == "AVAILABLE"
            and item.registration_role in {"PRIMARY_ANCHOR", "SECONDARY_ANCHOR"}
            and math.isfinite(item.uncertainty_px)
        ]
        if len(usable) < MIN_ANCHORS_PER_WICKET:
            reasons.append(f"{role}_wicket_insufficient_anchors")
        if not any("base" in item.semantic_id for item in usable):
            reasons.append(f"{role}_wicket_base_constraint_unavailable")
        if not any(
            "top" in item.semantic_id or item.semantic_id == "top_or_bail_line"
            for item in usable
        ):
            reasons.append(f"{role}_wicket_top_constraint_unavailable")
    return Eligibility(not reasons, reasons)


def build_intrinsics_candidates(
    image_width: int,
    image_height: int,
) -> list[CameraIntrinsicsCandidate]:
    """Build bounded deterministic zero-distortion focal hypotheses."""
    lower = image_width * 0.35
    upper = image_width * 3.5
    candidates: list[CameraIntrinsicsCandidate] = []
    for fov in FOCAL_FOV_HYPOTHESES:
        focal = image_width / (2 * math.tan(math.radians(fov) / 2))
        candidates.append(
            CameraIntrinsicsCandidate(
                candidate_id=f"fov_{int(fov)}",
                focal_length_x_px=focal,
                focal_length_y_px=focal,
                principal_point_x_px=image_width / 2,
                principal_point_y_px=image_height / 2,
                distortion_coefficients=[0.0] * 5,
                source="bounded_image_hypothesis",
                confidence="LOW",
                horizontal_fov_degrees=fov,
                lower_focal_bound_px=lower,
                upper_focal_bound_px=upper,
                distortion_assumption=(
                    "Zero distortion; no trusted clip-specific lens metadata exists."
                ),
            )
        )
    return candidates


def _mapping_weight(item: WicketLandmarkObservation) -> float:
    return _clamp(
        item.confidence / (1.0 + max(item.uncertainty_px, 1.0) / 8.0) ** 2
    )


def _virtual_points_for_end(end: str) -> dict[str, WorldPoint3D]:
    specification = build_virtual_pitch_specification()
    points = {
        item.semantic_id: item.point
        for item in specification.landmarks
        if item.semantic_id.startswith(f"{end}_")
    }
    return points


def build_registration_correspondences(
    observation: WicketObservationResult,
    *,
    assignment_hypothesis: str,
    lateral_mapping: str,
) -> list[RegistrationCorrespondence]:
    """Map evidence semantics without promoting boxes or lines to exact points."""
    assignment = (
        {"near": "bowler", "far": "striker"}
        if assignment_hypothesis == "A"
        else {"near": "striker", "far": "bowler"}
    )
    correspondences: list[RegistrationCorrespondence] = []
    for role, wicket in (
        ("near", observation.near_wicket),
        ("far", observation.far_wicket),
    ):
        if wicket is None:
            continue
        end = assignment[role]
        points = _virtual_points_for_end(end)
        image_to_world_side = (
            {"left": "left", "right": "right"}
            if lateral_mapping == "image_left_to_world_left"
            else {"left": "right", "right": "left"}
        )
        point_mappings = {
            "wicket_outer_left_base": (
                f"{end}_{image_to_world_side['left']}_stump_base"
            ),
            "wicket_outer_right_base": (
                f"{end}_{image_to_world_side['right']}_stump_base"
            ),
            "wicket_base_center": f"{end}_wicket_center_base",
            "wicket_outer_left_top": (
                f"{end}_{image_to_world_side['left']}_stump_top"
            ),
            "wicket_outer_right_top": (
                f"{end}_{image_to_world_side['right']}_stump_top"
            ),
            "wicket_top_center": f"{end}_middle_stump_top",
        }
        detailed_mappings = {
            f"{side}_stump_{level}": (
                f"{end}_{image_to_world_side[side]}_stump_{level}"
            )
            for side in ("left", "right")
            for level in ("base", "top")
        }
        detailed_mappings.update(
            {
                f"middle_stump_{level}": f"{end}_middle_stump_{level}"
                for level in ("base", "top")
            }
        )
        for observed_id, virtual_id in {
            **point_mappings,
            **detailed_mappings,
        }.items():
            landmark = _available_landmark(wicket, observed_id)
            observed_point = (
                _point_from_landmark(landmark) if landmark is not None else None
            )
            detailed = observed_id in detailed_mappings
            usable = (
                landmark is not None
                and observed_point is not None
                and landmark.registration_role
                in {"PRIMARY_ANCHOR", "SECONDARY_ANCHOR"}
            )
            correspondences.append(
                RegistrationCorrespondence(
                    correspondence_id=f"{role}:{observed_id}",
                    observed_wicket_role=role,
                    observed_semantic_id=observed_id,
                    virtual_semantic_id=virtual_id,
                    mapping_type=(
                        "DETAILED_EXACT_POINT"
                        if detailed
                        else "COARSE_POINTLIKE"
                    ),
                    constraint_category="EXACT_OR_POINTLIKE_ANCHOR",
                    exactness="EXACT" if detailed else "POINTLIKE",
                    observed_pixel=observed_point,
                    world_point=points.get(virtual_id),
                    confidence=landmark.confidence if landmark else 0.0,
                    uncertainty_px=landmark.uncertainty_px if landmark else 0.0,
                    registration_weight=(
                        _mapping_weight(landmark) if usable and landmark else 0.0
                    ),
                    source_frames=landmark.supporting_frames if landmark else [],
                    status="USED" if usable else "UNAVAILABLE",
                    rejection_reason=(
                        None
                        if usable
                        else "landmark_unavailable_or_not_anchor_quality"
                    ),
                )
            )

        line_mappings = {
            "base_line": (
                "BASE_LINE",
                f"{end}_{image_to_world_side['left']}_stump_base",
                f"{end}_{image_to_world_side['right']}_stump_base",
            ),
            "top_or_bail_line": (
                "TOP_LINE",
                f"{end}_{image_to_world_side['left']}_stump_top",
                f"{end}_{image_to_world_side['right']}_stump_top",
            ),
            "left_outer_axis": (
                "OUTER_AXIS",
                f"{end}_{image_to_world_side['left']}_stump_base",
                f"{end}_{image_to_world_side['left']}_stump_top",
            ),
            "right_outer_axis": (
                "OUTER_AXIS",
                f"{end}_{image_to_world_side['right']}_stump_base",
                f"{end}_{image_to_world_side['right']}_stump_top",
            ),
        }
        for observed_id, (mapping_type, start_id, end_id) in line_mappings.items():
            landmark = _available_landmark(wicket, observed_id)
            usable = landmark is not None and landmark.line is not None
            correspondences.append(
                RegistrationCorrespondence(
                    correspondence_id=f"{role}:{observed_id}",
                    observed_wicket_role=role,
                    observed_semantic_id=observed_id,
                    virtual_semantic_id=f"{start_id}->{end_id}",
                    mapping_type=mapping_type,
                    constraint_category="SOFT_GEOMETRIC_CONSTRAINT",
                    exactness="SOFT",
                    observed_line_start=(
                        landmark.line.start if usable and landmark else None
                    ),
                    observed_line_end=(
                        landmark.line.end if usable and landmark else None
                    ),
                    virtual_line_start=points.get(start_id),
                    virtual_line_end=points.get(end_id),
                    confidence=landmark.confidence if landmark else 0.0,
                    uncertainty_px=landmark.uncertainty_px if landmark else 0.0,
                    registration_weight=(
                        _mapping_weight(landmark) * 0.25
                        if usable and landmark
                        else 0.0
                    ),
                    source_frames=landmark.supporting_frames if landmark else [],
                    status="SOFT_ONLY" if usable else "UNAVAILABLE",
                    rejection_reason=None if usable else "line_evidence_unavailable",
                )
            )
        centre = _available_landmark(wicket, "wicket_center")
        correspondences.append(
            RegistrationCorrespondence(
                correspondence_id=f"{role}:wicket_center",
                observed_wicket_role=role,
                observed_semantic_id="wicket_center",
                virtual_semantic_id=f"derived:{end}_wicket_body_center",
                mapping_type="WICKET_CENTRE",
                constraint_category="SOFT_GEOMETRIC_CONSTRAINT",
                exactness="SOFT",
                observed_pixel=_point_from_landmark(centre) if centre else None,
                world_point=_world(
                    (
                        0.0,
                        points[f"{end}_wicket_center_base"].y,
                        build_virtual_pitch_specification().dimensions.stump_height_m
                        / 2,
                    )
                ),
                confidence=centre.confidence if centre else 0.0,
                uncertainty_px=centre.uncertainty_px if centre else 0.0,
                registration_weight=(
                    _mapping_weight(centre) * 0.2 if centre else 0.0
                ),
                source_frames=centre.supporting_frames if centre else [],
                status="SOFT_ONLY" if centre else "UNAVAILABLE",
                rejection_reason=None if centre else "centre_evidence_unavailable",
            )
        )
        correspondences.append(
            RegistrationCorrespondence(
                correspondence_id=f"{role}:wicket_envelope",
                observed_wicket_role=role,
                observed_semantic_id="detector_consensus_bbox",
                virtual_semantic_id=f"derived:{end}_wicket_envelope",
                mapping_type="WICKET_ENVELOPE",
                constraint_category="SOFT_GEOMETRIC_CONSTRAINT",
                exactness="SOFT",
                observed_bbox=wicket.region.bbox,
                confidence=wicket.region.detector_confidence,
                uncertainty_px=wicket.region.uncertainty_px,
                registration_weight=_clamp(
                    wicket.region.detector_confidence
                    / (1 + wicket.region.uncertainty_px / 8) ** 2
                    * 0.15
                ),
                source_frames=wicket.region.supporting_frame_ids,
                status="SOFT_ONLY",
            )
        )
    return correspondences


def apply_assisted_anchor_overrides(
    correspondences: list[RegistrationCorrespondence],
    *,
    assignment_hypothesis: str,
    lateral_mapping: str,
    point_overrides: dict[str, PixelPoint],
    crease_overrides: dict[str, PixelPoint] | None = None,
    manual_override_ids: set[str] | None = None,
) -> list[RegistrationCorrespondence]:
    """Apply native-pixel user anchors without warping projected geometry."""
    wicket_mapping = {
        "near_left_base": "near:wicket_outer_left_base",
        "near_right_base": "near:wicket_outer_right_base",
        "near_top_center": "near:wicket_top_center",
        "far_left_base": "far:wicket_outer_left_base",
        "far_right_base": "far:wicket_outer_right_base",
        "far_top_center": "far:wicket_top_center",
    }
    overrides_by_id = {
        correspondence_id: point_overrides[semantic_id]
        for semantic_id, correspondence_id in wicket_mapping.items()
        if semantic_id in point_overrides
    }
    manual_override_ids = manual_override_ids or set()
    fully_manual_roles = {
        role
        for role in ("near", "far")
        if {
            f"{role}_left_base",
            f"{role}_right_base",
            f"{role}_top_center",
        }.issubset(manual_override_ids)
    }
    updated = [
        item.model_copy(
            update={
                "observed_pixel": overrides_by_id[item.correspondence_id],
                "confidence": 0.92,
                "uncertainty_px": 2.0,
                "registration_weight": 0.58,
                "exactness": "EXACT",
                "status": "USED",
                "rejection_reason": None,
            }
        )
        if item.correspondence_id in overrides_by_id
        else item.model_copy(
            update={
                "status": "REJECTED",
                "registration_weight": 0.0,
                "rejection_reason": (
                    "superseded_by_complete_manual_wicket_anchors"
                ),
            }
        )
        if (
            item.observed_wicket_role in fully_manual_roles
            and item.status in {"USED", "SOFT_ONLY"}
        )
        else item
        for item in correspondences
    ]
    if not crease_overrides:
        return updated

    assignment = (
        {"near": "bowler", "far": "striker"}
        if assignment_hypothesis == "A"
        else {"near": "striker", "far": "bowler"}
    )
    side_mapping = (
        {"left": "left", "right": "right"}
        if lateral_mapping == "image_left_to_world_left"
        else {"left": "right", "right": "left"}
    )
    landmarks = {
        item.semantic_id: item.point
        for item in build_virtual_pitch_specification().landmarks
    }
    for semantic_id, point in sorted(crease_overrides.items()):
        parts = semantic_id.split("_")
        if (
            len(parts) != 4
            or parts[0] not in {"near", "far"}
            or parts[1] != "popping"
            or parts[2] != "crease"
            or parts[3] not in {"left", "right"}
        ):
            continue
        role, side = parts[0], parts[3]
        end = assignment[role]
        virtual_id = (
            f"{end}_popping_crease_{side_mapping[side]}_endpoint"
        )
        world_point = landmarks.get(virtual_id)
        if world_point is None:
            continue
        updated.append(
            RegistrationCorrespondence(
                correspondence_id=f"{role}:{semantic_id}",
                observed_wicket_role=role,
                observed_semantic_id=semantic_id,
                virtual_semantic_id=virtual_id,
                mapping_type="CREASE_POINT",
                constraint_category="EXACT_OR_POINTLIKE_ANCHOR",
                exactness="POINTLIKE",
                observed_pixel=point,
                world_point=world_point,
                confidence=0.72,
                uncertainty_px=4.0,
                registration_weight=0.12,
                status="USED",
            )
        )
    return updated


def _camera_matrix(intrinsics: CameraIntrinsicsCandidate, focal: float | None = None):
    value = float(focal or intrinsics.focal_length_x_px)
    return np.asarray(
        [
            [value, 0.0, intrinsics.principal_point_x_px],
            [0.0, value, intrinsics.principal_point_y_px],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _point_arrays(
    correspondences: Sequence[RegistrationCorrespondence],
) -> tuple[np.ndarray, np.ndarray, list[RegistrationCorrespondence]]:
    used = [
        item
        for item in correspondences
        if item.status == "USED"
        and item.observed_pixel is not None
        and item.world_point is not None
    ]
    world = np.asarray(
        [[item.world_point.x, item.world_point.y, item.world_point.z] for item in used],
        dtype=np.float64,
    )
    image = np.asarray(
        [[item.observed_pixel.x, item.observed_pixel.y] for item in used],
        dtype=np.float64,
    )
    return world, image, used


def _project(
    world: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    projected, _ = cv2.projectPoints(
        world,
        rotation,
        translation,
        camera_matrix,
        np.zeros(5, dtype=np.float64),
    )
    rotation_matrix, _ = cv2.Rodrigues(rotation)
    camera_points = (
        rotation_matrix @ world.T + translation.reshape(3, 1)
    ).T
    return projected.reshape(-1, 2), camera_points[:, 2]


def _line_residuals(
    correspondence: RegistrationCorrespondence,
    rotation: np.ndarray,
    translation: np.ndarray,
    camera_matrix: np.ndarray,
) -> list[float]:
    if (
        correspondence.observed_line_start is None
        or correspondence.observed_line_end is None
        or correspondence.virtual_line_start is None
        or correspondence.virtual_line_end is None
    ):
        return []
    world = np.asarray(
        [
            [
                correspondence.virtual_line_start.x,
                correspondence.virtual_line_start.y,
                correspondence.virtual_line_start.z,
            ],
            [
                correspondence.virtual_line_end.x,
                correspondence.virtual_line_end.y,
                correspondence.virtual_line_end.z,
            ],
        ],
        dtype=np.float64,
    )
    projected, depths = _project(world, rotation, translation, camera_matrix)
    if np.any(depths <= 0):
        return [25.0, 25.0]
    observed = np.asarray(
        [
            [correspondence.observed_line_start.x, correspondence.observed_line_start.y],
            [correspondence.observed_line_end.x, correspondence.observed_line_end.y],
        ],
        dtype=np.float64,
    )
    direct = np.linalg.norm(projected - observed, axis=1)
    swapped = np.linalg.norm(projected[::-1] - observed, axis=1)
    distances = direct if direct.sum() <= swapped.sum() else swapped
    scale = math.sqrt(max(correspondence.registration_weight, 1e-6))
    uncertainty = max(correspondence.uncertainty_px, 1.0)
    return [float(value / uncertainty * scale) for value in distances]


def _refinement_residuals(
    parameters: np.ndarray,
    correspondences: Sequence[RegistrationCorrespondence],
    intrinsics: CameraIntrinsicsCandidate,
    focal_prior: float,
) -> np.ndarray:
    rotation = parameters[:3].reshape(3, 1)
    translation = parameters[3:6].reshape(3, 1)
    focal = float(parameters[6])
    camera_matrix = _camera_matrix(intrinsics, focal)
    residuals: list[float] = []
    for item in correspondences:
        if (
            item.status == "USED"
            and item.observed_pixel is not None
            and item.world_point is not None
        ):
            world = np.asarray(
                [[item.world_point.x, item.world_point.y, item.world_point.z]],
                dtype=np.float64,
            )
            projected, depths = _project(
                world, rotation, translation, camera_matrix
            )
            if depths[0] <= 0:
                residuals.extend([25.0, 25.0])
                continue
            scale = math.sqrt(max(item.registration_weight, 1e-6))
            uncertainty = max(item.uncertainty_px, 1.0)
            residuals.extend(
                [
                    (projected[0, 0] - item.observed_pixel.x)
                    / uncertainty
                    * scale,
                    (projected[0, 1] - item.observed_pixel.y)
                    / uncertainty
                    * scale,
                ]
            )
        elif item.status == "SOFT_ONLY" and item.mapping_type in {
            "TOP_LINE",
            "BASE_LINE",
            "OUTER_AXIS",
        }:
            residuals.extend(
                _line_residuals(item, rotation, translation, camera_matrix)
            )
        elif (
            item.status == "SOFT_ONLY"
            and item.mapping_type == "WICKET_CENTRE"
            and item.observed_pixel is not None
            and item.world_point is not None
        ):
            world = np.asarray(
                [[item.world_point.x, item.world_point.y, item.world_point.z]],
                dtype=np.float64,
            )
            projected, depths = _project(
                world, rotation, translation, camera_matrix
            )
            if depths[0] > 0:
                scale = math.sqrt(max(item.registration_weight, 1e-6))
                uncertainty = max(item.uncertainty_px, 1.0)
                residuals.extend(
                    [
                        (projected[0, 0] - item.observed_pixel.x)
                        / uncertainty
                        * scale,
                        (projected[0, 1] - item.observed_pixel.y)
                        / uncertainty
                        * scale,
                    ]
                )
    residuals.append((focal - focal_prior) / max(focal_prior * 0.25, 1.0))
    return np.asarray(residuals, dtype=np.float64)


def _refine_pose(
    rotation: np.ndarray,
    translation: np.ndarray,
    correspondences: Sequence[RegistrationCorrespondence],
    intrinsics: CameraIntrinsicsCandidate,
) -> tuple[np.ndarray, np.ndarray, float, RefinementDiagnostics]:
    initial = np.concatenate(
        [
            rotation.reshape(3),
            translation.reshape(3),
            [intrinsics.focal_length_x_px],
        ]
    )
    lower = np.asarray(
        [-2 * math.pi] * 3
        + [-200.0, -200.0, -200.0]
        + [intrinsics.lower_focal_bound_px]
    )
    upper = np.asarray(
        [2 * math.pi] * 3
        + [200.0, 200.0, 200.0]
        + [intrinsics.upper_focal_bound_px]
    )
    initial = np.clip(initial, lower + 1e-8, upper - 1e-8)
    initial_residuals = _refinement_residuals(
        initial,
        correspondences,
        intrinsics,
        intrinsics.focal_length_x_px,
    )
    try:
        result = least_squares(
            _refinement_residuals,
            initial,
            args=(
                correspondences,
                intrinsics,
                intrinsics.focal_length_x_px,
            ),
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=180,
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return (
            rotation,
            translation,
            intrinsics.focal_length_x_px,
            RefinementDiagnostics(
                attempted=True,
                converged=False,
                method="scipy_least_squares",
                robust_loss="soft_l1",
                initial_cost=float(np.sum(initial_residuals**2) / 2),
                failure_reason=type(exc).__name__,
            ),
        )
    reached: list[str] = []
    if abs(result.x[6] - lower[6]) <= 1.0:
        reached.append("focal_lower_bound")
    if abs(result.x[6] - upper[6]) <= 1.0:
        reached.append("focal_upper_bound")
    final_residuals = _refinement_residuals(
        result.x,
        correspondences,
        intrinsics,
        intrinsics.focal_length_x_px,
    )
    return (
        result.x[:3].reshape(3, 1),
        result.x[3:6].reshape(3, 1),
        float(result.x[6]),
        RefinementDiagnostics(
            attempted=True,
            converged=bool(result.success),
            method="scipy_least_squares",
            robust_loss="soft_l1",
            initial_cost=float(np.sum(initial_residuals**2) / 2),
            final_cost=float(np.sum(final_residuals**2) / 2),
            iterations=int(result.nfev),
            parameters_changed=["rotation", "translation", "focal_length"],
            parameters_reaching_bounds=reached,
            residual_breakdown={
                "weighted_total": float(np.sqrt(np.mean(final_residuals**2))),
                "focal_prior": abs(
                    float(result.x[6] - intrinsics.focal_length_x_px)
                )
                / max(intrinsics.focal_length_x_px, 1.0),
            },
        ),
    )


def _box_iou(first: PixelBox, second: PixelBox) -> float:
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.x + first.width, second.x + second.width)
    bottom = min(first.y + first.height, second.y + second.height)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = (
        first.width * first.height
        + second.width * second.height
        - intersection
    )
    return intersection / max(union, 1e-9)


def _projected_wicket_box(
    end: str,
    rotation: np.ndarray,
    translation: np.ndarray,
    camera_matrix: np.ndarray,
) -> PixelBox | None:
    specification = build_virtual_pitch_specification()
    stumps = [item for item in specification.stumps if item.end == end]
    points: list[list[float]] = []
    for stump in stumps:
        points.extend(
            [
                [
                    stump.centre.x,
                    stump.centre.y,
                    stump.centre.z - stump.height_m / 2,
                ],
                [
                    stump.centre.x,
                    stump.centre.y,
                    stump.centre.z + stump.height_m / 2,
                ],
            ]
        )
    world = np.asarray(points, dtype=np.float64)
    projected, depths = _project(world, rotation, translation, camera_matrix)
    if np.any(depths <= 0) or not np.isfinite(projected).all():
        return None
    x1, y1 = np.min(projected, axis=0)
    x2, y2 = np.max(projected, axis=0)
    if x2 <= x1 or y2 <= y1:
        return None
    return PixelBox(x=max(0.0, x1), y=max(0.0, y1), width=x2 - x1, height=y2 - y1)


def _camera_position(
    rotation: np.ndarray, translation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rotation_matrix, _ = cv2.Rodrigues(rotation)
    position = -rotation_matrix.T @ translation.reshape(3, 1)
    return rotation_matrix, position.reshape(3)


def _plausibility_checks(
    rotation: np.ndarray,
    translation: np.ndarray,
    focal: float,
    intrinsics: CameraIntrinsicsCandidate,
    correspondences: Sequence[RegistrationCorrespondence],
    assignment_hypothesis: str,
    image_width: int,
    image_height: int,
) -> tuple[list[PlausibilityCheck], np.ndarray, np.ndarray]:
    camera_matrix = _camera_matrix(intrinsics, focal)
    rotation_matrix, position = _camera_position(rotation, translation)
    world, _, _ = _point_arrays(correspondences)
    _, depths = _project(world, rotation, translation, camera_matrix)
    pitch_centre = np.asarray([0.0, 10.06, 0.0])
    forward = rotation_matrix.T @ np.asarray([0.0, 0.0, 1.0])
    to_pitch = pitch_centre - position
    to_pitch /= max(np.linalg.norm(to_pitch), 1e-9)
    near_end = "bowler" if assignment_hypothesis == "A" else "striker"
    far_end = "striker" if near_end == "bowler" else "bowler"
    near_box = _projected_wicket_box(
        near_end, rotation, translation, camera_matrix
    )
    far_box = _projected_wicket_box(
        far_end, rotation, translation, camera_matrix
    )
    size_order = (
        near_box is not None
        and far_box is not None
        and near_box.height >= far_box.height * 0.85
    )
    camera_distance = float(np.linalg.norm(position - pitch_centre))
    checks = [
        PlausibilityCheck(
            check_id="finite_pose",
            passed=bool(
                np.isfinite(position).all()
                and np.isfinite(rotation_matrix).all()
                and np.isfinite(translation).all()
            ),
            reason="Pose matrices and camera centre must be finite.",
        ),
        PlausibilityCheck(
            check_id="camera_above_pitch",
            passed=0.2 <= float(position[2]) <= 20.0,
            value=float(position[2]),
            reason="Broad cricket-camera height bound is 0.2-20 m.",
        ),
        PlausibilityCheck(
            check_id="all_anchors_in_front",
            passed=bool(len(depths) > 0 and np.all(depths > 0.05)),
            value=float(np.min(depths)) if len(depths) else None,
            reason="All used wicket anchors must be in front of the camera.",
        ),
        PlausibilityCheck(
            check_id="camera_faces_pitch",
            passed=float(forward @ to_pitch) > 0.15,
            value=float(forward @ to_pitch),
            reason="Optical axis must face the pitch volume.",
        ),
        PlausibilityCheck(
            check_id="camera_distance",
            passed=0.5 <= camera_distance <= 150.0,
            value=camera_distance,
            reason="Camera must remain within broad non-setup-specific bounds.",
        ),
        PlausibilityCheck(
            check_id="focal_length_plausible",
            passed=(
                intrinsics.lower_focal_bound_px
                <= focal
                <= intrinsics.upper_focal_bound_px
            ),
            value=focal,
            reason="Focal length must remain within deterministic image bounds.",
        ),
        PlausibilityCheck(
            check_id="near_far_projected_size_order",
            passed=size_order,
            value=(
                None
                if near_box is None or far_box is None
                else near_box.height / max(far_box.height, 1e-9)
            ),
            reason="The perspective-near wicket should not project materially smaller.",
        ),
        PlausibilityCheck(
            check_id="projected_wickets_reasonable",
            passed=bool(
                near_box is not None
                and far_box is not None
                and near_box.width < image_width * 0.8
                and near_box.height < image_height * 0.8
                and far_box.width < image_width * 0.8
                and far_box.height < image_height * 0.8
            ),
            reason="Projected wicket envelopes must be finite and scene-sized.",
        ),
    ]
    return checks, rotation_matrix, position


def _edge_support(
    frame: np.ndarray,
    projected_lines: Sequence[Any],
) -> float | None:
    if frame.size == 0:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    height, width = edges.shape
    samples: list[float] = []
    for line in projected_lines:
        if (
            line.line_category
            not in {"bowling_crease", "popping_crease", "pitch_boundary"}
            or not line.projection_valid
            or line.pixel_start is None
            or line.pixel_end is None
        ):
            continue
        xs = np.linspace(line.pixel_start.x, line.pixel_end.x, 40)
        ys = np.linspace(line.pixel_start.y, line.pixel_end.y, 40)
        hits = 0
        valid = 0
        for x, y in zip(xs, ys, strict=True):
            ix, iy = int(round(x)), int(round(y))
            if not (0 <= ix < width and 0 <= iy < height):
                continue
            valid += 1
            patch = edges[
                max(0, iy - 3) : min(height, iy + 4),
                max(0, ix - 3) : min(width, ix + 4),
            ]
            hits += int(np.any(patch))
        if valid:
            samples.append(hits / valid)
    return float(sum(samples) / len(samples)) if samples else None


def _independent_validation(
    projection: Any,
    frame: np.ndarray,
    correspondences: Sequence[RegistrationCorrespondence],
    rotation: np.ndarray,
    translation: np.ndarray,
    camera_matrix: np.ndarray,
    rmse: float,
    checks: Sequence[PlausibilityCheck],
) -> IndependentValidation:
    envelope_scores: list[float] = []
    for item in correspondences:
        if item.mapping_type != "WICKET_ENVELOPE" or item.observed_bbox is None:
            continue
        end = (
            "bowler"
            if "bowler" in (item.virtual_semantic_id or "")
            else "striker"
        )
        projected = _projected_wicket_box(
            end, rotation, translation, camera_matrix
        )
        if projected is not None:
            envelope_scores.append(_box_iou(item.observed_bbox, projected))
    envelope_score = (
        sum(envelope_scores) / len(envelope_scores)
        if envelope_scores
        else 0.0
    )
    crease_support = _edge_support(frame, projection.projected_line_segments)
    anchor_score = math.exp(-rmse / 8.0)
    geometry_score = sum(item.passed for item in checks) / max(len(checks), 1)
    perspective_score = (
        1.0 if projection.diagnostics.perspective_order_valid else 0.0
    )
    independent_score = (
        0.55 * envelope_score
        + 0.30 * (crease_support if crease_support is not None else 0.0)
        + 0.15 * perspective_score
    )
    warnings: list[str] = []
    if crease_support is None:
        warnings.append("Projected crease support was unavailable.")
    elif crease_support < 0.15:
        warnings.append("Projected creases have weak image-edge support.")
    return IndependentValidation(
        anchor_fit_score=_clamp(anchor_score),
        independent_scene_score=_clamp(independent_score),
        geometry_plausibility_score=_clamp(geometry_score),
        projected_wicket_envelope_score=_clamp(envelope_score),
        crease_edge_support_score=(
            _clamp(crease_support) if crease_support is not None else None
        ),
        perspective_convergence_score=perspective_score,
        evidence=[
            "wicket_envelopes_not_used_as_exact_pnp_points",
            "crease_edge_support_is_heuristic_validation_only",
            "virtual_pitch_perspective_depth_order",
        ],
        warnings=warnings,
    )


def _temporal_validation(
    observation: WicketObservationResult,
    correspondences: Sequence[RegistrationCorrespondence],
    rotation: np.ndarray,
    translation: np.ndarray,
    camera_matrix: np.ndarray,
) -> TemporalValidation:
    assignment_by_role = {
        item.observed_wicket_role: (
            "bowler"
            if "bowler" in (item.virtual_semantic_id or "")
            else "striker"
        )
        for item in correspondences
        if item.mapping_type == "WICKET_ENVELOPE"
    }
    scores: list[float] = []
    evaluated: set[int] = set()
    for role, end in assignment_by_role.items():
        projected = _projected_wicket_box(
            end, rotation, translation, camera_matrix
        )
        if projected is None:
            continue
        wicket = observation.near_wicket if role == "near" else observation.far_wicket
        if wicket is None:
            continue
        for frame_id in wicket.region.supporting_frame_ids:
            raw = [
                item
                for item in observation.diagnostics.raw_detections
                if item.frame_index == frame_id
            ]
            if not raw:
                continue
            best = max(_box_iou(projected, item.bbox) for item in raw)
            scores.append(best)
            evaluated.add(frame_id)
    mean = sum(scores) / len(scores) if scores else None
    minimum = min(scores) if scores else None
    stability = _clamp((mean or 0.0) * min(1.0, len(evaluated) / 4))
    return TemporalValidation(
        supporting_frame_count=len(evaluated),
        evaluated_frame_ids=sorted(evaluated),
        mean_wicket_alignment_iou=mean,
        minimum_wicket_alignment_iou=minimum,
        stability_score=stability,
        warnings=(
            []
            if scores
            else ["No supporting-frame raw detector boxes could validate the pose."]
        ),
    )


def _reprojection_diagnostics(
    correspondences: Sequence[RegistrationCorrespondence],
    rotation: np.ndarray,
    translation: np.ndarray,
    camera_matrix: np.ndarray,
    inlier_indexes: Sequence[int],
) -> tuple[list[ReprojectionResidual], float, float, float, list[str], list[str]]:
    world, image, used = _point_arrays(correspondences)
    projected, _ = _project(world, rotation, translation, camera_matrix)
    errors = np.linalg.norm(projected - image, axis=1)
    inlier_set = {int(index) for index in inlier_indexes}
    diagnostics: list[ReprojectionResidual] = []
    for index, (item, observed, predicted, error) in enumerate(
        zip(used, image, projected, errors, strict=True)
    ):
        diagnostics.append(
            ReprojectionResidual(
                correspondence_id=item.correspondence_id,
                observed_pixel=PixelPoint(x=observed[0], y=observed[1]),
                projected_pixel=PixelPoint(x=predicted[0], y=predicted[1]),
                residual_px=float(error),
                weighted_residual=float(
                    error * math.sqrt(max(item.registration_weight, 1e-9))
                ),
                inlier=index in inlier_set,
            )
        )
    rmse = float(np.sqrt(np.mean(errors**2)))
    median_error = float(np.median(errors))
    inlier_errors = [
        float(errors[index])
        for index in inlier_set
        if 0 <= index < len(errors)
    ]
    maximum = max(inlier_errors) if inlier_errors else float(np.max(errors))
    inliers = [
        item.correspondence_id
        for index, item in enumerate(used)
        if index in inlier_set
    ]
    outliers = [
        item.correspondence_id
        for index, item in enumerate(used)
        if index not in inlier_set
    ]
    return diagnostics, rmse, median_error, maximum, inliers, outliers


def _failed_candidate(
    candidate_id: str,
    assignment: str,
    lateral_mapping: str,
    setup_frame_index: int,
    intrinsics: CameraIntrinsicsCandidate,
    reason: str,
) -> CameraPoseCandidate:
    return CameraPoseCandidate(
        candidate_id=candidate_id,
        assignment_hypothesis=assignment,
        near_semantic_end="bowler" if assignment == "A" else "striker",
        far_semantic_end="striker" if assignment == "A" else "bowler",
        lateral_mapping=lateral_mapping,
        setup_frame_index=setup_frame_index,
        intrinsics=intrinsics,
        anchor_subset_id="all_eligible_pointlike",
        attempted=True,
        solver_success=False,
        pnp_method="solvePnPRansac_EPNP",
        refinement=RefinementDiagnostics(
            attempted=False,
            converged=False,
            method="not_run",
            robust_loss="soft_l1",
            failure_reason=reason,
        ),
        score=0.0,
        classification="REGISTRATION_FAILED",
        eligible_for_selection=False,
        failure_reasons=[reason],
    )


def classify_pose_candidate(
    *,
    failed_hard_checks: Sequence[str],
    score: float,
    reprojection_rmse_px: float,
    exact_anchor_count: int,
    vertical_anchor_count: int,
    temporal_stability_score: float,
    independent_scene_score: float,
    wicket_envelope_score: float,
) -> str:
    if failed_hard_checks:
        return "REGISTRATION_FAILED"
    if (
        score >= 0.76
        and reprojection_rmse_px <= 5.0
        and exact_anchor_count >= 4
        and vertical_anchor_count >= 4
        and temporal_stability_score >= 0.55
        and independent_scene_score >= 0.35
    ):
        return "METRIC_3D_CANDIDATE"
    if (
        score >= 0.55
        and reprojection_rmse_px <= 9.0
        and wicket_envelope_score >= 0.25
    ):
        return "GROUND_PLANE_CANDIDATE"
    return "VISUAL_ONLY"


def resolve_registration_status(
    *,
    selected_classification: str,
    selected_score: float,
    competing_score: float | None,
    uncertainty_stable: bool,
) -> tuple[str, float]:
    score_gap = (
        selected_score - competing_score
        if competing_score is not None
        else 1.0
    )
    ambiguity = _clamp(1.0 - score_gap / AMBIGUITY_SCORE_GAP)
    if competing_score is not None and score_gap < AMBIGUITY_SCORE_GAP:
        return "AMBIGUOUS", ambiguity
    if (
        selected_classification == "METRIC_3D_CANDIDATE"
        and not uncertainty_stable
    ):
        return "GROUND_PLANE_CANDIDATE", ambiguity
    return selected_classification, ambiguity


def solve_pose_candidate(
    candidate_id: str,
    assignment: str,
    lateral_mapping: str,
    setup_frame_index: int,
    intrinsics: CameraIntrinsicsCandidate,
    correspondences: list[RegistrationCorrespondence],
    frame: np.ndarray,
    observation: WicketObservationResult,
    *,
    minimum_pointlike_anchors: int = MIN_POINTLIKE_ANCHORS,
) -> CameraPoseCandidate:
    world, image, used = _point_arrays(correspondences)
    if len(used) < minimum_pointlike_anchors:
        return _failed_candidate(
            candidate_id,
            assignment,
            lateral_mapping,
            setup_frame_index,
            intrinsics,
            "insufficient_pointlike_anchors",
        )
    if len({item.observed_wicket_role for item in used}) < 2:
        return _failed_candidate(
            candidate_id,
            assignment,
            lateral_mapping,
            setup_frame_index,
            intrinsics,
            "anchors_do_not_cover_both_wickets",
        )
    centred = world - np.mean(world, axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    if len(singular) < 3 or singular[2] <= max(1e-8, singular[0] * 1e-6):
        return _failed_candidate(
            candidate_id,
            assignment,
            lateral_mapping,
            setup_frame_index,
            intrinsics,
            "degenerate_world_geometry",
        )
    camera_matrix = _camera_matrix(intrinsics)
    threshold = _clamp(
        float(np.median([max(item.uncertainty_px, 1.0) for item in used])),
        3.0,
        14.0,
    )
    try:
        success, rotation, translation, inliers = cv2.solvePnPRansac(
            world,
            image,
            camera_matrix,
            np.zeros(5, dtype=np.float64),
            flags=cv2.SOLVEPNP_EPNP,
            iterationsCount=PNP_RANSAC_ITERATIONS,
            reprojectionError=threshold,
            confidence=PNP_CONFIDENCE,
        )
    except cv2.error:
        success, rotation, translation, inliers = False, None, None, None
    if (
        not success
        and minimum_pointlike_anchors == MIN_ASSISTED_POINTLIKE_ANCHORS
    ):
        try:
            success, rotation, translation = cv2.solvePnP(
                world,
                image,
                camera_matrix,
                np.zeros(5, dtype=np.float64),
                flags=cv2.SOLVEPNP_SQPNP,
            )
            inliers = (
                np.arange(len(used), dtype=np.int32).reshape(-1, 1)
                if success
                else None
            )
        except cv2.error:
            success, rotation, translation, inliers = (
                False,
                None,
                None,
                None,
            )
    if not success or rotation is None or translation is None:
        return _failed_candidate(
            candidate_id,
            assignment,
            lateral_mapping,
            setup_frame_index,
            intrinsics,
            "solvePnPRansac_failed",
        )
    rotation, translation, focal, refinement = _refine_pose(
        rotation, translation, correspondences, intrinsics
    )
    refined_intrinsics = intrinsics.model_copy(
        update={
            "focal_length_x_px": focal,
            "focal_length_y_px": focal,
            "horizontal_fov_degrees": math.degrees(
                2.0 * math.atan(frame.shape[1] / (2.0 * focal))
            ),
            "focal_bound_reached": bool(
                refinement.parameters_reaching_bounds
            ),
        }
    )
    camera_matrix = _camera_matrix(refined_intrinsics)
    inlier_indexes = (
        [int(value) for value in inliers.reshape(-1)]
        if inliers is not None
        else list(range(len(used)))
    )
    (
        residuals,
        rmse,
        median_error,
        maximum_error,
        inlier_ids,
        outlier_ids,
    ) = _reprojection_diagnostics(
        correspondences,
        rotation,
        translation,
        camera_matrix,
        inlier_indexes,
    )
    checks, rotation_matrix, position = _plausibility_checks(
        rotation,
        translation,
        focal,
        refined_intrinsics,
        correspondences,
        assignment,
        frame.shape[1],
        frame.shape[0],
    )
    camera = _virtual_camera(
        candidate_id,
        frame.shape[1],
        frame.shape[0],
        refined_intrinsics,
        rotation,
        translation,
        rotation_matrix,
        position,
    )
    projection = project_virtual_pitch(camera)
    independent = _independent_validation(
        projection,
        frame,
        correspondences,
        rotation,
        translation,
        camera_matrix,
        rmse,
        checks,
    )
    temporal = _temporal_validation(
        observation,
        correspondences,
        rotation,
        translation,
        camera_matrix,
    )
    hard_checks = {
        "finite_pose",
        "camera_above_pitch",
        "all_anchors_in_front",
        "camera_faces_pitch",
        "camera_distance",
        "focal_length_plausible",
        "projected_wickets_reasonable",
    }
    failed_hard = [
        item.check_id
        for item in checks
        if item.check_id in hard_checks and not item.passed
    ]
    score = _clamp(
        0.34 * independent.anchor_fit_score
        + 0.22 * independent.projected_wicket_envelope_score
        + 0.16 * independent.geometry_plausibility_score
        + 0.12 * independent.independent_scene_score
        + 0.10 * temporal.stability_score
        + 0.06 * (len(inlier_ids) / max(len(used), 1))
    )
    exact_count = sum(
        item.exactness == "EXACT" and item.status == "USED"
        for item in correspondences
    )
    vertical_count = sum(
        item.status == "USED" and "top" in item.observed_semantic_id
        for item in correspondences
    )
    classification = classify_pose_candidate(
        failed_hard_checks=failed_hard,
        score=score,
        reprojection_rmse_px=rmse,
        exact_anchor_count=exact_count,
        vertical_anchor_count=vertical_count,
        temporal_stability_score=temporal.stability_score,
        independent_scene_score=independent.independent_scene_score,
        wicket_envelope_score=independent.projected_wicket_envelope_score,
    )
    warnings = [
        "Candidate only; metric physics remains locked.",
        "Coarse wicket intersections are uncertainty-weighted pointlike anchors.",
    ]
    if refined_intrinsics.focal_bound_reached:
        warnings.append("Refined focal length reached a configured bound.")
    return CameraPoseCandidate(
        candidate_id=candidate_id,
        assignment_hypothesis=assignment,
        near_semantic_end="bowler" if assignment == "A" else "striker",
        far_semantic_end="striker" if assignment == "A" else "bowler",
        lateral_mapping=lateral_mapping,
        setup_frame_index=setup_frame_index,
        intrinsics=refined_intrinsics,
        anchor_subset_id="all_eligible_pointlike",
        attempted=True,
        solver_success=True,
        pnp_method="solvePnPRansac_EPNP",
        refinement=refinement,
        rotation_vector=rotation.reshape(-1).tolist(),
        translation_vector=translation.reshape(-1).tolist(),
        rotation_matrix=rotation_matrix.tolist(),
        camera_world_position=position.tolist(),
        inlier_correspondence_ids=inlier_ids,
        outlier_correspondence_ids=outlier_ids,
        reprojection_residuals=residuals,
        reprojection_rmse_px=rmse,
        median_reprojection_error_px=median_error,
        maximum_inlier_error_px=maximum_error,
        plausibility_checks=checks,
        independent_validation=independent,
        temporal_validation=temporal,
        score=score,
        classification=classification,
        eligible_for_selection=classification != "REGISTRATION_FAILED",
        failure_reasons=failed_hard,
        warnings=warnings,
    )


def _virtual_camera(
    name: str,
    image_width: int,
    image_height: int,
    intrinsics: CameraIntrinsicsCandidate,
    rotation: np.ndarray,
    translation: np.ndarray,
    rotation_matrix: np.ndarray,
    position: np.ndarray,
) -> VirtualCamera:
    horizontal_fov = math.degrees(
        2 * math.atan(image_width / (2 * intrinsics.focal_length_x_px))
    )
    target = np.asarray([0.0, 10.06, 0.0])
    return VirtualCamera(
        name=name,
        description="Real-video camera-pose candidate; not accepted.",
        image_width=image_width,
        image_height=image_height,
        camera_matrix=_camera_matrix(intrinsics).tolist(),
        distortion_coefficients=intrinsics.distortion_coefficients,
        rotation_vector=rotation.reshape(-1).tolist(),
        rotation_matrix=rotation_matrix.tolist(),
        translation_vector=translation.reshape(-1).tolist(),
        camera_position_world=position.tolist(),
        target_world=target.tolist(),
        near_m=0.05,
        far_m=150.0,
        horizontal_fov_degrees=horizontal_fov,
    )


def _candidate_camera(candidate: CameraPoseCandidate, width: int, height: int):
    rotation = np.asarray(candidate.rotation_vector, dtype=np.float64).reshape(3, 1)
    translation = np.asarray(candidate.translation_vector, dtype=np.float64).reshape(3, 1)
    rotation_matrix = np.asarray(candidate.rotation_matrix, dtype=np.float64)
    position = np.asarray(candidate.camera_world_position, dtype=np.float64)
    return _virtual_camera(
        candidate.candidate_id,
        width,
        height,
        candidate.intrinsics,
        rotation,
        translation,
        rotation_matrix,
        position,
    )


def _perturbation_uncertainty(
    candidate: CameraPoseCandidate,
    correspondences: Sequence[RegistrationCorrespondence],
    width: int,
    height: int,
    *,
    minimum_pointlike_anchors: int = MIN_POINTLIKE_ANCHORS,
) -> PoseUncertainty:
    world, image, used = _point_arrays(correspondences)
    if (
        not candidate.solver_success
        or len(used) < minimum_pointlike_anchors
    ):
        return PoseUncertainty(
            perturbation_count=0,
            deterministic_seed=PERTURBATION_SEED,
            stable_for_future_metric_use=False,
            warnings=["Pose perturbation was unavailable."],
        )
    rng = np.random.default_rng(PERTURBATION_SEED)
    camera_matrix = _camera_matrix(candidate.intrinsics)
    positions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    overlay_points: list[np.ndarray] = []
    specification = build_virtual_pitch_specification()
    probe_world = np.asarray(
        [
            [-specification.dimensions.pitch_width_m / 2, 0.0, 0.0],
            [specification.dimensions.pitch_width_m / 2, 0.0, 0.0],
            [-specification.dimensions.pitch_width_m / 2, 20.12, 0.0],
            [specification.dimensions.pitch_width_m / 2, 20.12, 0.0],
            [0.0, 10.06, 0.0],
        ],
        dtype=np.float64,
    )
    base_rotation = np.asarray(candidate.rotation_vector, dtype=np.float64).reshape(3, 1)
    base_translation = np.asarray(candidate.translation_vector, dtype=np.float64).reshape(3, 1)
    for _ in range(PERTURBATION_COUNT):
        perturbed = image.copy()
        for index, item in enumerate(used):
            angle = rng.uniform(0, 2 * math.pi)
            radius = rng.uniform(0.25, 0.75) * max(item.uncertainty_px, 1.0)
            perturbed[index] += [math.cos(angle) * radius, math.sin(angle) * radius]
        try:
            success, rotation, translation = cv2.solvePnP(
                world,
                perturbed,
                camera_matrix,
                np.zeros(5, dtype=np.float64),
                base_rotation.copy(),
                base_translation.copy(),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            continue
        if not success:
            continue
        rotation_matrix, position = _camera_position(rotation, translation)
        projected, depths = _project(
            probe_world, rotation, translation, camera_matrix
        )
        if np.all(depths > 0) and np.isfinite(projected).all():
            positions.append(position)
            rotations.append(rotation_matrix)
            overlay_points.append(projected)
    if len(positions) < 3:
        return PoseUncertainty(
            perturbation_count=len(positions),
            deterministic_seed=PERTURBATION_SEED,
            stable_for_future_metric_use=False,
            warnings=["Too few perturbation solves remained physically valid."],
        )
    position_array = np.asarray(positions)
    position_spread = float(
        np.max(np.linalg.norm(position_array - np.median(position_array, axis=0), axis=1))
    )
    base_matrix, _ = cv2.Rodrigues(base_rotation)
    rotation_errors: list[float] = []
    for matrix in rotations:
        relative = matrix @ base_matrix.T
        cosine = _clamp((np.trace(relative) - 1) / 2, -1.0, 1.0)
        rotation_errors.append(math.degrees(math.acos(cosine)))
    overlay_array = np.asarray(overlay_points)
    median_overlay = np.median(overlay_array, axis=0)
    overlay_spread = float(
        np.max(np.linalg.norm(overlay_array - median_overlay, axis=2))
    )
    stable = (
        position_spread <= 1.5
        and max(rotation_errors) <= 3.0
        and overlay_spread <= max(12.0, math.hypot(width, height) * 0.015)
    )
    return PoseUncertainty(
        perturbation_count=len(positions),
        deterministic_seed=PERTURBATION_SEED,
        camera_position_spread_m=position_spread,
        rotation_spread_degrees=max(rotation_errors),
        focal_length_spread_px=0.0,
        maximum_overlay_movement_px=overlay_spread,
        projected_bounce_location_sensitivity_px=float(
            np.max(
                np.linalg.norm(
                    overlay_array[:, -1, :] - median_overlay[-1, :],
                    axis=1,
                )
            )
        ),
        stable_for_future_metric_use=stable,
        warnings=(
            []
            if stable
            else ["Pose is sensitive to landmark uncertainty perturbations."]
        ),
    )


def build_pose_stability_diagnostics(
    candidate: CameraPoseCandidate,
    correspondences: Sequence[RegistrationCorrespondence],
    width: int,
    height: int,
    *,
    minimum_pointlike_anchors: int = MIN_POINTLIKE_ANCHORS,
) -> dict[str, object]:
    """Return per-candidate pose/stability diagnostics for debug reporting."""
    uncertainty = candidate.uncertainty
    if uncertainty is None and candidate.solver_success:
        uncertainty = _perturbation_uncertainty(
            candidate,
            correspondences,
            width,
            height,
            minimum_pointlike_anchors=minimum_pointlike_anchors,
        )
    temporal = candidate.temporal_validation
    zero_reason = None
    if temporal is not None and temporal.stability_score == 0.0:
        if not temporal.evaluated_frame_ids:
            zero_reason = "no_supporting_frame_raw_detections_for_temporal_iou"
        elif temporal.warnings:
            zero_reason = temporal.warnings[0]
    spreads = (
        uncertainty.camera_position_spread_m if uncertainty else None,
        uncertainty.rotation_spread_degrees if uncertainty else None,
        uncertainty.maximum_overlay_movement_px if uncertainty else None,
    )
    if any(value is not None and not math.isfinite(value) for value in spreads):
        stability_failure = "Stability calculation produced a non-finite value"
    elif uncertainty is None:
        stability_failure = (
            "Automatic stump landmarks are valid, but camera-stability "
            "validation encountered an internal error."
        )
    elif uncertainty.perturbation_count == 0:
        stability_failure = "PnP perturbation trials failed"
    elif not uncertainty.stable_for_future_metric_use:
        stability_failure = (
            "Camera pose changes excessively under a one-pixel perturbation"
        )
    else:
        stability_failure = None
    return {
        "candidate_id": candidate.candidate_id,
        "baseline": {
            "rotation_vector": candidate.rotation_vector,
            "translation_vector": candidate.translation_vector,
            "focal_length_px": candidate.intrinsics.focal_length_x_px,
            "reprojection_rmse_px": candidate.reprojection_rmse_px,
        },
        "perturbation": (
            uncertainty.model_dump(mode="json") if uncertainty is not None else None
        ),
        "perturbation_trial_count": (
            uncertainty.perturbation_count if uncertainty is not None else 0
        ),
        "perturbation_trials_succeeded": (
            uncertainty.perturbation_count if uncertainty is not None else 0
        ),
        "temporal_stability_score": (
            temporal.stability_score if temporal is not None else None
        ),
        "temporal_zero_reason": zero_reason,
        "stability_failure_reason": stability_failure,
        "correspondence_count": len(
            [item for item in correspondences if item.status == "USED"]
        ),
    }


def _read_setup_frame(
    analysis_id: str,
    frame_index: int,
) -> tuple[np.ndarray, Any]:
    analysis = load_video_analysis(analysis_id)
    raw_path = (
        VIDEO_ANALYSIS_ROOT / analysis_id / "raw" / analysis.stored_filename
    )
    capture = cv2.VideoCapture(str(raw_path))
    try:
        if not capture.isOpened():
            raise VideoAnalysisServiceError(
                "Could not open clean video for camera registration.",
                status_code=422,
            )
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None or frame.size == 0:
        raise VideoAnalysisServiceError(
            "Selected setup frame could not be decoded from the clean video.",
            status_code=422,
        )
    return frame, analysis


def _real_projection(candidate: CameraPoseCandidate, width: int, height: int):
    camera = _candidate_camera(candidate, width, height)
    projected = project_virtual_pitch(camera)
    data = projected.model_dump(exclude={"synthetic_only"})
    return RealProjectedPitchGeometry.model_validate(
        {**data, "registered_to_real_setup_frame": True}
    )


def _draw_projection_overlay(
    frame: np.ndarray,
    projection: RealProjectedPitchGeometry,
    correspondences: Sequence[RegistrationCorrespondence],
    candidate: CameraPoseCandidate,
    output_path: Path,
) -> None:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    line_colours = {
        "pitch_boundary": (80, 220, 255, 210),
        "bowling_crease": (255, 210, 70, 220),
        "popping_crease": (255, 145, 70, 220),
        "return_crease": (190, 160, 255, 210),
        "centreline": (120, 235, 110, 190),
        "bail": (255, 255, 255, 220),
    }
    for line in [
        *projection.projected_line_segments,
        *projection.projected_bails,
    ]:
        if (
            not line.projection_valid
            or line.pixel_start is None
            or line.pixel_end is None
        ):
            continue
        draw.line(
            (
                line.pixel_start.x,
                line.pixel_start.y,
                line.pixel_end.x,
                line.pixel_end.y,
            ),
            fill=line_colours.get(line.line_category, (180, 220, 180, 180)),
            width=2,
        )
    for polygon in projection.projected_polygons:
        if polygon.polygon_category != "lbw_corridor":
            continue
        points = [
            (point.x, point.y)
            for point in polygon.pixel_vertices
            if point is not None
        ]
        if len(points) >= 3:
            draw.polygon(points, fill=(255, 190, 50, 35), outline=(255, 190, 50, 130))
    for stump in projection.projected_stumps:
        if (
            stump.projection_valid
            and stump.pixel_base is not None
            and stump.pixel_top is not None
        ):
            draw.line(
                (
                    stump.pixel_base.x,
                    stump.pixel_base.y,
                    stump.pixel_top.x,
                    stump.pixel_top.y,
                ),
                fill=(255, 255, 255, 235),
                width=3,
            )
    residual_by_id = {
        item.correspondence_id: item
        for item in candidate.reprojection_residuals
    }
    for item in correspondences:
        residual = residual_by_id.get(item.correspondence_id)
        if residual is None:
            continue
        colour = (120, 235, 110, 255) if residual.inlier else (255, 90, 80, 255)
        draw.line(
            (
                residual.observed_pixel.x,
                residual.observed_pixel.y,
                residual.projected_pixel.x,
                residual.projected_pixel.y,
            ),
            fill=colour,
            width=2,
        )
        x, y = residual.observed_pixel.x, residual.observed_pixel.y
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=colour)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, "JPEG", quality=93)


def _write_result(path: Path, result: RealPitchRegistrationResult) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            result.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _not_attempted(
    analysis_id: str,
    observation: WicketObservationResult,
    reasons: list[str],
    *,
    result_filename: str = RESULT_FILENAME,
) -> RealPitchRegistrationResult:
    result = RealPitchRegistrationResult(
        analysis_id=analysis_id,
        status="NOT_ATTEMPTED",
        attempted=False,
        setup_frame=observation.setup_frame,
        supporting_frames=observation.supporting_frames,
        wicket_observation_source=(
            f"/static/video-analysis/{analysis_id}/reports/{WICKET_RESULT_FILENAME}"
        ),
        ambiguity_score=0.0,
        metrics_locked=True,
        acceptance_required=True,
        failure_reasons=reasons,
        warnings=[
            "Registration was not attempted because wicket evidence failed eligibility.",
            "Physics V1 remains image-space only.",
        ],
        diagnostics=RegistrationDiagnostics(
            result_json_url=(
            f"/static/video-analysis/{analysis_id}/reports/{result_filename}"
            ),
            eligibility_reasons=reasons,
        ),
        message="Real camera registration was not attempted.",
    )
    destination = VIDEO_ANALYSIS_ROOT / analysis_id / "reports" / result_filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_result(destination, result)
    return result


def run_real_pitch_registration(
    analysis_id: str,
    *,
    point_overrides: dict[str, PixelPoint] | None = None,
    crease_overrides: dict[str, PixelPoint] | None = None,
    manual_override_ids: set[str] | None = None,
    required_lateral_mapping: str | None = None,
    result_filename: str = RESULT_FILENAME,
    debug_directory: str = DEBUG_DIRECTORY,
) -> RealPitchRegistrationResult:
    observation = load_wicket_observation(analysis_id)
    eligibility = check_registration_eligibility(observation)
    if not eligibility.eligible:
        return _not_attempted(
            analysis_id,
            observation,
            eligibility.reasons,
            result_filename=result_filename,
        )
    assert observation.setup_frame is not None
    frame, _ = _read_setup_frame(
        analysis_id, observation.setup_frame.frame_index
    )
    intrinsics_candidates = build_intrinsics_candidates(
        frame.shape[1], frame.shape[0]
    )
    candidates: list[CameraPoseCandidate] = []
    correspondences_by_candidate: dict[str, list[RegistrationCorrespondence]] = {}
    for assignment in ("A", "B"):
        for lateral_mapping in (
            "image_left_to_world_left",
            "image_left_to_world_right",
        ):
            correspondences = build_registration_correspondences(
                observation,
                assignment_hypothesis=assignment,
                lateral_mapping=lateral_mapping,
            )
            if point_overrides or crease_overrides:
                correspondences = apply_assisted_anchor_overrides(
                    correspondences,
                    assignment_hypothesis=assignment,
                    lateral_mapping=lateral_mapping,
                    point_overrides=point_overrides or {},
                    crease_overrides=crease_overrides,
                    manual_override_ids=manual_override_ids,
                )
            for intrinsics in intrinsics_candidates:
                candidate_id = (
                    f"{assignment}:{lateral_mapping}:{intrinsics.candidate_id}"
                )
                candidate = solve_pose_candidate(
                    candidate_id,
                    assignment,
                    lateral_mapping,
                    observation.setup_frame.frame_index,
                    intrinsics,
                    correspondences,
                    frame,
                    observation,
                    minimum_pointlike_anchors=(
                        MIN_ASSISTED_POINTLIKE_ANCHORS
                        if manual_override_ids
                        else MIN_POINTLIKE_ANCHORS
                    ),
                )
                candidates.append(candidate)
                correspondences_by_candidate[candidate_id] = correspondences
    if required_lateral_mapping is not None:
        if required_lateral_mapping not in {
            "image_left_to_world_left",
            "image_left_to_world_right",
        }:
            raise VideoAnalysisServiceError(
                "Unsupported lateral orientation mapping.",
                status_code=422,
            )
        candidates = [
            item
            if item.lateral_mapping == required_lateral_mapping
            else item.model_copy(
                update={
                    "score": 0.0,
                    "eligible_for_selection": False,
                    "failure_reasons": [
                        *item.failure_reasons,
                        "rejected_by_explicit_orientation_evidence",
                    ],
                    "warnings": [
                        *item.warnings,
                        (
                            "Candidate retained for diagnostics but rejected "
                            "by explicit native-video orientation evidence."
                        ),
                    ],
                }
            )
            for item in candidates
        ]
    ranked = sorted(
        [item for item in candidates if item.eligible_for_selection],
        key=lambda item: (-item.score, item.candidate_id),
    )
    if not ranked:
        status = "REGISTRATION_FAILED"
        selected = None
        competing = None
        ambiguity = 0.0
        projection = None
        competing_projection = None
        selected_correspondences: list[RegistrationCorrespondence] = []
        failure_reasons = sorted(
            {
                reason
                for candidate in candidates
                for reason in candidate.failure_reasons
            }
        )
    else:
        selected = ranked[0]
        competing = ranked[1] if len(ranked) > 1 else None
        selected_correspondences = correspondences_by_candidate[
            selected.candidate_id
        ]
        selected = selected.model_copy(
            update={
                "uncertainty": _perturbation_uncertainty(
                    selected,
                    selected_correspondences,
                    frame.shape[1],
                    frame.shape[0],
                    minimum_pointlike_anchors=(
                        MIN_ASSISTED_POINTLIKE_ANCHORS
                        if manual_override_ids
                        else MIN_POINTLIKE_ANCHORS
                    ),
                )
            }
        )
        candidates = [
            selected if item.candidate_id == selected.candidate_id else item
            for item in candidates
        ]
        competing_assignment = next(
            (
                item
                for item in ranked[1:]
                if item.assignment_hypothesis
                != selected.assignment_hypothesis
            ),
            competing,
        )
        competing = competing_assignment
        status, ambiguity = resolve_registration_status(
            selected_classification=selected.classification,
            selected_score=selected.score,
            competing_score=competing.score if competing is not None else None,
            uncertainty_stable=(
                selected.uncertainty is not None
                and selected.uncertainty.stable_for_future_metric_use
            ),
        )
        projection = _real_projection(
            selected, frame.shape[1], frame.shape[0]
        )
        competing_projection = (
            _real_projection(competing, frame.shape[1], frame.shape[0])
            if competing is not None and competing.solver_success
            else None
        )
        failure_reasons = []

    analysis_dir = VIDEO_ANALYSIS_ROOT / analysis_id
    debug_dir = analysis_dir / "calibration" / debug_directory
    debug_dir.mkdir(parents=True, exist_ok=True)
    setup_filename = "setup_frame.jpg"
    cv2.imwrite(str(debug_dir / setup_filename), frame)
    overlay_url = None
    alternate_overlay_url = None
    if selected is not None and projection is not None:
        overlay_filename = "projected_pitch_overlay.jpg"
        _draw_projection_overlay(
            frame,
            projection,
            selected_correspondences,
            selected,
            debug_dir / overlay_filename,
        )
        overlay_url = (
            f"/static/video-analysis/{analysis_id}/calibration/"
            f"{debug_directory}/{overlay_filename}"
        )
    if competing is not None and competing_projection is not None:
        alternate_filename = "alternate_assignment_overlay.jpg"
        _draw_projection_overlay(
            frame,
            competing_projection,
            correspondences_by_candidate[competing.candidate_id],
            competing,
            debug_dir / alternate_filename,
        )
        alternate_overlay_url = (
            f"/static/video-analysis/{analysis_id}/calibration/"
            f"{debug_directory}/{alternate_filename}"
        )
    warnings = [
        "Real camera pose candidate only; metric analytics remain locked.",
        "No registration candidate is accepted by this milestone.",
        "Principal point remains fixed at image centre and distortion is assumed zero.",
    ]
    if status == "AMBIGUOUS":
        warnings.append("Competing end assignments or camera poses score similarly.")
    result = RealPitchRegistrationResult(
        analysis_id=analysis_id,
        status=status,
        attempted=True,
        setup_frame=observation.setup_frame,
        supporting_frames=observation.supporting_frames,
        wicket_observation_source=(
            f"/static/video-analysis/{analysis_id}/reports/{WICKET_RESULT_FILENAME}"
        ),
        correspondences=selected_correspondences,
        candidates=candidates,
        selected_candidate=selected,
        competing_candidate=competing,
        ambiguity_score=ambiguity,
        projected_pitch_geometry=projection,
        competing_projected_pitch_geometry=competing_projection,
        warnings=warnings,
        metrics_locked=True,
        acceptance_required=True,
        failure_reasons=failure_reasons,
        diagnostics=RegistrationDiagnostics(
            setup_frame_image_url=(
                f"/static/video-analysis/{analysis_id}/calibration/"
                f"{debug_directory}/{setup_filename}"
            ),
            projected_overlay_url=overlay_url,
            anchor_residual_overlay_url=overlay_url,
            alternate_assignment_overlay_url=alternate_overlay_url,
            result_json_url=(
                f"/static/video-analysis/{analysis_id}/reports/{result_filename}"
            ),
            focal_candidate_count=len(intrinsics_candidates),
            pose_candidate_count=len(candidates),
            rejected_correspondence_count=sum(
                item.status in {"REJECTED", "UNAVAILABLE"}
                for item in selected_correspondences
            ),
        ),
        message=(
            "Real camera pose candidate generated; metric acceptance is still required."
            if selected is not None
            else "No physically acceptable real camera pose candidate was found."
        ),
    )
    reports = analysis_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    _write_result(reports / result_filename, result)
    return result


def load_real_pitch_registration(
    analysis_id: str,
) -> RealPitchRegistrationResult:
    load_video_analysis(analysis_id)
    path = VIDEO_ANALYSIS_ROOT / analysis_id / "reports" / RESULT_FILENAME
    if not path.is_file():
        raise VideoAnalysisServiceError(
            "Real pitch registration has not been generated for this analysis.",
            status_code=404,
        )
    try:
        return RealPitchRegistrationResult.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise VideoAnalysisServiceError(
            "Stored real pitch registration is unavailable.",
            status_code=500,
        ) from exc
