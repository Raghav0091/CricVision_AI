"""Exact conversion between CricVision preset parameters and OpenCV cameras.

Virtual Pitch V1 uses metres, +x to pitch right, +y bowler to striker, and
+z up. OpenCV extrinsics are world-to-camera: ``Xc = R @ Xw + t`` with camera
+x right, +y down, and +z forward.

Yaw and pitch define the optical axis, then roll rotates the camera right/down
basis about that axis. Positive yaw turns toward the camera-end viewer's right,
positive pitch looks upward, and positive roll turns image-right toward image-
down. The Euler-equivalent construction order is yaw, pitch, optical-axis roll.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .virtual_pitch_service import build_virtual_pitch_specification


_CAMERA_ENDS = {"bowler", "striker"}
_MAPPING_ALIASES = {
    "image_left_is_pitch_left": "image_left_to_world_left",
    "image_left_is_pitch_right": "image_left_to_world_right",
    "image_left_to_world_left": "image_left_to_world_left",
    "image_left_to_world_right": "image_left_to_world_right",
}
PARAMETER_NAMES = (
    "lateral_offset_m",
    "distance_behind_wicket_m",
    "camera_height_m",
    "yaw_deg",
    "pitch_deg",
    "roll_deg",
    "horizontal_fov_deg",
    "principal_point_offset_x_px",
    "principal_point_offset_y_px",
    "focal_y_over_x",
    "skew_px",
)


@dataclass(frozen=True)
class PresetCameraParameters:
    lateral_offset_m: float
    distance_behind_wicket_m: float
    camera_height_m: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    horizontal_fov_deg: float
    principal_point_offset_x_px: float = 0.0
    principal_point_offset_y_px: float = 0.0
    focal_y_over_x: float = 1.0
    skew_px: float = 0.0


@dataclass(frozen=True)
class OpenCVCamera:
    camera_matrix: np.ndarray
    rotation_matrix: np.ndarray
    rotation_vector: np.ndarray
    translation_vector: np.ndarray
    camera_position_world: np.ndarray
    forward_vector_world: np.ndarray
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class CameraDecomposition:
    parameters: PresetCameraParameters
    camera_position_world: np.ndarray
    forward_vector_world: np.ndarray
    diagnostics: Mapping[str, Any]
    bound_diagnostics: tuple[Mapping[str, Any], ...] = ()


def _normalise_camera_end(camera_end: str) -> str:
    value = str(camera_end).lower()
    if value not in _CAMERA_ENDS:
        raise ValueError("camera_end must be 'bowler' or 'striker'.")
    return value


def _normalise_mapping(image_left_mapping: str) -> str:
    value = _MAPPING_ALIASES.get(str(image_left_mapping).lower())
    if value is None:
        raise ValueError("Unsupported image_left_mapping.")
    return value


def validate_rotation_matrix(
    rotation_matrix: Sequence[Sequence[float]], *, tolerance: float = 1e-8
) -> np.ndarray:
    """Return a valid proper rotation or reject it; never silently repair it."""
    rotation = np.asarray(rotation_matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("Rotation matrix must be a finite 3x3 matrix.")
    error = float(np.linalg.norm(rotation @ rotation.T - np.eye(3), ord="fro"))
    determinant = float(np.linalg.det(rotation))
    if error > tolerance or abs(determinant - 1.0) > tolerance:
        raise ValueError(
            "Rotation matrix must be orthonormal with determinant +1 "
            f"(error={error:.3g}, determinant={determinant:.9g})."
        )
    return rotation


def _reference_basis(
    forward: np.ndarray, camera_end: str, image_left_mapping: str
) -> tuple[np.ndarray, np.ndarray]:
    world_up = np.asarray([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    norm = float(np.linalg.norm(right))
    if norm < 1e-10:
        raise ValueError("Preset parameterization is undefined for a vertical optical axis.")
    right /= norm

    # At zero yaw, bowler-end image-right naturally points +x and striker-end
    # image-right points -x. The alternate mapping is the same proper camera
    # frame rotated 180 degrees about its optical axis, never a reflection.
    natural_right_sign = 1.0 if camera_end == "bowler" else -1.0
    requested_right_sign = (
        1.0 if image_left_mapping == "image_left_to_world_left" else -1.0
    )
    if natural_right_sign != requested_right_sign:
        right = -right
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    return right, down


def build_opencv_camera_from_preset_parameters(
    parameters: PresetCameraParameters | Mapping[str, float],
    *,
    image_width: int,
    image_height: int,
    camera_end: str,
    image_left_mapping: str,
) -> OpenCVCamera:
    """Reconstruct an OpenCV world-to-camera model from preset parameters."""
    if isinstance(parameters, Mapping):
        parameters = PresetCameraParameters(**parameters)
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    end = _normalise_camera_end(camera_end)
    mapping = _normalise_mapping(image_left_mapping)
    values = np.asarray([getattr(parameters, name) for name in PARAMETER_NAMES])
    if not np.isfinite(values).all():
        raise ValueError("Preset parameters must be finite.")
    if not 0.0 < parameters.horizontal_fov_deg < 180.0:
        raise ValueError("horizontal_fov_deg must lie strictly between 0 and 180.")
    if parameters.camera_height_m <= 0 or parameters.distance_behind_wicket_m < 0:
        raise ValueError("Camera height must be positive and distance non-negative.")
    if parameters.focal_y_over_x <= 0:
        raise ValueError("focal_y_over_x must be positive.")

    pitch_length = build_virtual_pitch_specification().dimensions.pitch_length_m
    end_sign = 1.0 if end == "bowler" else -1.0
    end_y = 0.0 if end == "bowler" else pitch_length
    position = np.asarray(
        [
            parameters.lateral_offset_m,
            end_y - end_sign * parameters.distance_behind_wicket_m,
            parameters.camera_height_m,
        ],
        dtype=np.float64,
    )
    yaw = math.radians(parameters.yaw_deg)
    pitch = math.radians(parameters.pitch_deg)
    forward = np.asarray(
        [
            end_sign * math.sin(yaw) * math.cos(pitch),
            end_sign * math.cos(yaw) * math.cos(pitch),
            math.sin(pitch),
        ],
        dtype=np.float64,
    )
    forward /= np.linalg.norm(forward)
    reference_right, reference_down = _reference_basis(forward, end, mapping)
    roll = math.radians(parameters.roll_deg)
    right = math.cos(roll) * reference_right + math.sin(roll) * reference_down
    down = -math.sin(roll) * reference_right + math.cos(roll) * reference_down
    rotation = validate_rotation_matrix(np.vstack([right, down, forward]))
    rotation_vector, _ = cv2.Rodrigues(rotation)
    translation = -rotation @ position.reshape(3, 1)

    fx = image_width / (2.0 * math.tan(math.radians(parameters.horizontal_fov_deg) / 2.0))
    fy = fx * parameters.focal_y_over_x
    matrix = np.asarray(
        [
            [fx, parameters.skew_px, image_width / 2.0 + parameters.principal_point_offset_x_px],
            [0.0, fy, image_height / 2.0 + parameters.principal_point_offset_y_px],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return OpenCVCamera(
        camera_matrix=matrix,
        rotation_matrix=rotation,
        rotation_vector=rotation_vector.reshape(3),
        translation_vector=translation.reshape(3),
        camera_position_world=position,
        forward_vector_world=forward,
        diagnostics={
            "coordinate_convention": "opencv_world_to_camera_Xc_equals_R_Xw_plus_t",
            "rotation_order": "yaw_then_pitch_then_optical_axis_roll",
            "unequal_focal_lengths": not math.isclose(fx, fy, rel_tol=1e-12, abs_tol=1e-12),
            "skew_nonzero": not math.isclose(parameters.skew_px, 0.0, abs_tol=1e-12),
        },
    )


def decompose_opencv_camera_to_preset_parameters(
    *,
    camera_matrix: Sequence[Sequence[float]],
    translation_vector: Sequence[float],
    image_width: int,
    image_height: int,
    camera_end: str,
    image_left_mapping: str,
    rotation_matrix: Sequence[Sequence[float]] | None = None,
    rotation_vector: Sequence[float] | None = None,
    preset: Any | None = None,
) -> CameraDecomposition:
    """Decompose a valid OpenCV camera into the exact inverse preset model."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    end = _normalise_camera_end(camera_end)
    mapping = _normalise_mapping(image_left_mapping)
    if (rotation_matrix is None) == (rotation_vector is None):
        raise ValueError("Provide exactly one of rotation_matrix or rotation_vector.")
    if rotation_matrix is None:
        rvec = np.asarray(rotation_vector, dtype=np.float64).reshape(3, 1)
        if not np.isfinite(rvec).all():
            raise ValueError("Rotation vector must be finite.")
        rotation_matrix, _ = cv2.Rodrigues(rvec)
    rotation = validate_rotation_matrix(rotation_matrix)
    translation = np.asarray(translation_vector, dtype=np.float64).reshape(3)
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.isfinite(translation).all():
        raise ValueError("K and t must be finite with shapes 3x3 and 3.")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or abs(matrix[2, 2]) < 1e-12:
        raise ValueError("Camera focal lengths and homogeneous scale must be positive.")
    matrix = matrix / matrix[2, 2]
    if not np.allclose(matrix[1, 0], 0.0, atol=1e-10) or not np.allclose(matrix[2, :2], 0.0, atol=1e-10):
        raise ValueError("K must use the standard upper-triangular OpenCV form.")

    position = -rotation.T @ translation
    forward = rotation[2].copy()
    end_sign = 1.0 if end == "bowler" else -1.0
    horizontal_norm = math.hypot(float(forward[0]), float(forward[1]))
    if horizontal_norm < 1e-10:
        raise ValueError("Preset yaw is undefined for a vertical optical axis.")
    pitch = math.degrees(math.atan2(float(forward[2]), horizontal_norm))
    yaw = math.degrees(
        math.atan2(end_sign * float(forward[0]), end_sign * float(forward[1]))
    )
    reference_right, reference_down = _reference_basis(forward, end, mapping)
    roll = math.degrees(
        math.atan2(
            float(rotation[0] @ reference_down),
            float(rotation[0] @ reference_right),
        )
    )
    pitch_length = build_virtual_pitch_specification().dimensions.pitch_length_m
    end_y = 0.0 if end == "bowler" else pitch_length
    distance = end_sign * (end_y - float(position[1]))
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    parameters = PresetCameraParameters(
        lateral_offset_m=float(position[0]),
        distance_behind_wicket_m=distance,
        camera_height_m=float(position[2]),
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
        horizontal_fov_deg=math.degrees(2.0 * math.atan(image_width / (2.0 * fx))),
        principal_point_offset_x_px=float(matrix[0, 2] - image_width / 2.0),
        principal_point_offset_y_px=float(matrix[1, 2] - image_height / 2.0),
        focal_y_over_x=fy / fx,
        skew_px=float(matrix[0, 1]),
    )
    diagnostics = {
        "coordinate_convention": "opencv_world_to_camera_Xc_equals_R_Xw_plus_t",
        "rotation_order": "yaw_then_pitch_then_optical_axis_roll",
        "rotation_orthonormality_error": float(np.linalg.norm(rotation @ rotation.T - np.eye(3), ord="fro")),
        "rotation_determinant": float(np.linalg.det(rotation)),
        "gimbal_lock": horizontal_norm < 1e-6,
        "unequal_focal_lengths": not math.isclose(fx, fy, rel_tol=1e-12, abs_tol=1e-12),
        "focal_y_over_x": fy / fx,
        "skew_px": float(matrix[0, 1]),
    }
    bounds = tuple(compare_parameters_to_preset_bounds(parameters, preset)) if preset is not None else ()
    return CameraDecomposition(parameters, position, forward, diagnostics, bounds)


def pack_parameters(parameters: PresetCameraParameters) -> np.ndarray:
    return np.asarray([getattr(parameters, name) for name in PARAMETER_NAMES], dtype=np.float64)


def unpack_parameters(values: Sequence[float]) -> PresetCameraParameters:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != len(PARAMETER_NAMES) or not np.isfinite(array).all():
        raise ValueError(f"Expected {len(PARAMETER_NAMES)} finite parameter values.")
    return PresetCameraParameters(**dict(zip(PARAMETER_NAMES, array.tolist(), strict=True)))


def normalize_parameters(values: Sequence[float], lower: Sequence[float], upper: Sequence[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    if values_array.shape != lower_array.shape or values_array.shape != upper_array.shape:
        raise ValueError("Values and bounds must have identical shapes.")
    span = upper_array - lower_array
    if not np.isfinite(values_array).all() or not np.isfinite(span).all() or np.any(span <= 0):
        raise ValueError("Normalized-variable bounds must be finite and increasing.")
    return 2.0 * (values_array - lower_array) / span - 1.0


def denormalize_parameters(normalized: Sequence[float], lower: Sequence[float], upper: Sequence[float]) -> np.ndarray:
    normalized_array = np.asarray(normalized, dtype=np.float64)
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    if normalized_array.shape != lower_array.shape or normalized_array.shape != upper_array.shape:
        raise ValueError("Normalized values and bounds must have identical shapes.")
    span = upper_array - lower_array
    if not np.isfinite(normalized_array).all() or not np.isfinite(span).all() or np.any(span <= 0):
        raise ValueError("Normalized-variable bounds must be finite and increasing.")
    return lower_array + (normalized_array + 1.0) * span / 2.0


def _bound_pair(source: Any, field: str) -> tuple[float, float]:
    value = source[field] if isinstance(source, Mapping) else getattr(source, field)
    names = (("minimum_m", "maximum_m"), ("minimum_deg", "maximum_deg"), ("minimum", "maximum"))
    for low_name, high_name in names:
        low = value.get(low_name) if isinstance(value, Mapping) else getattr(value, low_name, None)
        high = value.get(high_name) if isinstance(value, Mapping) else getattr(value, high_name, None)
        if low is not None and high is not None:
            return float(low), float(high)
    raise ValueError(f"Cannot read preset bound {field}.")


def compare_parameters_to_preset_bounds(parameters: PresetCameraParameters, preset: Any) -> list[dict[str, Any]]:
    definitions = (
        ("camera_height_m", "camera_height_bounds_m", "nominal_camera_height_m"),
        ("distance_behind_wicket_m", "distance_bounds_m", "nominal_distance_behind_wicket_m"),
        ("lateral_offset_m", "lateral_offset_bounds_m", "nominal_lateral_offset_m"),
        ("yaw_deg", "yaw_bounds_deg", "nominal_yaw_deg"),
        ("pitch_deg", "pitch_bounds_deg", "nominal_pitch_deg"),
        ("roll_deg", "roll_bounds_deg", "nominal_roll_deg"),
        ("horizontal_fov_deg", "horizontal_fov_bounds_deg", "nominal_horizontal_fov_deg"),
    )
    result = []
    for name, bound_name, nominal_name in definitions:
        low, high = _bound_pair(preset, bound_name)
        nominal = float(preset[nominal_name] if isinstance(preset, Mapping) else getattr(preset, nominal_name))
        value = float(getattr(parameters, name))
        result.append({
            "parameter_name": name,
            "value": value,
            "lower_bound": low,
            "upper_bound": high,
            "inside_bounds": low <= value <= high,
            "distance_to_nearest_bound": min(abs(value - low), abs(high - value)),
            "nominal_value": nominal,
            "difference_from_nominal": value - nominal,
            "critical_bound_proximity": low <= value <= high and min(value - low, high - value) <= 0.02 * (high - low),
        })
    return result


def round_trip_projection_diagnostic(original: OpenCVCamera, reconstructed: OpenCVCamera) -> dict[str, Any]:
    specification = build_virtual_pitch_specification()
    world = np.asarray([[item.point.x, item.point.y, item.point.z] for item in specification.landmarks], dtype=np.float64)

    def project(camera: OpenCVCamera) -> tuple[np.ndarray, np.ndarray]:
        camera_points = (camera.rotation_matrix @ world.T + camera.translation_vector.reshape(3, 1)).T
        pixels_h = (camera.camera_matrix @ camera_points.T).T
        return pixels_h[:, :2] / pixels_h[:, 2:3], camera_points[:, 2]

    original_pixels, original_depths = project(original)
    rebuilt_pixels, rebuilt_depths = project(reconstructed)
    delta = rebuilt_pixels - original_pixels
    errors = np.linalg.norm(delta, axis=1)
    relative_rotation = reconstructed.rotation_matrix @ original.rotation_matrix.T
    angle = math.degrees(math.acos(float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))))
    return {
        "point_count": len(specification.landmarks),
        "projection_rmse_px": float(np.sqrt(np.mean(errors**2))),
        "projection_median_error_px": float(np.median(errors)),
        "projection_maximum_error_px": float(np.max(errors)),
        "x_bias_px": float(np.mean(delta[:, 0])),
        "y_bias_px": float(np.mean(delta[:, 1])),
        "camera_position_difference_m": float(np.linalg.norm(reconstructed.camera_position_world - original.camera_position_world)),
        "rotation_angular_difference_deg": angle,
        "mirror_warning": bool(np.linalg.det(reconstructed.rotation_matrix) < 0),
        "bowler_striker_reversal_warning": bool(np.dot(original.forward_vector_world, reconstructed.forward_vector_world) < 0),
        "positive_depth_mismatch_count": int(np.count_nonzero((original_depths > 0) != (rebuilt_depths > 0))),
    }


def known_camera_diagnostic(
    registration_path: str | Path,
    *,
    camera_end: str,
    image_left_mapping: str,
    preset: Any | None = None,
) -> dict[str, Any]:
    """Load one persisted selected camera and expose decomposition evidence."""
    path = Path(registration_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate = payload.get("selected_candidate")
    setup = payload.get("setup_frame")
    if not candidate or not setup:
        raise ValueError("Persisted registration has no selected camera/setup frame.")
    intrinsics = candidate["intrinsics"]
    matrix = [
        [intrinsics["focal_length_x_px"], 0.0, intrinsics["principal_point_x_px"]],
        [0.0, intrinsics["focal_length_y_px"], intrinsics["principal_point_y_px"]],
        [0.0, 0.0, 1.0],
    ]
    decomposition = decompose_opencv_camera_to_preset_parameters(
        camera_matrix=matrix,
        rotation_matrix=candidate["rotation_matrix"],
        translation_vector=candidate["translation_vector"],
        image_width=setup["image_width"],
        image_height=setup["image_height"],
        camera_end=camera_end,
        image_left_mapping=image_left_mapping,
        preset=preset,
    )
    reported_fov = intrinsics.get("horizontal_fov_degrees")
    effective_fov = decomposition.parameters.horizontal_fov_deg
    fov_mismatch = reported_fov is not None and not math.isclose(
        float(reported_fov), effective_fov, rel_tol=1e-9, abs_tol=1e-9
    )
    warnings = list(candidate.get("warnings", []))
    if fov_mismatch:
        warnings.append(
            "Persisted horizontal_fov_degrees does not match K; the effective "
            "FOV derived from image width and fx is authoritative for round trip."
        )
    return {
        "analysis_id": payload.get("analysis_id"),
        "source": str(path.resolve()),
        "candidate_id": candidate.get("candidate_id"),
        "image_size": [setup["image_width"], setup["image_height"]],
        "camera_matrix": matrix,
        "distortion_coefficients": intrinsics.get("distortion_coefficients", []),
        "rotation_matrix": candidate["rotation_matrix"],
        "rotation_vector": candidate.get("rotation_vector"),
        "translation_vector": candidate["translation_vector"],
        "camera_position_world": decomposition.camera_position_world.tolist(),
        "forward_vector_world": decomposition.forward_vector_world.tolist(),
        "camera_end": _normalise_camera_end(camera_end),
        "image_left_mapping": _normalise_mapping(image_left_mapping),
        "geometric_classification": candidate.get("classification", payload.get("status")),
        "production_accepted": False,
        "coordinate_convention": decomposition.diagnostics["coordinate_convention"],
        "parameters": asdict(decomposition.parameters),
        "reported_horizontal_fov_deg": reported_fov,
        "effective_horizontal_fov_deg": effective_fov,
        "reported_fov_matches_camera_matrix": not fov_mismatch,
        "parameter_bounds": list(decomposition.bound_diagnostics),
        "source_anchors": candidate.get("inlier_correspondence_ids", []),
        "setup_frame_index": candidate.get("setup_frame_index", setup.get("frame_index")),
        "warnings": warnings,
    }
