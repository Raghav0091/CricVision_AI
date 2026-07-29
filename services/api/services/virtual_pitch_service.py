"""Permanent virtual-pitch geometry, synthetic projection and PnP validation."""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Iterable, Sequence

import cv2
import numpy as np

from packages.cricket_vision.calibration.cricket_pitch_geometry import (
    BOWLING_CREASE_LENGTH_M,
    CANONICAL_COORDINATE_SYSTEM,
    CREASE_LINE_WIDTH_M,
    DISPLAY_DECIMAL_PLACES,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    POPPING_CREASE_OFFSET_M,
    RETURN_CREASE_OFFSET_M,
    STUMP_DIAMETER_MAX_M,
    STUMP_DIAMETER_MIN_M,
    STUMP_HEIGHT_M,
    VIRTUAL_PITCH_MODEL_VERSION,
    WICKET_WIDTH_M,
)

from ..schemas.virtual_pitch import (
    BailPrimitive,
    DisplayRoundingPolicy,
    PixelPoint2D,
    PitchLineSegment,
    PitchPolygon,
    PnPObservation,
    PnPRecoveryResult,
    ProjectedLandmark,
    ProjectedLineSegment,
    ProjectedPitchGeometry,
    ProjectedPolygon,
    ProjectedStump,
    ProjectionDiagnostics,
    StumpPrimitive,
    SyntheticPitchPreviewResponse,
    VirtualCamera,
    VirtualPitchCoordinateSystem,
    VirtualPitchDimensions,
    VirtualPitchLandmark,
    VirtualPitchProfile,
    VirtualPitchSpecification,
    WorldPoint3D,
)


PROJECTION_NEAR_M = 0.05
PROJECTION_FAR_M = 150.0
PNP_MIN_OBSERVATIONS = 6
PNP_RANSAC_REPROJECTION_ERROR_PX = 4.0
PNP_RANSAC_ITERATIONS = 300


def _point(x: float, y: float, z: float = 0.0) -> WorldPoint3D:
    return WorldPoint3D(x=x, y=y, z=z)


def _landmark(
    semantic_id: str,
    point: WorldPoint3D,
    *,
    category: str,
    end: str,
    anchor: bool,
    description: str,
    geometry_class: str = "official",
) -> VirtualPitchLandmark:
    return VirtualPitchLandmark(
        semantic_id=semantic_id,
        point=point,
        geometry_category=category,
        geometry_class=geometry_class,
        end=end,
        calibration_anchor=anchor,
        description=description,
    )


@lru_cache(maxsize=1)
def build_virtual_pitch_specification() -> VirtualPitchSpecification:
    """Build the immutable V1 metric model from authoritative constants."""
    half_pitch = PITCH_WIDTH_M / 2
    half_wicket = WICKET_WIDTH_M / 2
    stump_radius = STUMP_DIAMETER_MAX_M / 2
    outer_stump_x = (WICKET_WIDTH_M - STUMP_DIAMETER_MAX_M) / 2
    half_bowling_crease = BOWLING_CREASE_LENGTH_M / 2

    landmarks: list[VirtualPitchLandmark] = []
    stumps: list[StumpPrimitive] = []
    bails: list[BailPrimitive] = []
    for end, y in (("bowler", 0.0), ("striker", PITCH_LENGTH_M)):
        landmarks.append(
            _landmark(
                f"{end}_wicket_center_base",
                _point(0.0, y),
                category="wicket",
                end=end,
                anchor=True,
                description=f"Centre of the {end}-end wicket at pitch level.",
            )
        )
        positions = {
            "left": -outer_stump_x,
            "middle": 0.0,
            "right": outer_stump_x,
        }
        for stump_index, x in positions.items():
            base_id = f"{end}_{stump_index}_stump_base"
            top_id = f"{end}_{stump_index}_stump_top"
            landmarks.extend(
                [
                    _landmark(
                        base_id,
                        _point(x, y),
                        category="wicket",
                        end=end,
                        anchor=True,
                        description=(
                            f"{end.title()}-end {stump_index} stump body base."
                        ),
                    ),
                    _landmark(
                        top_id,
                        _point(x, y, STUMP_HEIGHT_M),
                        category="wicket",
                        end=end,
                        anchor=True,
                        description=(
                            f"{end.title()}-end {stump_index} stump body top, "
                            "excluding bails."
                        ),
                    ),
                ]
            )
            stumps.append(
                StumpPrimitive(
                    primitive_id=f"{end}_{stump_index}_stump",
                    centre=_point(x, y, STUMP_HEIGHT_M / 2),
                    radius_m=stump_radius,
                    height_m=STUMP_HEIGHT_M,
                    orientation=_point(0.0, 0.0, 1.0),
                    end=end,
                    stump_index=stump_index,
                )
            )
        for bail_index, start_x, end_x in (
            ("left_middle", -outer_stump_x, 0.0),
            ("middle_right", 0.0, outer_stump_x),
        ):
            bails.append(
                BailPrimitive(
                    primitive_id=f"{end}_{bail_index}_bail",
                    start=_point(start_x, y, STUMP_HEIGHT_M + stump_radius * 0.45),
                    end_point=_point(
                        end_x,
                        y,
                        STUMP_HEIGHT_M + stump_radius * 0.45,
                    ),
                    radius_m=stump_radius * 0.45,
                    end=end,
                    bail_index=bail_index,
                )
            )

    pitch_corners = {
        "bowler_left_pitch_corner": _point(-half_pitch, 0.0),
        "bowler_right_pitch_corner": _point(half_pitch, 0.0),
        "striker_left_pitch_corner": _point(-half_pitch, PITCH_LENGTH_M),
        "striker_right_pitch_corner": _point(half_pitch, PITCH_LENGTH_M),
    }
    for semantic_id, point in pitch_corners.items():
        end = "bowler" if semantic_id.startswith("bowler") else "striker"
        landmarks.append(
            _landmark(
                semantic_id,
                point,
                category="pitch",
                end=end,
                anchor=True,
                description="Official pitch-surface corner.",
            )
        )

    landmarks.extend(
        [
            _landmark(
                "pitch_centerline_bowler_endpoint",
                _point(0.0, 0.0),
                category="analytical",
                end="bowler",
                anchor=False,
                geometry_class="analytical",
                description="Analytical centreline endpoint at the bowler end.",
            ),
            _landmark(
                "pitch_centerline_striker_endpoint",
                _point(0.0, PITCH_LENGTH_M),
                category="analytical",
                end="striker",
                anchor=False,
                geometry_class="analytical",
                description="Analytical centreline endpoint at the striker end.",
            ),
        ]
    )

    line_segments: list[PitchLineSegment] = []

    def add_line(
        primitive_id: str,
        start: WorldPoint3D,
        end_point: WorldPoint3D,
        category: str,
        geometry_class: str,
        end: str,
        profile_id: str | None = None,
    ) -> None:
        line_segments.append(
            PitchLineSegment(
                primitive_id=primitive_id,
                start=start,
                end_point=end_point,
                line_category=category,
                geometry_class=geometry_class,
                line_width_m=CREASE_LINE_WIDTH_M,
                end=end,
                profile_id=profile_id,
            )
        )

    add_line(
        "pitch_left_boundary",
        _point(-half_pitch, 0.0),
        _point(-half_pitch, PITCH_LENGTH_M),
        "pitch_boundary",
        "official",
        "both",
    )
    add_line(
        "pitch_right_boundary",
        _point(half_pitch, 0.0),
        _point(half_pitch, PITCH_LENGTH_M),
        "pitch_boundary",
        "official",
        "both",
    )
    add_line(
        "pitch_centerline",
        _point(0.0, 0.0, 0.002),
        _point(0.0, PITCH_LENGTH_M, 0.002),
        "centreline",
        "analytical",
        "both",
        "analytical",
    )

    for end, wicket_y, direction in (
        ("bowler", 0.0, 1.0),
        ("striker", PITCH_LENGTH_M, -1.0),
    ):
        popping_y = wicket_y + direction * POPPING_CREASE_OFFSET_M
        crease_points = {
            f"{end}_bowling_crease_left_endpoint": _point(
                -half_bowling_crease,
                wicket_y,
            ),
            f"{end}_bowling_crease_right_endpoint": _point(
                half_bowling_crease,
                wicket_y,
            ),
            f"{end}_popping_crease_left_endpoint": _point(
                -RETURN_CREASE_OFFSET_M,
                popping_y,
            ),
            f"{end}_popping_crease_right_endpoint": _point(
                RETURN_CREASE_OFFSET_M,
                popping_y,
            ),
            f"{end}_left_return_bowling_intersection": _point(
                -RETURN_CREASE_OFFSET_M,
                wicket_y,
            ),
            f"{end}_right_return_bowling_intersection": _point(
                RETURN_CREASE_OFFSET_M,
                wicket_y,
            ),
            f"{end}_left_return_popping_intersection": _point(
                -RETURN_CREASE_OFFSET_M,
                popping_y,
            ),
            f"{end}_right_return_popping_intersection": _point(
                RETURN_CREASE_OFFSET_M,
                popping_y,
            ),
        }
        for semantic_id, point in crease_points.items():
            landmarks.append(
                _landmark(
                    semantic_id,
                    point,
                    category="crease",
                    end=end,
                    anchor=True,
                    description=f"Official {end}-end crease reference.",
                )
            )
        add_line(
            f"{end}_bowling_crease",
            _point(-half_bowling_crease, wicket_y),
            _point(half_bowling_crease, wicket_y),
            "bowling_crease",
            "official",
            end,
        )
        add_line(
            f"{end}_popping_crease",
            _point(-RETURN_CREASE_OFFSET_M, popping_y),
            _point(RETURN_CREASE_OFFSET_M, popping_y),
            "popping_crease",
            "official",
            end,
        )
        for side, x in (
            ("left", -RETURN_CREASE_OFFSET_M),
            ("right", RETURN_CREASE_OFFSET_M),
        ):
            add_line(
                f"{end}_{side}_return_crease_registration_span",
                _point(x, wicket_y),
                _point(x, popping_y),
                "return_crease",
                "official",
                end,
            )

    polygons = [
        PitchPolygon(
            primitive_id="pitch_surface",
            vertices=[
                _point(-half_pitch, 0.0),
                _point(half_pitch, 0.0),
                _point(half_pitch, PITCH_LENGTH_M),
                _point(-half_pitch, PITCH_LENGTH_M),
            ],
            polygon_category="pitch_surface",
            geometry_class="official",
            end="both",
            display_opacity=0.10,
        ),
        PitchPolygon(
            primitive_id="lbw_stump_to_stump_corridor",
            vertices=[
                _point(-half_wicket, 0.0, 0.003),
                _point(half_wicket, 0.0, 0.003),
                _point(half_wicket, PITCH_LENGTH_M, 0.003),
                _point(-half_wicket, PITCH_LENGTH_M, 0.003),
            ],
            polygon_category="lbw_corridor",
            geometry_class="analytical",
            end="both",
            profile_id="analytical",
            display_opacity=0.16,
        ),
    ]
    profiles = [
        VirtualPitchProfile(
            profile_id="official_core",
            label="Official Core",
            geometry_class="official",
            description=(
                "Pitch, wickets and crease registration spans generated from "
                "the official base dimensions."
            ),
            enabled_primitive_ids=[],
            universal_official_geometry=True,
        ),
        VirtualPitchProfile(
            profile_id="analytical",
            label="CricVision Analytical",
            geometry_class="analytical",
            description="Centreline and stump-to-stump analytical corridor.",
            enabled_primitive_ids=[
                "pitch_centerline",
                "lbw_stump_to_stump_corridor",
            ],
            universal_official_geometry=False,
        ),
        VirtualPitchProfile(
            profile_id="training_default",
            label="Training Guides",
            geometry_class="optional",
            description=(
                "Reserved for competition or training-specific guides; no "
                "universal wide-line claim is made."
            ),
            enabled_primitive_ids=[],
            universal_official_geometry=False,
        ),
    ]
    return VirtualPitchSpecification(
        coordinate_system=VirtualPitchCoordinateSystem(
            description=CANONICAL_COORDINATE_SYSTEM
        ),
        dimensions=VirtualPitchDimensions(
            pitch_length_m=PITCH_LENGTH_M,
            pitch_width_m=PITCH_WIDTH_M,
            wicket_width_m=WICKET_WIDTH_M,
            stump_height_m=STUMP_HEIGHT_M,
            stump_diameter_min_m=STUMP_DIAMETER_MIN_M,
            stump_diameter_max_m=STUMP_DIAMETER_MAX_M,
            bowling_crease_length_m=BOWLING_CREASE_LENGTH_M,
            popping_crease_offset_m=POPPING_CREASE_OFFSET_M,
            return_crease_offset_m=RETURN_CREASE_OFFSET_M,
        ),
        landmarks=landmarks,
        stumps=stumps,
        bails=bails,
        line_segments=line_segments,
        polygons=polygons,
        profiles=profiles,
        display_rounding=DisplayRoundingPolicy(
            display_decimal_places=DISPLAY_DECIMAL_PLACES
        ),
        synthetic_camera_names=list(synthetic_camera_names()),
    )


_CAMERA_CASES = {
    "centred_bowler_end": {
        "description": "Centred camera behind the bowler-end wicket.",
        "size": (1280, 720),
        "position": (0.0, -8.0, 3.2),
        "target": (0.0, 10.0, 0.25),
        "fov": 58.0,
    },
    "centred_striker_end": {
        "description": "Centred camera behind the striker-end wicket.",
        "size": (1280, 720),
        "position": (0.0, 28.0, 3.2),
        "target": (0.0, 10.0, 0.25),
        "fov": 58.0,
    },
    "left_of_centre": {
        "description": "Bowler-end camera three metres left of centre.",
        "size": (1280, 720),
        "position": (-3.0, -7.0, 3.0),
        "target": (0.0, 10.0, 0.3),
        "fov": 60.0,
    },
    "right_of_centre": {
        "description": "Bowler-end camera three metres right of centre.",
        "size": (1280, 720),
        "position": (3.0, -7.0, 3.0),
        "target": (0.0, 10.0, 0.3),
        "fov": 60.0,
    },
    "low_camera": {
        "description": "Low centred bowler-end camera.",
        "size": (1280, 720),
        "position": (0.0, -7.0, 1.05),
        "target": (0.0, 10.0, 0.25),
        "fov": 60.0,
    },
    "elevated_camera": {
        "description": "Elevated centred bowler-end camera.",
        "size": (1280, 720),
        "position": (0.0, -9.0, 8.0),
        "target": (0.0, 10.0, 0.0),
        "fov": 62.0,
    },
    "narrow_focal_length": {
        "description": "Centred landscape camera with a narrow field of view.",
        "size": (1280, 720),
        "position": (0.0, -10.0, 3.5),
        "target": (0.0, 10.0, 0.25),
        "fov": 36.0,
    },
    "wide_focal_length": {
        "description": "Centred landscape camera with a wide field of view.",
        "size": (1280, 720),
        "position": (0.0, -6.0, 3.0),
        "target": (0.0, 10.0, 0.25),
        "fov": 78.0,
    },
    "portrait_bowler_end": {
        "description": "Centred bowler-end portrait video camera.",
        "size": (720, 1280),
        "position": (0.0, -7.0, 3.2),
        "target": (0.0, 10.0, 0.3),
        "fov": 55.0,
    },
    "landscape_bowler_end": {
        "description": "Centred bowler-end landscape video camera.",
        "size": (1920, 1080),
        "position": (0.0, -8.0, 3.2),
        "target": (0.0, 10.0, 0.3),
        "fov": 55.0,
    },
}


def synthetic_camera_names() -> tuple[str, ...]:
    return tuple(_CAMERA_CASES)


@lru_cache(maxsize=len(_CAMERA_CASES))
def build_synthetic_camera(name: str) -> VirtualCamera:
    try:
        case = _CAMERA_CASES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown synthetic camera: {name}") from exc
    width, height = case["size"]
    position = np.asarray(case["position"], dtype=np.float64)
    target = np.asarray(case["target"], dtype=np.float64)
    fov = float(case["fov"])
    focal = width / (2 * math.tan(math.radians(fov) / 2))
    camera_matrix = np.asarray(
        [
            [focal, 0.0, width / 2],
            [0.0, focal, height / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack([right, down, forward])
    translation = -rotation @ position
    rotation_vector, _ = cv2.Rodrigues(rotation)
    return VirtualCamera(
        name=name,
        description=str(case["description"]),
        image_width=int(width),
        image_height=int(height),
        camera_matrix=camera_matrix.tolist(),
        distortion_coefficients=[0.0] * 5,
        rotation_vector=rotation_vector.reshape(-1).tolist(),
        rotation_matrix=rotation.tolist(),
        translation_vector=translation.tolist(),
        camera_position_world=position.tolist(),
        target_world=target.tolist(),
        near_m=PROJECTION_NEAR_M,
        far_m=PROJECTION_FAR_M,
        horizontal_fov_degrees=fov,
    )


def _camera_arrays(camera: VirtualCamera):
    return (
        np.asarray(camera.camera_matrix, dtype=np.float64),
        np.asarray(camera.distortion_coefficients, dtype=np.float64),
        np.asarray(camera.rotation_vector, dtype=np.float64).reshape(3, 1),
        np.asarray(camera.translation_vector, dtype=np.float64).reshape(3, 1),
        np.asarray(camera.rotation_matrix, dtype=np.float64),
    )


def _project_world_points(
    points: Sequence[WorldPoint3D],
    camera: VirtualCamera,
) -> tuple[np.ndarray, np.ndarray]:
    camera_matrix, distortion, rotation_vector, translation, rotation = (
        _camera_arrays(camera)
    )
    world = np.asarray([[p.x, p.y, p.z] for p in points], dtype=np.float64)
    projected, _ = cv2.projectPoints(
        world,
        rotation_vector,
        translation,
        camera_matrix,
        distortion,
    )
    camera_points = (rotation @ world.T + translation).T
    return projected.reshape(-1, 2), camera_points[:, 2]


def _pixel(point: np.ndarray) -> PixelPoint2D:
    return PixelPoint2D(x=round(float(point[0]), 6), y=round(float(point[1]), 6))


def _in_frame(pixel: np.ndarray, camera: VirtualCamera) -> bool:
    return bool(
        0 <= pixel[0] < camera.image_width
        and 0 <= pixel[1] < camera.image_height
    )


def project_virtual_pitch(
    camera: VirtualCamera,
    specification: VirtualPitchSpecification | None = None,
) -> ProjectedPitchGeometry:
    specification = specification or build_virtual_pitch_specification()
    projected_landmarks: list[ProjectedLandmark] = []
    for landmark in specification.landmarks:
        pixels, depths = _project_world_points([landmark.point], camera)
        depth = float(depths[0])
        valid = depth > camera.near_m and np.isfinite(pixels[0]).all()
        projected_landmarks.append(
            ProjectedLandmark(
                semantic_id=landmark.semantic_id,
                world_point=landmark.point,
                pixel_point=_pixel(pixels[0]) if valid else None,
                visible=valid,
                in_frame=valid and _in_frame(pixels[0], camera),
                depth_m=round(depth, 6),
                projection_valid=valid,
                invalid_reason=None if valid else "behind_or_on_camera_plane",
            )
        )

    projected_lines: list[ProjectedLineSegment] = []
    for line in specification.line_segments:
        pixels, depths = _project_world_points(
            [line.start, line.end_point],
            camera,
        )
        valid = bool(
            np.all(depths > camera.near_m) and np.isfinite(pixels).all()
        )
        in_frame = [_in_frame(pixel, camera) for pixel in pixels]
        projected_lines.append(
            ProjectedLineSegment(
                primitive_id=line.primitive_id,
                line_category=line.line_category,
                geometry_class=line.geometry_class,
                world_start=line.start,
                world_end=line.end_point,
                pixel_start=_pixel(pixels[0]) if valid else None,
                pixel_end=_pixel(pixels[1]) if valid else None,
                start_depth_m=round(float(depths[0]), 6),
                end_depth_m=round(float(depths[1]), 6),
                projection_valid=valid,
                partially_out_of_frame=valid and not all(in_frame),
            )
        )

    projected_stumps: list[ProjectedStump] = []
    for stump in specification.stumps:
        base = _point(
            stump.centre.x,
            stump.centre.y,
            stump.centre.z - stump.height_m / 2,
        )
        top = _point(
            stump.centre.x,
            stump.centre.y,
            stump.centre.z + stump.height_m / 2,
        )
        pixels, depths = _project_world_points([base, top], camera)
        valid = bool(
            np.all(depths > camera.near_m) and np.isfinite(pixels).all()
        )
        height_px = float(np.linalg.norm(pixels[1] - pixels[0])) if valid else None
        focal = float(camera.camera_matrix[0][0])
        radius_px = focal * stump.radius_m / float(np.mean(depths)) if valid else None
        projected_stumps.append(
            ProjectedStump(
                primitive_id=stump.primitive_id,
                end=stump.end,
                stump_index=stump.stump_index,
                pixel_base=_pixel(pixels[0]) if valid else None,
                pixel_top=_pixel(pixels[1]) if valid else None,
                base_depth_m=round(float(depths[0]), 6),
                top_depth_m=round(float(depths[1]), 6),
                projected_height_px=round(height_px, 6) if height_px is not None else None,
                projected_radius_px=round(radius_px, 6) if radius_px is not None else None,
                projection_valid=valid,
                in_frame=valid and any(_in_frame(pixel, camera) for pixel in pixels),
            )
        )

    projected_polygons: list[ProjectedPolygon] = []
    for polygon in specification.polygons:
        pixels, depths = _project_world_points(polygon.vertices, camera)
        validity = depths > camera.near_m
        valid = bool(np.all(validity) and np.isfinite(pixels).all())
        frame_flags = [
            _in_frame(pixel, camera) if point_valid else False
            for pixel, point_valid in zip(pixels, validity, strict=True)
        ]
        projected_polygons.append(
            ProjectedPolygon(
                primitive_id=polygon.primitive_id,
                polygon_category=polygon.polygon_category,
                geometry_class=polygon.geometry_class,
                world_vertices=polygon.vertices,
                pixel_vertices=[
                    _pixel(pixel) if point_valid and np.isfinite(pixel).all() else None
                    for pixel, point_valid in zip(pixels, validity, strict=True)
                ],
                depths_m=[round(float(depth), 6) for depth in depths],
                projection_valid=valid,
                partially_out_of_frame=valid and not all(frame_flags),
            )
        )

    projected_bails: list[ProjectedLineSegment] = []
    for bail in specification.bails:
        pixels, depths = _project_world_points(
            [bail.start, bail.end_point],
            camera,
        )
        valid = bool(
            np.all(depths > camera.near_m) and np.isfinite(pixels).all()
        )
        projected_bails.append(
            ProjectedLineSegment(
                primitive_id=bail.primitive_id,
                line_category="bail",
                geometry_class="official",
                world_start=bail.start,
                world_end=bail.end_point,
                pixel_start=_pixel(pixels[0]) if valid else None,
                pixel_end=_pixel(pixels[1]) if valid else None,
                start_depth_m=round(float(depths[0]), 6),
                end_depth_m=round(float(depths[1]), 6),
                projection_valid=valid,
                partially_out_of_frame=valid
                and not all(_in_frame(pixel, camera) for pixel in pixels),
            )
        )

    bowler = [
        stump
        for stump in projected_stumps
        if stump.end == "bowler" and stump.projection_valid
    ]
    striker = [
        stump
        for stump in projected_stumps
        if stump.end == "striker" and stump.projection_valid
    ]
    bowler_depth = _mean([item.base_depth_m for item in bowler])
    striker_depth = _mean([item.base_depth_m for item in striker])
    bowler_height = _mean(
        [item.projected_height_px for item in bowler if item.projected_height_px is not None]
    )
    striker_height = _mean(
        [item.projected_height_px for item in striker if item.projected_height_px is not None]
    )
    if bowler_depth is None or striker_depth is None:
        nearer = "unavailable"
        perspective_valid = False
    elif abs(bowler_depth - striker_depth) < 1e-6:
        nearer = "equal"
        perspective_valid = True
    else:
        nearer = "bowler" if bowler_depth < striker_depth else "striker"
        near_height = bowler_height if nearer == "bowler" else striker_height
        far_height = striker_height if nearer == "bowler" else bowler_height
        perspective_valid = (
            near_height is not None
            and far_height is not None
            and near_height > far_height
        )
    valid_landmarks = [
        item for item in projected_landmarks if item.projection_valid
    ]
    in_frame_landmarks = [item for item in valid_landmarks if item.in_frame]
    diagnostics = ProjectionDiagnostics(
        projected_landmark_count=len(projected_landmarks),
        valid_landmark_count=len(valid_landmarks),
        in_frame_landmark_count=len(in_frame_landmarks),
        behind_camera_count=sum(
            not item.projection_valid for item in projected_landmarks
        ),
        out_of_frame_count=sum(
            item.projection_valid and not item.in_frame
            for item in projected_landmarks
        ),
        bowler_wicket_mean_depth_m=bowler_depth,
        striker_wicket_mean_depth_m=striker_depth,
        bowler_wicket_mean_height_px=bowler_height,
        striker_wicket_mean_height_px=striker_height,
        nearer_wicket=nearer,
        perspective_order_valid=perspective_valid,
        warnings=(
            []
            if perspective_valid
            else ["Projected wicket size does not match expected depth ordering."]
        ),
    )
    return ProjectedPitchGeometry(
        source_camera=camera,
        projected_landmarks=projected_landmarks,
        projected_line_segments=projected_lines,
        projected_stumps=projected_stumps,
        projected_polygons=projected_polygons,
        projected_bails=projected_bails,
        diagnostics=diagnostics,
    )


def build_synthetic_preview(
    camera_name: str,
    profile: str = "analytical",
) -> SyntheticPitchPreviewResponse:
    specification = build_virtual_pitch_specification()
    if profile not in {item.profile_id for item in specification.profiles}:
        raise ValueError(f"Unknown virtual-pitch profile: {profile}")
    camera = build_synthetic_camera(camera_name)
    return SyntheticPitchPreviewResponse(
        specification=specification,
        projection=project_virtual_pitch(camera, specification),
        selected_profile=profile,
        message=(
            "Synthetic camera projection only. The virtual pitch is not "
            "registered to this video and metric analytics remain locked."
        ),
    )


def recover_synthetic_camera_pose(
    camera: VirtualCamera,
    observations: Sequence[PnPObservation],
) -> PnPRecoveryResult:
    specification = build_virtual_pitch_specification()
    landmark_by_id = {
        landmark.semantic_id: landmark for landmark in specification.landmarks
    }
    usable = [
        observation
        for observation in observations
        if observation.semantic_id in landmark_by_id
    ]
    if len(usable) < PNP_MIN_OBSERVATIONS:
        return PnPRecoveryResult(
            success=False,
            solver_method="solvePnPRansac",
            failure_reason="insufficient_correspondences",
        )
    world = np.asarray(
        [
            [
                landmark_by_id[item.semantic_id].point.x,
                landmark_by_id[item.semantic_id].point.y,
                landmark_by_id[item.semantic_id].point.z,
            ]
            for item in usable
        ],
        dtype=np.float64,
    )
    centred = world - np.mean(world, axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    if len(singular) < 2 or singular[1] <= max(1e-8, singular[0] * 1e-5):
        return PnPRecoveryResult(
            success=False,
            solver_method="solvePnPRansac",
            failure_reason="degenerate_world_geometry",
        )
    image = np.asarray(
        [[item.pixel_point.x, item.pixel_point.y] for item in usable],
        dtype=np.float64,
    )
    camera_matrix, distortion, _, _, _ = _camera_arrays(camera)
    success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
        world,
        image,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
        iterationsCount=PNP_RANSAC_ITERATIONS,
        reprojectionError=PNP_RANSAC_REPROJECTION_ERROR_PX,
        confidence=0.999,
    )
    if not success or inliers is None or len(inliers) < 4:
        return PnPRecoveryResult(
            success=False,
            solver_method="solvePnPRansac",
            failure_reason="ransac_pose_recovery_failed",
        )
    inlier_indexes = inliers.reshape(-1)
    refinement = None
    if hasattr(cv2, "solvePnPRefineLM"):
        rotation_vector, translation_vector = cv2.solvePnPRefineLM(
            world[inlier_indexes],
            image[inlier_indexes],
            camera_matrix,
            distortion,
            rotation_vector,
            translation_vector,
        )
        refinement = "solvePnPRefineLM"
    projected, _ = cv2.projectPoints(
        world[inlier_indexes],
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion,
    )
    residuals = np.linalg.norm(
        projected.reshape(-1, 2) - image[inlier_indexes],
        axis=1,
    )
    recovered_rotation, _ = cv2.Rodrigues(rotation_vector)
    expected_rotation = np.asarray(camera.rotation_matrix, dtype=np.float64)
    relative = recovered_rotation @ expected_rotation.T
    cosine = _clamp((np.trace(relative) - 1) / 2, -1.0, 1.0)
    rotation_error = math.degrees(math.acos(cosine))
    expected_translation = np.asarray(
        camera.translation_vector,
        dtype=np.float64,
    ).reshape(3)
    translation_error = float(
        np.linalg.norm(translation_vector.reshape(3) - expected_translation)
    )
    inlier_set = set(int(index) for index in inlier_indexes)
    return PnPRecoveryResult(
        success=True,
        solver_method="solvePnPRansac",
        refinement_method=refinement,
        rotation_vector=rotation_vector.reshape(-1).tolist(),
        translation_vector=translation_vector.reshape(-1).tolist(),
        rotation_error_degrees=round(rotation_error, 9),
        translation_error_m=round(translation_error, 9),
        reprojection_rmse_px=round(
            float(np.sqrt(np.mean(np.square(residuals)))),
            9,
        ),
        inlier_count=len(inlier_indexes),
        inlier_landmark_ids=[
            item.semantic_id
            for index, item in enumerate(usable)
            if index in inlier_set
        ],
        outlier_landmark_ids=[
            item.semantic_id
            for index, item in enumerate(usable)
            if index not in inlier_set
        ],
    )


def projected_landmark_observations(
    projection: ProjectedPitchGeometry,
    *,
    calibration_anchors_only: bool = True,
) -> list[PnPObservation]:
    specification = build_virtual_pitch_specification()
    anchor_ids = {
        item.semantic_id
        for item in specification.landmarks
        if item.calibration_anchor
    }
    return [
        PnPObservation(
            semantic_id=item.semantic_id,
            pixel_point=item.pixel_point,
        )
        for item in projection.projected_landmarks
        if item.projection_valid
        and item.pixel_point is not None
        and (not calibration_anchors_only or item.semantic_id in anchor_ids)
    ]


def map_native_pixel_to_contained_display(
    pixel_x: float,
    pixel_y: float,
    *,
    native_width: float,
    native_height: float,
    container_width: float,
    container_height: float,
) -> tuple[float, float]:
    """Map video-native pixels through CSS object-contain letterboxing."""
    if min(native_width, native_height, container_width, container_height) <= 0:
        raise ValueError("Image and container dimensions must be positive.")
    scale = min(
        container_width / native_width,
        container_height / native_height,
    )
    rendered_width = native_width * scale
    rendered_height = native_height * scale
    offset_x = (container_width - rendered_width) / 2
    offset_y = (container_height - rendered_height) / 2
    return (
        offset_x + pixel_x * scale,
        offset_y + pixel_y * scale,
    )


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return round(float(np.mean(values)), 6) if values else None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


assert VIRTUAL_PITCH_MODEL_VERSION == "v1"
