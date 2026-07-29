"""Contracts for the permanent metric virtual-pitch model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


GeometryClass = Literal["official", "analytical", "optional"]
PitchEnd = Literal["bowler", "striker", "both", "none"]


class VirtualPitchModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class WorldPoint3D(VirtualPitchModel):
    x: float
    y: float
    z: float


class PixelPoint2D(VirtualPitchModel):
    x: float
    y: float


class VirtualPitchCoordinateSystem(VirtualPitchModel):
    units: Literal["metres"] = "metres"
    handedness: Literal["right_handed"] = "right_handed"
    origin: Literal["bowler_end_middle_stump_base"] = (
        "bowler_end_middle_stump_base"
    )
    x_axis: Literal["lateral_camera_neutral_right"] = (
        "lateral_camera_neutral_right"
    )
    y_axis: Literal["bowler_to_striker"] = "bowler_to_striker"
    z_axis: Literal["up"] = "up"
    description: str
    off_leg_assignment: Literal["not_assigned"] = "not_assigned"


class VirtualPitchDimensions(VirtualPitchModel):
    pitch_length_m: float = Field(gt=0)
    pitch_width_m: float = Field(gt=0)
    wicket_width_m: float = Field(gt=0)
    stump_height_m: float = Field(gt=0)
    stump_diameter_min_m: float = Field(gt=0)
    stump_diameter_max_m: float = Field(gt=0)
    bowling_crease_length_m: float = Field(gt=0)
    popping_crease_offset_m: float = Field(gt=0)
    return_crease_offset_m: float = Field(gt=0)


class VirtualPitchLandmark(VirtualPitchModel):
    semantic_id: str
    point: WorldPoint3D
    geometry_category: Literal["wicket", "crease", "pitch", "analytical"]
    geometry_class: GeometryClass
    end: PitchEnd
    calibration_anchor: bool
    description: str


class StumpPrimitive(VirtualPitchModel):
    primitive_id: str
    centre: WorldPoint3D
    radius_m: float = Field(gt=0)
    height_m: float = Field(gt=0)
    orientation: WorldPoint3D
    end: Literal["bowler", "striker"]
    stump_index: Literal["left", "middle", "right"]
    geometry_class: Literal["official"] = "official"


class BailPrimitive(VirtualPitchModel):
    primitive_id: str
    start: WorldPoint3D
    end_point: WorldPoint3D
    radius_m: float = Field(gt=0)
    end: Literal["bowler", "striker"]
    bail_index: Literal["left_middle", "middle_right"]
    geometry_class: Literal["official"] = "official"
    cosmetic: Literal[True] = True


class PitchLineSegment(VirtualPitchModel):
    primitive_id: str
    start: WorldPoint3D
    end_point: WorldPoint3D
    line_category: Literal[
        "pitch_boundary",
        "bowling_crease",
        "popping_crease",
        "return_crease",
        "centreline",
        "trajectory_grid",
        "coaching_guide",
    ]
    geometry_class: GeometryClass
    line_width_m: float = Field(gt=0)
    end: PitchEnd
    profile_id: str | None = None


class PitchPolygon(VirtualPitchModel):
    primitive_id: str
    vertices: list[WorldPoint3D] = Field(min_length=3)
    polygon_category: Literal[
        "pitch_surface",
        "pitch_boundary",
        "lbw_corridor",
        "trajectory_plane",
        "coaching_region",
    ]
    geometry_class: GeometryClass
    end: PitchEnd
    profile_id: str | None = None
    display_opacity: float = Field(default=0.15, ge=0, le=1)


class VirtualPitchProfile(VirtualPitchModel):
    profile_id: str
    label: str
    geometry_class: GeometryClass
    description: str
    enabled_primitive_ids: list[str] = Field(default_factory=list)
    universal_official_geometry: bool


class DisplayRoundingPolicy(VirtualPitchModel):
    stored_precision: Literal["full_float"] = "full_float"
    display_decimal_places: int = Field(ge=0, le=8)
    display_units: Literal["metres"] = "metres"


class VirtualPitchSpecification(VirtualPitchModel):
    virtual_pitch_model_version: Literal["v1"] = "v1"
    coordinate_system: VirtualPitchCoordinateSystem
    dimensions: VirtualPitchDimensions
    landmarks: list[VirtualPitchLandmark]
    stumps: list[StumpPrimitive]
    bails: list[BailPrimitive]
    line_segments: list[PitchLineSegment]
    polygons: list[PitchPolygon]
    profiles: list[VirtualPitchProfile]
    display_rounding: DisplayRoundingPolicy
    synthetic_camera_names: list[str]


class VirtualCamera(VirtualPitchModel):
    name: str
    description: str
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    camera_matrix: list[list[float]]
    distortion_coefficients: list[float] = Field(min_length=4, max_length=14)
    rotation_vector: list[float] = Field(min_length=3, max_length=3)
    rotation_matrix: list[list[float]]
    translation_vector: list[float] = Field(min_length=3, max_length=3)
    camera_position_world: list[float] = Field(min_length=3, max_length=3)
    target_world: list[float] = Field(min_length=3, max_length=3)
    near_m: float = Field(gt=0)
    far_m: float = Field(gt=0)
    horizontal_fov_degrees: float = Field(gt=1, lt=179)
    developer_only: Literal[True] = True


class ProjectedLandmark(VirtualPitchModel):
    semantic_id: str
    world_point: WorldPoint3D
    pixel_point: PixelPoint2D | None = None
    visible: bool
    in_frame: bool
    depth_m: float
    projection_valid: bool
    invalid_reason: str | None = None


class ProjectedLineSegment(VirtualPitchModel):
    primitive_id: str
    line_category: str
    geometry_class: GeometryClass
    world_start: WorldPoint3D
    world_end: WorldPoint3D
    pixel_start: PixelPoint2D | None = None
    pixel_end: PixelPoint2D | None = None
    start_depth_m: float
    end_depth_m: float
    projection_valid: bool
    partially_out_of_frame: bool


class ProjectedStump(VirtualPitchModel):
    primitive_id: str
    end: Literal["bowler", "striker"]
    stump_index: Literal["left", "middle", "right"]
    pixel_base: PixelPoint2D | None = None
    pixel_top: PixelPoint2D | None = None
    base_depth_m: float
    top_depth_m: float
    projected_height_px: float | None = Field(default=None, ge=0)
    projected_radius_px: float | None = Field(default=None, ge=0)
    projection_valid: bool
    in_frame: bool


class ProjectedPolygon(VirtualPitchModel):
    primitive_id: str
    polygon_category: str
    geometry_class: GeometryClass
    world_vertices: list[WorldPoint3D]
    pixel_vertices: list[PixelPoint2D | None]
    depths_m: list[float]
    projection_valid: bool
    partially_out_of_frame: bool


class ProjectionDiagnostics(VirtualPitchModel):
    projected_landmark_count: int = Field(ge=0)
    valid_landmark_count: int = Field(ge=0)
    in_frame_landmark_count: int = Field(ge=0)
    behind_camera_count: int = Field(ge=0)
    out_of_frame_count: int = Field(ge=0)
    bowler_wicket_mean_depth_m: float | None = Field(default=None, gt=0)
    striker_wicket_mean_depth_m: float | None = Field(default=None, gt=0)
    bowler_wicket_mean_height_px: float | None = Field(default=None, ge=0)
    striker_wicket_mean_height_px: float | None = Field(default=None, ge=0)
    nearer_wicket: Literal["bowler", "striker", "equal", "unavailable"]
    perspective_order_valid: bool
    warnings: list[str] = Field(default_factory=list)


class ProjectedPitchGeometry(VirtualPitchModel):
    virtual_pitch_model_version: Literal["v1"] = "v1"
    source_camera: VirtualCamera
    projected_landmarks: list[ProjectedLandmark]
    projected_line_segments: list[ProjectedLineSegment]
    projected_stumps: list[ProjectedStump]
    projected_polygons: list[ProjectedPolygon]
    projected_bails: list[ProjectedLineSegment]
    diagnostics: ProjectionDiagnostics
    synthetic_only: Literal[True] = True


class SyntheticPitchPreviewResponse(VirtualPitchModel):
    specification: VirtualPitchSpecification
    projection: ProjectedPitchGeometry
    selected_profile: str
    developer_only: Literal[True] = True
    registration_status: Literal["not_registered_to_video"] = (
        "not_registered_to_video"
    )
    message: str


class PnPObservation(VirtualPitchModel):
    semantic_id: str
    pixel_point: PixelPoint2D


class PnPRecoveryResult(VirtualPitchModel):
    success: bool
    solver_method: str
    refinement_method: str | None = None
    rotation_vector: list[float] | None = None
    translation_vector: list[float] | None = None
    rotation_error_degrees: float | None = Field(default=None, ge=0)
    translation_error_m: float | None = Field(default=None, ge=0)
    reprojection_rmse_px: float | None = Field(default=None, ge=0)
    inlier_count: int = Field(default=0, ge=0)
    inlier_landmark_ids: list[str] = Field(default_factory=list)
    outlier_landmark_ids: list[str] = Field(default_factory=list)
    failure_reason: str | None = None
