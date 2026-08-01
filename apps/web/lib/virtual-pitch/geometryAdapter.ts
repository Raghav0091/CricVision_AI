import type {
  BailModel,
  CoordinateSystem,
  GeometryClass,
  PitchDimensions,
  PitchEnd,
  PitchLandmark,
  PitchLineSegment,
  PitchPolygon,
  PitchProfile,
  StumpModel,
  VirtualPitchModel,
  WicketEnd,
  CricketVector3
} from "./types";

export class VirtualPitchContractError extends Error {
  readonly path: string;

  constructor(path: string, expectation: string) {
    super(`Invalid virtual-pitch response at ${path}: expected ${expectation}.`);
    this.name = "VirtualPitchContractError";
    this.path = path;
  }
}

type JsonObject = Record<string, unknown>;

const GEOMETRY_CLASSES = ["official", "analytical", "optional"] as const;
const PITCH_ENDS = ["bowler", "striker", "both", "none"] as const;
const WICKET_ENDS = ["bowler", "striker"] as const;
const STUMP_INDEXES = ["left", "middle", "right"] as const;
const BAIL_INDEXES = ["left_middle", "middle_right"] as const;
const LANDMARK_CATEGORIES = ["wicket", "crease", "pitch", "analytical"] as const;
const LINE_CATEGORIES = [
  "pitch_boundary",
  "bowling_crease",
  "popping_crease",
  "return_crease",
  "centreline",
  "trajectory_grid",
  "coaching_guide"
] as const;
const POLYGON_CATEGORIES = [
  "pitch_surface",
  "pitch_boundary",
  "lbw_corridor",
  "trajectory_plane",
  "coaching_region"
] as const;

function object(value: unknown, path: string): JsonObject {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new VirtualPitchContractError(path, "object");
  }
  return value as JsonObject;
}

function array(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) throw new VirtualPitchContractError(path, "array");
  return value;
}

function string(value: unknown, path: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new VirtualPitchContractError(path, "non-empty string");
  }
  return value;
}

function boolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") throw new VirtualPitchContractError(path, "boolean");
  return value;
}

function finite(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new VirtualPitchContractError(path, "finite number");
  }
  return value;
}

function positive(value: unknown, path: string): number {
  const parsed = finite(value, path);
  if (parsed <= 0) throw new VirtualPitchContractError(path, "number greater than zero");
  return parsed;
}

function literal<const T extends readonly string[]>(
  value: unknown,
  allowed: T,
  path: string
): T[number] {
  if (typeof value !== "string" || !allowed.includes(value)) {
    throw new VirtualPitchContractError(path, allowed.map((item) => JSON.stringify(item)).join(" | "));
  }
  return value as T[number];
}

function nullableString(value: unknown, path: string): string | null {
  return value === null || value === undefined ? null : string(value, path);
}

function point(value: unknown, path: string): CricketVector3 {
  const item = object(value, path);
  return {
    x: finite(item.x, `${path}.x`),
    y: finite(item.y, `${path}.y`),
    z: finite(item.z, `${path}.z`)
  };
}

function uniqueIds(items: readonly { primitiveId?: string; semanticId?: string; profileId?: string | null }[], path: string) {
  const seen = new Set<string>();
  items.forEach((item, index) => {
    const id = item.primitiveId ?? item.semanticId ?? item.profileId;
    if (!id || seen.has(id)) throw new VirtualPitchContractError(`${path}[${index}]`, "unique semantic identifier");
    seen.add(id);
  });
}

function requireRenderableGeometry(
  stumps: readonly StumpModel[],
  bails: readonly BailModel[],
  lines: readonly PitchLineSegment[],
  polygons: readonly PitchPolygon[]
) {
  const stumpKeys = new Set(stumps.map((item) => `${item.end}:${item.stumpIndex}`));
  const bailKeys = new Set(bails.map((item) => `${item.end}:${item.bailIndex}`));
  if (stumpKeys.size !== 6 || stumps.length !== 6) {
    throw new VirtualPitchContractError("$.stumps", "one left, middle, and right stump at each end");
  }
  if (bailKeys.size !== 4 || bails.length !== 4) {
    throw new VirtualPitchContractError("$.bails", "two distinct bails at each end");
  }
  if (!polygons.some((item) => item.polygonCategory === "pitch_surface")) {
    throw new VirtualPitchContractError("$.polygons", "pitch_surface polygon");
  }
  for (const category of ["pitch_boundary", "bowling_crease", "popping_crease", "return_crease"] as const) {
    if (!lines.some((item) => item.lineCategory === category)) {
      throw new VirtualPitchContractError("$.line_segments", `${category} segment`);
    }
  }
  if (!lines.some((item) => item.lineCategory === "centreline")) {
    throw new VirtualPitchContractError("$.line_segments", "centreline segment");
  }
  if (!polygons.some((item) => item.polygonCategory === "lbw_corridor")) {
    throw new VirtualPitchContractError("$.polygons", "lbw_corridor polygon");
  }
}

function coordinateSystem(value: unknown): CoordinateSystem {
  const item = object(value, "$.coordinate_system");
  return {
    units: literal(item.units, ["metres"] as const, "$.coordinate_system.units"),
    handedness: literal(item.handedness, ["right_handed"] as const, "$.coordinate_system.handedness"),
    origin: literal(item.origin, ["bowler_end_middle_stump_base"] as const, "$.coordinate_system.origin"),
    xAxis: literal(item.x_axis, ["lateral_camera_neutral_right"] as const, "$.coordinate_system.x_axis"),
    yAxis: literal(item.y_axis, ["bowler_to_striker"] as const, "$.coordinate_system.y_axis"),
    zAxis: literal(item.z_axis, ["up"] as const, "$.coordinate_system.z_axis"),
    description: string(item.description, "$.coordinate_system.description"),
    offLegAssignment: literal(item.off_leg_assignment, ["not_assigned"] as const, "$.coordinate_system.off_leg_assignment")
  };
}

function dimensions(value: unknown): PitchDimensions {
  const item = object(value, "$.dimensions");
  return {
    pitchLengthM: positive(item.pitch_length_m, "$.dimensions.pitch_length_m"),
    pitchWidthM: positive(item.pitch_width_m, "$.dimensions.pitch_width_m"),
    wicketWidthM: positive(item.wicket_width_m, "$.dimensions.wicket_width_m"),
    stumpHeightM: positive(item.stump_height_m, "$.dimensions.stump_height_m"),
    stumpDiameterMinM: positive(item.stump_diameter_min_m, "$.dimensions.stump_diameter_min_m"),
    stumpDiameterMaxM: positive(item.stump_diameter_max_m, "$.dimensions.stump_diameter_max_m"),
    bowlingCreaseLengthM: positive(item.bowling_crease_length_m, "$.dimensions.bowling_crease_length_m"),
    poppingCreaseOffsetM: positive(item.popping_crease_offset_m, "$.dimensions.popping_crease_offset_m"),
    returnCreaseOffsetM: positive(item.return_crease_offset_m, "$.dimensions.return_crease_offset_m")
  };
}

function landmark(value: unknown, index: number): PitchLandmark {
  const path = `$.landmarks[${index}]`;
  const item = object(value, path);
  return {
    semanticId: string(item.semantic_id, `${path}.semantic_id`),
    point: point(item.point, `${path}.point`),
    geometryCategory: literal(item.geometry_category, LANDMARK_CATEGORIES, `${path}.geometry_category`),
    geometryClass: literal(item.geometry_class, GEOMETRY_CLASSES, `${path}.geometry_class`) as GeometryClass,
    end: literal(item.end, PITCH_ENDS, `${path}.end`) as PitchEnd,
    calibrationAnchor: boolean(item.calibration_anchor, `${path}.calibration_anchor`),
    description: string(item.description, `${path}.description`)
  };
}

function stump(value: unknown, index: number): StumpModel {
  const path = `$.stumps[${index}]`;
  const item = object(value, path);
  return {
    primitiveId: string(item.primitive_id, `${path}.primitive_id`),
    centre: point(item.centre, `${path}.centre`),
    radiusM: positive(item.radius_m, `${path}.radius_m`),
    heightM: positive(item.height_m, `${path}.height_m`),
    orientation: point(item.orientation, `${path}.orientation`),
    end: literal(item.end, WICKET_ENDS, `${path}.end`) as WicketEnd,
    stumpIndex: literal(item.stump_index, STUMP_INDEXES, `${path}.stump_index`),
    geometryClass: literal(item.geometry_class, ["official"] as const, `${path}.geometry_class`)
  };
}

function bail(value: unknown, index: number): BailModel {
  const path = `$.bails[${index}]`;
  const item = object(value, path);
  return {
    primitiveId: string(item.primitive_id, `${path}.primitive_id`),
    start: point(item.start, `${path}.start`),
    endPoint: point(item.end_point, `${path}.end_point`),
    radiusM: positive(item.radius_m, `${path}.radius_m`),
    end: literal(item.end, WICKET_ENDS, `${path}.end`) as WicketEnd,
    bailIndex: literal(item.bail_index, BAIL_INDEXES, `${path}.bail_index`),
    geometryClass: literal(item.geometry_class, ["official"] as const, `${path}.geometry_class`),
    cosmetic: item.cosmetic === true ? true : (() => { throw new VirtualPitchContractError(`${path}.cosmetic`, "true"); })()
  };
}

function line(value: unknown, index: number): PitchLineSegment {
  const path = `$.line_segments[${index}]`;
  const item = object(value, path);
  return {
    primitiveId: string(item.primitive_id, `${path}.primitive_id`),
    start: point(item.start, `${path}.start`),
    endPoint: point(item.end_point, `${path}.end_point`),
    lineCategory: literal(item.line_category, LINE_CATEGORIES, `${path}.line_category`),
    geometryClass: literal(item.geometry_class, GEOMETRY_CLASSES, `${path}.geometry_class`) as GeometryClass,
    lineWidthM: positive(item.line_width_m, `${path}.line_width_m`),
    end: literal(item.end, PITCH_ENDS, `${path}.end`) as PitchEnd,
    profileId: nullableString(item.profile_id, `${path}.profile_id`)
  };
}

function polygon(value: unknown, index: number): PitchPolygon {
  const path = `$.polygons[${index}]`;
  const item = object(value, path);
  const vertices = array(item.vertices, `${path}.vertices`).map((vertex, vertexIndex) => point(vertex, `${path}.vertices[${vertexIndex}]`));
  if (vertices.length < 3) throw new VirtualPitchContractError(`${path}.vertices`, "at least three points");
  const displayOpacity = finite(item.display_opacity, `${path}.display_opacity`);
  if (displayOpacity < 0 || displayOpacity > 1) throw new VirtualPitchContractError(`${path}.display_opacity`, "number from zero to one");
  return {
    primitiveId: string(item.primitive_id, `${path}.primitive_id`),
    vertices,
    polygonCategory: literal(item.polygon_category, POLYGON_CATEGORIES, `${path}.polygon_category`),
    geometryClass: literal(item.geometry_class, GEOMETRY_CLASSES, `${path}.geometry_class`) as GeometryClass,
    end: literal(item.end, PITCH_ENDS, `${path}.end`) as PitchEnd,
    profileId: nullableString(item.profile_id, `${path}.profile_id`),
    displayOpacity
  };
}

function profile(value: unknown, index: number): PitchProfile {
  const path = `$.profiles[${index}]`;
  const item = object(value, path);
  return {
    profileId: string(item.profile_id, `${path}.profile_id`),
    label: string(item.label, `${path}.label`),
    geometryClass: literal(item.geometry_class, GEOMETRY_CLASSES, `${path}.geometry_class`) as GeometryClass,
    description: string(item.description, `${path}.description`),
    enabledPrimitiveIds: array(item.enabled_primitive_ids, `${path}.enabled_primitive_ids`).map((id, idIndex) => string(id, `${path}.enabled_primitive_ids[${idIndex}]`)),
    universalOfficialGeometry: boolean(item.universal_official_geometry, `${path}.universal_official_geometry`)
  };
}

export function adaptVirtualPitchResponse(value: unknown): VirtualPitchModel {
  const root = object(value, "$");
  const modelVersion = literal(root.virtual_pitch_model_version, ["v1"] as const, "$.virtual_pitch_model_version");
  const landmarks = array(root.landmarks, "$.landmarks").map(landmark);
  const stumps = array(root.stumps, "$.stumps").map(stump);
  const bails = array(root.bails, "$.bails").map(bail);
  const lineSegments = array(root.line_segments, "$.line_segments").map(line);
  const polygons = array(root.polygons, "$.polygons").map(polygon);
  const profiles = array(root.profiles, "$.profiles").map(profile);
  uniqueIds(landmarks, "$.landmarks");
  uniqueIds(stumps, "$.stumps");
  uniqueIds(bails, "$.bails");
  uniqueIds(lineSegments, "$.line_segments");
  uniqueIds(polygons, "$.polygons");
  uniqueIds(profiles, "$.profiles");
  requireRenderableGeometry(stumps, bails, lineSegments, polygons);
  const rounding = object(root.display_rounding, "$.display_rounding");
  return {
    modelVersion,
    coordinateSystem: coordinateSystem(root.coordinate_system),
    dimensions: dimensions(root.dimensions),
    landmarks,
    stumps,
    bails,
    lineSegments,
    polygons,
    profiles,
    displayRounding: {
      storedPrecision: literal(rounding.stored_precision, ["full_float"] as const, "$.display_rounding.stored_precision"),
      displayDecimalPlaces: finite(rounding.display_decimal_places, "$.display_rounding.display_decimal_places"),
      displayUnits: literal(rounding.display_units, ["metres"] as const, "$.display_rounding.display_units")
    },
    syntheticCameraNames: array(root.synthetic_camera_names, "$.synthetic_camera_names").map((name, index) => string(name, `$.synthetic_camera_names[${index}]`))
  };
}

export const adaptVirtualPitchSpecification = adaptVirtualPitchResponse;
