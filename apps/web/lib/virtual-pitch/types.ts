export type GeometryClass = "official" | "analytical" | "optional";
export type PitchEnd = "bowler" | "striker" | "both" | "none";
export type WicketEnd = "bowler" | "striker";

export type WorldPoint3D = { x: number; y: number; z: number };
export type CricketVector3 = WorldPoint3D;
export type ThreeVector3 = { x: number; y: number; z: number };

export type CoordinateSystem = {
  units: "metres";
  handedness: "right_handed";
  origin: "bowler_end_middle_stump_base";
  xAxis: "lateral_camera_neutral_right";
  yAxis: "bowler_to_striker";
  zAxis: "up";
  description: string;
  offLegAssignment: "not_assigned";
};

export type PitchDimensions = {
  pitchLengthM: number;
  pitchWidthM: number;
  wicketWidthM: number;
  stumpHeightM: number;
  stumpDiameterMinM: number;
  stumpDiameterMaxM: number;
  bowlingCreaseLengthM: number;
  poppingCreaseOffsetM: number;
  returnCreaseOffsetM: number;
};

export type PitchLandmark = {
  semanticId: string;
  point: WorldPoint3D;
  geometryCategory: "wicket" | "crease" | "pitch" | "analytical";
  geometryClass: GeometryClass;
  end: PitchEnd;
  calibrationAnchor: boolean;
  description: string;
};

export type StumpPrimitive = {
  primitiveId: string;
  centre: WorldPoint3D;
  radiusM: number;
  heightM: number;
  orientation: WorldPoint3D;
  end: WicketEnd;
  stumpIndex: "left" | "middle" | "right";
  geometryClass: "official";
};
export type StumpModel = StumpPrimitive;

export type BailPrimitive = {
  primitiveId: string;
  start: WorldPoint3D;
  endPoint: WorldPoint3D;
  radiusM: number;
  end: WicketEnd;
  bailIndex: "left_middle" | "middle_right";
  geometryClass: "official";
  cosmetic: true;
};
export type BailModel = BailPrimitive;

export type PitchLineSegment = {
  primitiveId: string;
  start: WorldPoint3D;
  endPoint: WorldPoint3D;
  lineCategory:
    | "pitch_boundary"
    | "bowling_crease"
    | "popping_crease"
    | "return_crease"
    | "centreline"
    | "trajectory_grid"
    | "coaching_guide";
  geometryClass: GeometryClass;
  lineWidthM: number;
  end: PitchEnd;
  profileId: string | null;
};

export type PitchPolygon = {
  primitiveId: string;
  vertices: WorldPoint3D[];
  polygonCategory:
    | "pitch_surface"
    | "pitch_boundary"
    | "lbw_corridor"
    | "trajectory_plane"
    | "coaching_region";
  geometryClass: GeometryClass;
  end: PitchEnd;
  profileId: string | null;
  displayOpacity: number;
};

export type PitchProfile = {
  profileId: string;
  label: string;
  geometryClass: GeometryClass;
  description: string;
  enabledPrimitiveIds: string[];
  universalOfficialGeometry: boolean;
};

export type VirtualPitchModel = {
  modelVersion: "v1";
  coordinateSystem: CoordinateSystem;
  dimensions: PitchDimensions;
  landmarks: PitchLandmark[];
  stumps: StumpPrimitive[];
  bails: BailPrimitive[];
  lineSegments: PitchLineSegment[];
  polygons: PitchPolygon[];
  profiles: PitchProfile[];
  displayRounding: {
    storedPrecision: "full_float";
    displayDecimalPlaces: number;
    displayUnits: "metres";
  };
  syntheticCameraNames: string[];
};

export type CameraPresetId =
  | "setup"
  | "bowler-end"
  | "striker-end"
  | "side"
  | "top-down"
  | "free-orbit";

export type CameraPreset = {
  id: CameraPresetId;
  label: string;
  synthetic: true;
  position: ThreeVector3;
  target: ThreeVector3;
  up: ThreeVector3;
  verticalFovDegrees: number;
  near: number;
  far: number;
  orbitEnabled: boolean;
};

export type CameraAdjustments = {
  heightM?: number;
  distanceBehindM?: number;
  lateralOffsetM?: number;
  verticalFovDegrees?: number;
  targetHeightM?: number;
  yawDegrees?: number;
  pitchDegrees?: number;
  rollDegrees?: number;
};

export type ResolvedCameraAdjustments = Required<CameraAdjustments>;

export type MaterialPresetId = "cricvision-dark" | "broadcast-light" | "debug-wireframe";
export type MaterialStyle = {
  color: string;
  opacity: number;
  roughness: number;
  metalness: number;
  wireframe: boolean;
  transparent: boolean;
  depthWrite: boolean;
};
export type VirtualPitchMaterialPreset = {
  id: MaterialPresetId;
  label: string;
  background: string;
  ambientLight: string;
  keyLight: string;
  pitch: MaterialStyle;
  stump: MaterialStyle;
  bail: MaterialStyle;
  officialLine: MaterialStyle;
  analyticalLine: MaterialStyle;
  corridor: MaterialStyle;
};
