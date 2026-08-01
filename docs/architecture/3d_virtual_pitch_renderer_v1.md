# 3D Virtual Pitch Renderer V1

## Scope

The 3D Virtual Pitch Renderer V1 is a standalone development surface inside the
existing Next.js application at `/virtual-pitch-lab`. It renders the permanent
Virtual Pitch V1 model with genuine Three.js geometry and a perspective camera.

V1 does not consume video, calibration, detections, trajectories, physics
results, or live-camera state. It does not project the pitch into image pixels.
Those integrations remain later milestones.

## Architectural Responsibilities

The renderer is split into four boundaries:

1. The backend owns canonical cricket dimensions, semantic geometry, and model
   versioning.
2. The frontend API boundary fetches and validates the canonical response.
3. A coordinate adapter converts canonical cricket vectors to Three.js vectors.
4. React Three Fiber components construct and style the validated primitives.

The lab page owns controls and development state. Render components receive a
validated model and do not fetch data, define official dimensions, or know about
video analysis.

## Backend Source Of Truth

`packages/cricket_vision/calibration/cricket_pitch_geometry.py` remains the only
source of official dimensions and the canonical coordinate convention.
`services/api/services/virtual_pitch_service.py` converts those constants into
renderable semantic primitives. `GET /video-analysis/virtual-pitch` serializes
the model through `VirtualPitchSpecification`.

The frontend must not duplicate pitch length, pitch width, wicket dimensions,
crease offsets, stump positions, or line positions. A renderer may define
visual-only values such as colour, roughness, line elevation used to avoid
z-fighting, and camera-control limits. These values do not become cricket
geometry.

The developer-only `GET /video-analysis/virtual-pitch/synthetic-projection`
endpoint is not an input to the 3D lab. It produces OpenCV pixel projections for
projection and registration testing, while the 3D renderer consumes world-space
geometry directly.

## Backend Contract Audit

The existing `GET /video-analysis/virtual-pitch` contract is sufficient for the
V1 renderer. No backend schema, service, or route change is required.

| Renderer requirement | Existing contract | Result |
| --- | --- | --- |
| Model version | `virtual_pitch_model_version` | Sufficient |
| Coordinate convention | `coordinate_system` | Sufficient |
| Pitch dimensions | `dimensions` | Sufficient |
| Pitch surface | `polygons[pitch_surface]` | Sufficient |
| Two wickets | `stumps[].end` and `bails[].end` | Sufficient |
| Six stump bodies | Six `StumpPrimitive` records | Sufficient |
| Stump size and pose | `centre`, `radius_m`, `height_m`, `orientation` | Sufficient |
| Four bails | Four `BailPrimitive` records | Sufficient |
| Bail size and pose | `start`, `end_point`, `radius_m` | Sufficient |
| Bowling creases | `line_segments[bowling_crease]` | Sufficient |
| Popping creases | `line_segments[popping_crease]` | Sufficient |
| Return creases | `line_segments[return_crease]` | Sufficient |
| Pitch boundaries | `line_segments[pitch_boundary]` | Sufficient |
| Centreline | `line_segments[centreline]` | Sufficient |
| Analysis corridor | `polygons[lbw_corridor]` | Sufficient |
| Stable object identity | `primitive_id` and landmark `semantic_id` | Sufficient |

The endpoint currently returns 36 landmarks, 6 stumps, 4 bails, 11 line
segments, and 2 polygons. The stable primitive IDs include:

- `bowler_left_stump`, `bowler_middle_stump`, `bowler_right_stump`
- `striker_left_stump`, `striker_middle_stump`, `striker_right_stump`
- `bowler_left_middle_bail`, `bowler_middle_right_bail`
- `striker_left_middle_bail`, `striker_middle_right_bail`
- `bowler_bowling_crease`, `bowler_popping_crease`
- `striker_bowling_crease`, `striker_popping_crease`
- four end-and-side-specific return-crease registration spans
- `pitch_left_boundary`, `pitch_right_boundary`, `pitch_centerline`
- `pitch_surface`, `lbw_stump_to_stump_corridor`

The return creases intentionally represent only the unambiguous registration
span between bowling and popping creases. The renderer must display the supplied
segments and must not invent extensions.

## Frontend Geometry Adapter

The API client may retain its transport type, but render code consumes a
validated `VirtualPitchModel`. Validation occurs once at the API boundary and
must reject incomplete or non-finite geometry with a useful error state.

The adapter validates at least:

- model version and canonical coordinate metadata;
- unique, non-empty semantic and primitive IDs;
- positive dimensions, stump sizes, bail radii, and line widths;
- exactly six stumps and four bails for model `v1`;
- finite coordinates for every point and orientation;
- at least three vertices for each polygon;
- presence of the pitch surface, required crease categories, boundaries,
  centreline, and analytical corridor;
- references to known geometry classes, ends, and profile IDs.

It preserves backend IDs and numeric values. It may group primitives by
category for rendering, but it must not derive a second official pitch model.
The dimensions object is metadata for display and validation, not a source from
which React components rebuild geometry already supplied by the backend.

## Coordinate Mapping

CricVision world coordinates are right-handed metres:

- origin: bowler-end middle-stump base;
- `+x`: pitch-right looking from bowler to striker;
- `+y`: bowler end toward striker end;
- `+z`: upward.

The Three.js scene uses `+x` horizontally right, `+y` upward, and a camera-facing
depth convention in which the striker direction is `-z`. The only permitted
mapping is centralized in the coordinate adapter:

```text
CricVision (x, y, z) -> Three.js (x, z, -y)
Three.js   (x, y, z) -> CricVision (x, -z, y)
```

Individual components must not swap or negate axes. Tests cover the origin,
lateral direction, bowler-to-striker direction, vertical direction, both wicket
ends, full-pitch length, and round-trip conversion.

## Component Structure

The renderer should use focused components under
`apps/web/components/virtual-pitch/`:

- `VirtualPitchCanvas` owns the React Three Fiber canvas, DPR cap, demand-based
  rendering, resize behavior, and WebGL fallback.
- `VirtualPitchScene` composes a validated model with visual and camera options.
- `VirtualPitchSurface` triangulates the backend pitch polygon.
- `VirtualWicket`, `VirtualStump`, and `VirtualBails` render backend primitives.
- `VirtualPitchLines` renders supplied line segments as thin strip meshes.
- `VirtualPitchCorridor` triangulates analytical polygons with transparent
  materials and disabled depth writing where needed.
- `VirtualPitchCamera` applies synthetic development presets and orbit controls.
- `VirtualPitchLighting` provides restrained ambient and directional lighting.
- `VirtualPitchDebugHelpers` contains development-only axes, grid, labels, and
  measurements.

Stumps are cylinders aligned between their backend-defined centre, orientation,
and height. Bails are cylinders or capsule-like meshes aligned between supplied
endpoints. Components must not assume all future primitives are axis-aligned.
Line strips sit a small visual offset above coplanar surfaces to avoid
z-fighting; the offset is renderer policy, not canonical geometry.

## Renderer API And Modes

The reusable entry point should have a mode-aware API:

```tsx
<VirtualPitchCanvas
  model={pitchModel}
  mode="development"
  camera={cameraConfiguration}
  visualOptions={visualOptions}
/>
```

The mode type may reserve `video-overlay`, `live-overlay`, and
`interactive-replay`, but V1 implements only `development`. Reserved modes must
not simulate integrations or accept fake trajectory/video data.

## Camera Presets

All preset calculations use canonical model geometry and one coordinate adapter.
They must not repeat the official pitch length or wicket coordinates.

- **Synthetic Setup Camera:** behind the bowler wicket, approximately 1.2 to
  1.3 metres high, aimed toward the striker wicket, with a perspective suitable
  for portrait framing. It is explicitly synthetic, never calibrated.
- **Bowler-End View:** looks from behind the bowler end toward `+y`.
- **Striker-End View:** looks from behind the striker end toward `-y`.
- **Side View:** looks across the pitch with both wickets in frame.
- **Top-Down View:** looks down along canonical `-z` before coordinate mapping.
- **Free Orbit:** starts from a defined preset and enables development controls.

Camera configuration supports height, distance behind the wicket, lateral
offset, field of view, target height, yaw, pitch, and roll. Reset reconstructs
the selected preset from the current backend model and viewport aspect ratio.
Portrait and landscape framing are calculations, not separate geometry.

## Visual Ownership

Materials and colours belong to the frontend and are centralized independently
from geometry. V1 supports CricVision Dark, Broadcast Light, and Debug Wireframe
presets. These presets may style the pitch, lines, wickets, corridor, lighting,
and background, but cannot change dimensions or semantic identity.

## Performance Decisions

The scene is static by default:

- use `frameloop="demand"` where controls permit;
- cap device pixel ratio between 1 and 2;
- memoize parsed geometry, triangulation, and stable materials;
- avoid geometry or material allocation in frame callbacks;
- avoid post-processing and high-resolution shadows;
- use stable backend primitive IDs as React keys;
- dispose replaced Three.js resources;
- provide a low-performance mode with simpler materials, no shadows, and lower
  DPR.

The lab must expose a clear fallback when WebGL or the geometry dependency fails.

## Lab Page Boundary

`/virtual-pitch-lab` is a development-only page inside the existing application
shell. It fetches the canonical endpoint, validates it, and passes the model into
the renderer. It owns viewport orientation, camera controls, visibility toggles,
material preset selection, reset behavior, and information panels.

The page does not import calibration, detector, physics, or trajectory services.
A temporary navigation item belongs under an existing developer/tools area and
must not imply that the lab is a finished user feature.

## Why Unity Is Not Used

The current product is a Next.js application and V1 is a small, mostly static
web scene. React Three Fiber integrates Three.js with the existing React state,
layout, API client, and build pipeline. Unity would introduce a second runtime,
asset pipeline, deployment path, coordinate boundary, and application shell
without improving this milestone's geometry or interaction needs.

## Future Integration Boundaries

### OpenCV To Three.js Camera Conversion

The implemented camera bridge is documented in
[`opencv_three_camera_bridge_v1.md`](opencv_three_camera_bridge_v1.md). It owns
the full intrinsics and extrinsics conversion, distortion policy, numerical
validation, and developer-only real-frame overlay. Primitive components and
camera presets remain free of camera-conversion mathematics.

### Video Analysis

A future real-frame overlay may compose a video texture or DOM video layer with
the renderer after calibration acceptance. The standalone model and coordinate
adapter remain reusable; the Video Analysis page supplies camera state and frame
dimensions without redefining pitch geometry.

### Live Analysis

Live integration additionally requires camera drift monitoring, lifecycle-safe
media handling, and controlled recalibration. The V1 lab has no live-camera
dependency and does not change Live Analysis behavior.

### Ball Replay

A future replay layer consumes canonical trajectory samples from Physics Engine
outputs and converts them through the same coordinate adapter. Ball meshes,
bounce markers, timelines, and playback state remain separate from the static
pitch model. The renderer must never infer or recompute physics.

## V1 Decision

The existing backend contract is renderer-ready. The implementation work for
this milestone belongs entirely in the frontend, except for ordinary API-client
typing. Backend geometry, calibration, physics, detection, and projection code
must remain unchanged.
