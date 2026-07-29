# Virtual Pitch Geometry and Synthetic Projection V1

## Scope

Virtual Pitch V1 is CricVision's permanent, parametric cricket-pitch model. It
generates metric landmarks and primitives and projects them through deterministic
synthetic cameras. It does not register, calibrate, or unlock metric analytics
for a real video.

The overlay is not a fixed image. The same world model is reused for every
camera; only camera intrinsics and pose change.

## Source Of Truth

`packages/cricket_vision/calibration/cricket_pitch_geometry.py` is the single
backend source of truth for official dimensions and the permanent coordinate
system. FastAPI serializes that model for the Next.js renderer. TypeScript does
not contain cricket dimensions or reproduce camera projection.

Model version: `v1`

Official/core dimensions, in metres:

| Quantity | Value |
| --- | ---: |
| Pitch length | 20.12 |
| Pitch width | 3.05 |
| Wicket width | 0.2286 |
| Stump height | 0.7112 |
| Stump diameter range | 0.0350-0.0381 |
| Bowling crease length | 2.64 |
| Popping crease offset | 1.22 |
| Return crease offset | 1.32 |

API display values use three decimal places. Full-precision floats remain
available for projection and camera-pose recovery.

## Coordinate Convention

The origin is the bowler-end middle-stump base. `+x` is camera-neutral right
when looking from the bowler end toward the striker, `+y` runs bowler to
striker, and `+z` is upward. This is right-handed. Off and leg sides are not
assigned because they depend on batter handedness.

Calibration V2 historically stores `(longitudinal_x, lateral_y, z)`. Explicit
`canonical_to_calibration_world` and `calibration_to_canonical_world` adapters
swap those two axes at that boundary. Physics Engine V1 consumes the canonical
constants and convention; its fitted equations are unchanged.

## Geometry

Official geometry includes the pitch surface and boundary, two mathematically
generated wicket systems, stumps, cosmetic bails, bowling creases, popping
creases, and return-crease registration spans. The supplied official dimensions
do not define the full rear extent of a return crease, so V1 represents the
unambiguous span between bowling and popping creases and labels it accordingly.

Analytical geometry includes the pitch centreline and an LBW stump-to-stump
corridor generated directly from wicket width. Optional training profiles are
separate and are not represented as universal Laws geometry.

Every landmark has a stable semantic ID, world coordinate, category, end,
calibration-anchor flag, and description. IDs include both wicket centres,
every stump base and top, pitch corners, both crease systems and intersections,
and centreline endpoints.

## Projection Pipeline

`GET /video-analysis/virtual-pitch` returns the permanent specification.

`GET /video-analysis/virtual-pitch/synthetic-projection` accepts a developer-only
synthetic camera name and profile. It returns camera metadata, projected
landmarks, lines, stumps, bails, polygons, depth/visibility state, and
perspective diagnostics.

Projection uses `cv2.projectPoints` with explicit camera matrices, distortion
coefficients, Rodrigues rotation vectors, and translation vectors. Points behind
the near plane are invalid; valid points outside the image are retained and
marked out of frame rather than clamped.

Synthetic cases cover centred bowler and striker views, left and right offsets,
low and elevated views, narrow and wide focal lengths, and portrait and
landscape frames.

## PnP Validation

Synthetic validation projects known anchors, recovers camera pose with
`solvePnPRansac`, optionally refines with `solvePnPRefineLM`, and reprojects the
model. Results include rotation error, translation error, reprojection RMSE,
inlier count, outlier IDs, and an explicit failure reason.

Automated tolerances are:

- perfect input: rotation and translation error below `0.001`, RMSE below
  `0.01 px`
- controlled pixel noise: rotation and translation error below `0.05`, RMSE
  below `0.75 px`
- one bad correspondence: recovered as an outlier with RMSE below `0.1 px`

Insufficient and geometrically degenerate anchor sets fail explicitly.

## Frontend Consistency

`VirtualPitchOverlay` consumes backend-projected native pixel coordinates and
renders an SVG with the camera image as its `viewBox`. `preserveAspectRatio`
uses contain semantics, so browser layout, letterboxing, portrait/landscape
orientation, and device pixel ratio do not alter the native-coordinate
relationship. The tested contain-mapping tolerance is floating-point
approximation at the native/display transform boundary; projected anchor
coordinates are not manually adjusted in the browser.

The developer panel provides camera selection, visibility, and opacity controls.
It always identifies the projection as synthetic and not registered to the
video. Metric analytics remain locked.

## Milestone Boundary

This milestone does not perform automatic stump extraction, real `solvePnP`,
manual drag calibration, real pitch-aligned replay, or real-video speed
measurement.

Automatic Wicket Registration V1 should next map detector-derived stump and
wicket observations to these stable semantic anchors, reject ambiguous or
degenerate configurations, solve and score camera pose on the backend, and
require explicit registration acceptance before enabling any metric output.
