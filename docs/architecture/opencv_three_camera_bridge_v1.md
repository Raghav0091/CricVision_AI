# OpenCV-Three.js Camera Bridge V1

## Scope

This bridge makes the existing Virtual Pitch V1 renderer reproduce an OpenCV
pinhole camera in native image pixels. It supports synthetic camera validation
and developer-only overlay of the canonical 3D pitch on a stored setup frame.
It does not solve cameras, accept calibration, unlock metrics, or integrate with
Video Analysis or Live Analysis.

## Camera Contract

The backend exposes read-only normalized responses at:

- `GET /video-analysis/virtual-pitch/camera-bridge?camera_name=...`
- `GET /video-analysis/{analysis_id}/camera-bridge`

The response records camera provenance, version, candidate ID, classification,
acceptance state, native width and height, full `K`, scalar intrinsics,
distortion, Rodrigues vector, rotation matrix, translation, camera centre,
near/far planes, setup-frame metadata, warnings, and the existing backend
projected geometry. Real-camera precedence is accepted scene snapshot, selected
refined scene candidate, selected real-registration candidate, then unavailable.
An unaccepted candidate remains explicitly unaccepted and `metrics_unlocked` is
always false.

The browser API adapter maps this transport object to the one renderer contract,
`CameraBridgeInput`. React components never interpret OpenCV axes or apply
individual sign changes.

## Coordinate Conventions

CricVision world is right-handed metres with the origin at the bowler-end
middle-stump base: `+x` pitch-right, `+y` bowler to striker, and `+z` up.
Three world uses the existing renderer mapping:

```text
S = CricVision world -> Three world
    [ 1  0  0  0 ]
    [ 0  0  1  0 ]
    [ 0 -1  0  0 ]
    [ 0  0  0  1 ]

(x, y, z) -> (x, z, -y)
```

OpenCV camera coordinates are `+x` right, `+y` down, and `+z` forward. Three
camera coordinates are `+x` right, `+y` up, and look along `-z`:

```text
C = OpenCV camera -> Three camera = diag(1, -1, -1, 1)
```

## Extrinsics

Stored OpenCV extrinsics use:

```text
X_camera_cv = R_cv * X_world_cv + t_cv
```

The bridge builds the complete Three world-to-camera matrix by multiplication,
not Euler-angle guessing:

```text
matrixWorldInverse_three = C * [R_cv | t_cv] * inverse(S)
matrixWorld_three        = inverse(matrixWorldInverse_three)
```

Camera position is the translation of `matrixWorld`. Camera forward is Three
camera `(0, 0, -1)` transformed as a direction by `matrixWorld`. The bridge
checks finiteness, invertibility, rotation determinant, and handedness.

## Projection Matrix

For native width `W`, height `H`, intrinsics `fx`, `fy`, `cx`, `cy`, skew `s`,
near `n`, and far `f`, the calibrated WebGL projection is:

```text
[ 2fx/W  -2s/W   1-2cx/W            0          ]
[   0     2fy/H  2cy/H-1            0          ]
[   0       0   -(f+n)/(f-n)  -2fn/(f-n)       ]
[   0       0       -1              0          ]
```

This preserves unequal focal lengths, off-centre principal points, skew,
portrait frames, and landscape frames. The renderer assigns both
`projectionMatrix` and its inverse directly. It never calls
`updateProjectionMatrix()` in calibrated mode.

The OpenCV reference projection remains:

```text
Xc = R * Xw + t
u = (fx * Xc + skew * Yc) / Zc + cx
v = fy * Yc / Zc + cy
```

## Distortion Policy

- `ZERO_DISTORTION`: exact pinhole rendering is supported.
- `PREUNDISTORTED_FRAME`: supported only when the background was undistorted
  with the same calibration; projection uses the corresponding pinhole model.
- `NONZERO_DISTORTION_UNSUPPORTED`: rendering is rejected with a warning.

V1 has no lens-distortion shader and never silently ignores non-zero distortion.

## Native And Display Mapping

Calibration and camera validation always operate in native image pixels. The
overlay stage reuses `containedMediaRect` from the scene-calibration UI to place
one aspect-correct media rectangle inside the responsive stage. The frame, SVG,
transparent WebGL canvas, and diagnostic SVG share those exact bounds. CSS
resize and device pixel ratio therefore change raster density, not geometry.
OpenCV pixels map to NDC with `x = 2u/W - 1`, `y = 1 - 2v/H`; inverse and contain
round trips are tested for portrait, landscape, letterboxing, desktop, mobile,
and DPR-independent dimensions.

## Validation

Deterministic tests project every supplied landmark through both the OpenCV
reference and converted Three matrices. Reports retain all points and record
depth, frame state, clipping, residuals, mean, median, RMSE, maximum error,
biases, mirror warning, end-reversal warning, and matrix finiteness.

Synthetic tests cover centred, offset, elevated, yawed, pitched, portrait,
landscape, off-centre, skewed, and unequal-focal cameras, plus stump, pitch,
crease, centreline, and corridor landmarks. Exact zero-distortion agreement is
below `1e-7 px` RMSE and maximum error in the deterministic suite, well inside
the 0.25 px / 0.75 px limits.

## Real-Frame Overlay

The lab loads setup-frame identity and camera state from the backend. It layers
the setup JPEG, transparent Three canvas, existing backend SVG projection, and
screen-space residual diagnostics. Orbit and development FOV/position controls
are disabled in calibrated modes. Rendering remains demand-driven with capped
DPR, no shadows, no post-processing, and no animation loop.

Bridge accuracy and calibration accuracy are different measurements. A near-zero
OpenCV-versus-Three residual proves conversion correctness. It does not prove
that the OpenCV camera matches the photographed scene. For
`analysis_20260728_120858_762989`, the bridge is exact for the stored pinhole
camera while the unaccepted scene candidate itself has approximately 3.2456 px
anchor RMSE.

## Camera Ownership And Responsive Stability

Each WebGL canvas has one `VirtualPitchCameraController` and one camera family.
Development mode owns a conventional FOV-based `PerspectiveCamera`; calibrated
mode owns a separate bridge-configured `PerspectiveCamera`. A camera object is
never shared between canvases. Orbit controls are mounted only for the
development family and have bounded distance and polar angles.

The calibrated camera is configured atomically before it is passed to R3F's
`Canvas`: projection, projection inverse, world matrix, world inverse, near,
far, and matrix-update policy are all complete. Its `manual` marker prevents
R3F's responsive `updateCamera` path from assigning a display aspect ratio and
calling `updateProjectionMatrix()`. The scene remains unmounted until the inner
controller confirms that `useThree().camera` is the exact owned instance and
that its pose and projection checksums still match the bridge. Source changes
create a fully configured replacement and remount the isolated canvas; ordinary
CSS resize, DPR changes, and overlay changes retain camera identity.

The giant-wicket defect was caused by calibrated mode mutating R3F's ordinary
default camera after Canvas mount without marking it manual. On a renderer or
container resize, R3F therefore treated it as a normal development camera,
replaced the calibrated projection with a FOV/aspect projection, and left the
independent screen-space diagnostics correctly positioned. The stability patch
removes that overwrite path and the temporary default-camera frame.

The lab reports the active camera UUID, family, readiness, camera/canvas counts,
controls state, native and displayed dimensions, pose/projection checksums, and
actual-camera RMSE. Every validation landmark includes OpenCV, bridge, and
active-renderer pixels plus both residuals. Invalid zero-sized contained bounds
suspend the stage instead of retaining or expanding a stale canvas rectangle.

## Future Boundaries

Video Analysis may later compose accepted calibration and current video frames
with this same camera contract. Live Analysis additionally needs drift checks
and lifecycle-safe recalibration. Neither surface changes in V1. Automatic
camera fitting belongs to Preset-Constrained One-Click Auto Registration V1.
