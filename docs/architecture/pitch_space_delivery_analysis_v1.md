# Pitch-Space Delivery Analysis Lab V1

## Scope

Pitch-Space Delivery Analysis V1 is an isolated development workflow at
`/pitch-space-analysis` inside the existing CricVision Next.js application. It
reuses the existing upload/storage flow, stump detector, Wicket Observation V1,
Complete Delivery Tracking V2, FastAPI service, and Virtual Pitch V1. It does
not add another application, detector, tracker, calibration system, pitch model,
or Python environment. Video Analysis and Live Analysis are outside this
milestone.

The lab automatically establishes a two-dimensional pitch coordinate system for
a fixed-camera delivery, preserves original image observations, and reports
carefully labelled pitch-space estimates. It does not recover airborne height,
accept production calibration, or unlock 3D physics.

## Coordinate Ownership

`services/api/services/virtual_pitch_service.py` and
`packages/cricket_vision/calibration/cricket_pitch_geometry.py` remain the sole
owners of canonical pitch geometry. The lab consumes Virtual Pitch V1 rather
than copying dimensions into its backend adapter or frontend.

Canonical coordinates are metres in the existing right-handed system:

- origin: bowler-end middle-stump base;
- `+x`: pitch-right looking from bowler end to striker end;
- `+y`: bowler end toward striker end;
- `+z`: upward.

Image points retain frame index, timestamp, native pixel coordinates, detector
confidence, tracker provenance, and validity. Their pitch-space counterparts
add projected `x/y`, fit and combined confidence, bounds state, bounce phase,
and warnings. A ground-plane homography never converts an airborne observation
into a true 3D ball position.

## Setup-Frame Decision

Frame 0 is decoded and evaluated first. When both wickets are usable, Frame 0
is the setup frame even if another early frame has a slightly higher score.
Nearby frames may stabilise its boxes but cannot silently replace the visual
reference.

If Frame 0 fails, fallback candidates are deterministic early samples, normally
frames `5`, `10`, `15`, and `20`, or their deterministic timestamp equivalents
when FPS or clip length requires it. The first defensible candidate is used;
when candidates require ranking, stable score fields and frame index provide a
deterministic tie break. No random selection or full-delivery search is used
while usable early evidence exists.

The decision records the preferred attempt, Frame 0 evaluation, fallback list,
selected frame/timestamp, reasons, and quality score. Decode failure, blur,
exposure, clipping, missing wickets, implausible geometry, and role ambiguity
remain explicit evidence.

## Wicket Observation And Stabilisation

Valid persisted Wicket Observation V1 evidence is reused. Otherwise the
existing stump detector runs only on required setup/supporting frames. There is
no second detector loop. Missing or inseparable near/far wickets returns
`INSUFFICIENT_WICKETS`; the user is not asked for anchors.

Supporting boxes are stabilized independently by role using confidence-weighted
medians for centre, width, and height. Severe clipping and temporal outliers are
rejected. Support frames improve the boxes while the selected setup frame stays
the visual reference.

## Two-Wicket Pitch Fit

Each stable box contributes approximate left base, centre base, and right base
image points. Deterministic camera-end and pitch-left/right hypotheses map these
observations to the corresponding Virtual Pitch V1 wicket-base coordinates.
Hypotheses are scored for wicket agreement, polygon area and containment,
perspective order, delivery direction when available, non-collapse, and temporal
box consistency. Orientation is inferred; no orientation question, camera
height, field of view, or manual calibration is requested.

OpenCV produces `image_to_pitch_homography` and its inverse. Validation requires
finite matrices, a finite inverse, acceptable determinant and condition number,
round-trip/reprojection consistency, positive pitch polygon area, correct
near/far ordering, and no severe collapse or unsupported reflection. The full
overlay projects the existing Virtual Pitch V1 pitch surface, wickets, creases,
boundaries, centreline, and analysis corridor.

## Fixed-Camera Policy

The selected transform is fixed for the delivery. It is not recalibrated per
frame. At deterministic intervals, observed wicket centres and sizes are
compared with their expected locations.

- `FIXED_CAMERA`: retain normal estimate confidence.
- `MINOR_DRIFT`: continue while reducing confidence and recording drift.
- `UNSTABLE_CAMERA`: preserve image tracking but stop presenting reliable
  pitch-space metrics after the drift point.

The system never requests manual recalibration. Missing check-frame evidence is
reported rather than interpreted as proof of stability.

## Track And Metrics

Complete Delivery Tracking V2 remains the ball source. The lab transforms every
valid tracked centre while preserving `OBSERVED`, `RECOVERED`, and `PROJECTED`
provenance. Out-of-bounds or invalid transforms produce warnings or unavailable
points, not clipped metric values.

Bounce combines existing candidates with image trajectory, pitch progression,
local velocity/curvature, and direction change. It may return alternatives or
`BOUNCE_UNAVAILABLE`. Line is the bounce displacement from middle stump and uses
camera-neutral `PITCH_LEFT`, `PITCH_RIGHT`, or centre terminology. Length is
distance from the striker wicket and popping crease plus a centralized category.

Speed is `ESTIMATED_PLANAR_SPEED`, preferably from robust longitudinal movement
against timestamps over multiple reliable pre-bounce samples. Movement is
`ESTIMATED_LATERAL_MOVEMENT`, measured as robust lateral residual from an early
direction fit. Neither estimate claims aerodynamic cause, true path length, or
airborne height.

## Result And Partial Failure

`PitchSpaceDeliveryAnalysisV1` is versioned, strict, atomically persisted as
`reports/pitch_space_delivery_analysis_v1.json`, and carries source metadata,
setup evidence, stable wickets, fit diagnostics, both homographies, projected
geometry, camera stability, image/pitch tracks, metric evidence, confidence,
warnings, unavailable metrics, and stage timings.

Stage failures are independent. `NO_VIDEO`, `UPLOAD_FAILED`,
`FRAME_ZERO_UNUSABLE`, `INSUFFICIENT_WICKETS`, `PITCH_FIT_FAILED`,
`UNSTABLE_CAMERA`, `BALL_TRACK_UNAVAILABLE`, `BOUNCE_UNAVAILABLE`,
`SPEED_UNAVAILABLE`, and `MOVEMENT_UNAVAILABLE` cannot erase valid earlier
results. Development estimates remain distinct from production calibration:
production acceptance stays false, exact airborne 3D stays unavailable, and no
existing production metric lock is bypassed.

## Replay

The real-video replay and top-down Virtual Pitch replay share one time/frame
controller for play, pause, speed, and scrubbing. Both render the same track
point identity and distinguish observed from recovered/projected evidence. The
complete trail, trail-to-current-frame, recent fade, pre/post-bounce segments,
active ball, and bounce marker are display modes over one result rather than
independent analyses.

## Milestone Boundary

V1 is a fixed-camera, two-wicket, ground-plane analysis lab. It does not modify
normal Video Analysis or Live Analysis, request manual calibration, train
keypoints, reconstruct true 3D flight, estimate exact airborne height, or make
LBW decisions.
