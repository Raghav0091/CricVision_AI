# Physics and Delivery Analytics Engine V1

## Integration

Physics Engine V1 is an optional stage inside the existing offline Video
Analysis job:

`Next.js Video Analysis -> FastAPI -> selected E2/E3/E4C detector -> Complete
Delivery Tracking v2 -> Physics Engine V1 -> existing replay and result UI`

The engine consumes persisted every-frame detections and the tracker's primary
track. It does not rerun detection, alter detector weights or thresholds, or
change the tracker. A physics exception produces a `FAILED` physics result
while preserving the successful detector and tracker result.

The result is embedded in `tracking_result.json` and also written to
`tracking/physics_result.json`. The existing delivery replay renderer receives
the fitted pixel path. The Next.js result panel renders analytics and a
top-down SVG from the same backend trajectory samples.

## Coordinates

Physics V1 uses one right-handed coordinate system:

- Origin: centre of the bowler's wicket on the pitch surface.
- `x`: lateral across the pitch, positive toward calibration world-left.
- `y`: longitudinal from the bowler's wicket toward the striker's wicket.
- `z`: height above the pitch.

Existing calibration V2 stores longitudinal, lateral and up as `x`, `y`, `z`.
The physics service swaps the first two axes at its boundary. Official
dimensions are 20.12 m pitch length, 3.05 m pitch width, 0.2286 m wicket
width, and 0.7112 m stump height.

Line side labels follow the confirmed calibration convention. Batter
handedness is not independently inferred. Line and length bands are
centralized CricVision coaching categories, not MCC definitions.

## Calibration Modes

`METRIC_3D` requires an accepted wicket camera-pose solution with rotation,
translation, camera intrinsics and reprojection diagnostics. World points are
projected with OpenCV `projectPoints`.

`METRIC_GROUND_PLANE` requires a confirmed calibration V2 homography. It can
provide metric bounce position, line and length. It does not provide airborne
height, 3D speed or metric lateral movement.

`IMAGE_SPACE_ONLY` is the fallback when metric calibration is unavailable. It
provides a robust fitted pixel trajectory and provenance, while all metric
analytics remain explicitly unavailable.

Calibration evidence is loaded from the existing calibration services. A
loose stump box is never promoted to exact stump keypoints by this engine.

## Observation Contract

Only `OBSERVED` tracker points become canonical detector observations.
Frame index, timestamp, pixel centre, detector confidence, tracker confidence,
candidate ID and detector bounding box are preserved when available. Duplicate
frames, invalid timestamps, very low confidence and impossible pixel jumps are
rejected with reason codes. The reprojection fit performs a second controlled
outlier pass and exposes all inlier and outlier frames.

A metric fit requires at least six reliable observations. Image fitting
requires at least three. Missing or schema-incompatible historical artifacts
are reported separately by the validation tool.

## Model Hierarchy

The pre-bounce hierarchy is:

1. `BALLISTIC`: constant lateral and forward velocity with fixed gravity.
2. `BALLISTIC_LATERAL`: adds bounded effective lateral acceleration.
3. `BALLISTIC_LATERAL_DECELERATION`: also adds bounded forward deceleration.

The coordinate equations use gravity `g = 9.81 m/s^2`. A more complex model is
accepted only when reprojection RMSE improves materially and no new parameter
reaches a bound.

Central bounds are:

- initial lateral position: -2.5 to 2.5 m
- initial longitudinal position: -1 to 21 m
- initial height: 0.15 to 3.5 m
- lateral velocity: -10 to 10 m/s
- forward velocity: 4 to 50 m/s
- vertical velocity: -12 to 15 m/s
- effective lateral acceleration: -20 to 20 m/s^2
- forward acceleration: -15 to 0 m/s^2

The image-only fallback uses deterministic confidence-weighted quadratic fits.
When the tracker supplies a credible bounce, pre-bounce and post-bounce image
segments are fitted separately.

## Optimisation And Outliers

Metric fitting uses SciPy `least_squares` with bounded parameters, deterministic
initialization, `soft_l1` robust loss and confidence-weighted pixel residuals.
Physical regularization discourages implausible initial height, lateral speed,
lateral acceleration and deceleration. A median absolute deviation gate then
removes reprojection outliers before one bounded refit.

Diagnostics include convergence, optimizer status, iterations, weighted
reprojection RMSE, median error, maximum inlier error, inlier/outlier frames,
reached bounds and processing duration.

## Bounce And Post-Bounce

The metric bounce is the bounded positive root where the fitted pre-bounce
height reaches the pitch plane. Confidence increases when an observed point is
near that frame and when the existing tracker slope transition agrees. Bounce
outside pitch bounds is rejected.

Post-bounce fitting is a separate segment and requires at least four observed
post-bounce points. It fits bounded lateral, forward and upward velocity from
the bounce point. With less evidence, continuation is `PROJECTED` and measured
post-bounce turn is unavailable.

## Metrics

Speed is the norm of fitted world velocity, converted using `km/h = m/s * 3.6`.
The UI calls the first supported value `Earliest measured speed`, never release
speed. Average pre-bounce speed and speed approaching bounce come from the
fitted pre-bounce segment. Values outside 4 to 50 m/s are withheld.

Pre-bounce lateral movement is the displacement against the same initial state
with lateral acceleration set to zero. Direction is reported in coordinate
terms; it is not evidence of conventional swing, reverse swing or an
aerodynamic cause.

Post-bounce turn and speed loss are measured only with the separate observed
post-bounce fit. Exact spin RPM is always unavailable because ordinary tiny,
blurred detections do not directly observe ball surface rotation. Exact seam
angle is not measured.

Line and length use the metric bounce point. Raw bounce coordinates and
distance from the striker's wicket remain in the result alongside coaching
categories.

## Provenance And Confidence

Every frame in the bounded delivery interval is one of:

- `OBSERVED`: an accepted tracker/detector observation supports that frame.
- `RECONSTRUCTED`: a short internal gap bounded by observations.
- `PROJECTED`: after the final supporting observation.

Each sample records the nearest observation and frame/time distance.
Projection confidence decays exponentially with unsupported duration. The
terminal condition is the striker wicket plane when reached, otherwise a
maximum 0.35 second continuation.

Overall confidence combines calibration quality, observation count and
coverage, fit error and rejected evidence. Metric-level results include
availability reasons and deterministic residual or bounded-parameter
uncertainty estimates. These grades are evidence quality, not calibrated
probabilities.

## Validation

Run persisted-delivery diagnostics without rerunning detection:

```powershell
python scripts\validate_delivery_physics_v1.py <analysis_id> [<analysis_id> ...]
```

Outputs are written under `outputs/physics_validation_v1/` and remain generated
developer artifacts.

## Limitations

Single-camera monocular fitting is depth-sensitive and can be weakly
identifiable even with a good image fit. Full metric speed requires an accepted
3D camera pose; a ground homography alone is insufficient for airborne speed.
Projected paths are estimates. Bounce and post-bounce analytics depend on the
quality and temporal coverage of the upstream track.

Exact seam angle is not measured. Exact spin RPM is not measured from ordinary
clips. Single-camera outputs are coaching analytics, not umpiring-grade
decisions.

The same contracts can later support live analysis by feeding timestamped
observations incrementally, but V1 deliberately remains an offline post-track
stage.
