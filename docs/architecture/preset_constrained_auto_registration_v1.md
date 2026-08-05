# Preset-Constrained One-Click Auto Registration V1

## Repository evidence and reuse

This milestone extends the existing scene-calibration architecture. It does not
introduce a second detector, pitch model, pose solver, or renderer bridge.

- Wicket Observation V1 persists the setup frame, supporting frames, raw
  detections, stable near/far wicket regions, landmarks, confidence,
  uncertainty, clipping factors, and rejection status in
  `reports/wicket_observations_v1.json`.
- Real Pitch Registration V1 converts exact, pointlike, and soft wicket evidence
  into weighted correspondences; generates deterministic focal and assignment
  candidates; runs OpenCV/SciPy pose refinement; and reports plausibility,
  temporal validation, uncertainty, and projected pitch geometry.
- Assisted Scene Calibration V1 owns manual anchors, refinement, readiness
  thresholds, orientation resolution, user acceptance, and metric locking.
- Virtual Pitch V1 remains the only owner of official geometry.
- The OpenCV-Three.js bridge remains the only renderer-camera conversion path.

Automatic registration therefore loads persisted Wicket Observation V1 evidence
by default, calls existing registration/refinement utilities with preset bounds,
and returns its camera through `CameraBridgeInput`. Redetection is allowed only
when explicitly requested. Soft detector boxes remain soft constraints and are
never promoted to exact physical landmarks.

## Contracts

`services/api/schemas/preset_auto_registration.py` owns the versioned
`CameraSetupPreset`, compatibility models, status enums, diagnostics, and
`PresetAutoRegistrationResult`. Models reject extra fields and non-finite
numbers. Parameter names carry units (`_m`, `_deg`, `_px`, or `_ms`).
`PresetAutoRegistrationRunRequest` fixes the orchestration defaults to reuse
persisted observations, avoid redetection, and keep development diagnostics off.
`CameraSetupPresetListResponse` provides the versioned preset catalogue.
Services and routes use deterministic `list_camera_setup_presets()` and
`get_camera_setup_preset(preset_id)` accessors; the underlying catalogue and
preset constraints are immutable. `PresetCompatibilityInput` is the strict,
typed pre-fit evidence shape, and compatibility reasons use a closed reason-code
enum rather than arbitrary dictionaries.

`PresetAutoRegistrationResult.preset_auto_registration_version` is `v1`.
Automatic status is independent of geometric classification:

- `NOT_STARTED`
- `PRESET_INCOMPATIBLE`
- `INSUFFICIENT_WICKETS`
- `INSUFFICIENT_EVIDENCE`
- `FITTING`
- `AUTO_REGISTRATION_READY`
- `VISUAL_OVERLAY_READY`
- `NEEDS_ASSISTANCE`
- `FAILED`

Geometric classification is `METRIC_3D_CANDIDATE`,
`GROUND_PLANE_CANDIDATE`, `VISUAL_ONLY`, or `REGISTRATION_FAILED`.
`AUTO_REGISTRATION_READY` means an automatic candidate and stable overlay are
ready for the next integration step. It is not production acceptance.
`production_accepted` is always false and `metrics_unlocked` is always empty.

The result reuses `SetupFrameCandidate`, `CameraPoseCandidate`,
`RealProjectedPitchGeometry`, and `CameraBridgeInput`. It adds preset
compatibility, attempted-candidate summaries, initial/fitted parameters,
parameter deltas, active bounds, anchor/envelope/temporal metrics, physical
checks, uncertainty, ambiguity, timings, warnings, and failure reasons.

## Development preset

`STANDARD_REAR_WICKET_NET_V1` is development-only and assumes a fixed tripod
behind the bowler-end wicket, both wickets visible, and FullTrack-style
practice-net framing. Its reference clip is portrait `720 x 1280`; the strongest
  existing refined solution is approximately `0.053 m` lateral, `7.74 m` behind
  the wicket, and `1.39 m` high. Its persisted candidate label says `45 deg`, but
  final `K` has an effective `21.9609 deg` horizontal FOV. Final camera matrices,
  not seed labels, are authoritative. These observations inform broad priors only.

| Constraint | Nominal | Bounds |
| --- | ---: | ---: |
| Camera height | `1.5 m` | `0.75..3.0 m` |
| Distance behind near wicket | `8.0 m` | `3.0..15.0 m` |
| Lateral offset | `0.0 m` | `-2.5..2.5 m` |
| Yaw | `0 deg` | `-15..15 deg` |
| Pitch | `-4 deg` | `-18..8 deg` |
| Roll | `0 deg` | `-8..8 deg` |
| Horizontal FOV | `45 deg` | `25..80 deg` |
| Long-edge / short-edge ratio | n/a | `1.25..2.25` |

Yaw rotates around CricVision `+z`; positive yaw turns toward `+x`. Pitch is
camera elevation relative to the horizontal pitch direction, with downward
pitch negative. Roll is clockwise around the optical axis when viewed from the
camera. Distance is positive behind the preset near wicket; lateral offset
follows CricVision `+x`.

The exact inverse camera conventions, solver calibration, objective ablations,
and build-stability findings are documented in
`auto_registration_solver_calibration_patch_v1.md`.

The preset accepts native portrait or landscape orientation, uses the
`official_core` pitch profile, fixes the camera end to `bowler`, and defines
image left as pitch left. It requires both wickets, at least three supporting
frames in addition to the setup frame, and minimum per-wicket confidence `0.35`.
The preset policy is `ZERO_DISTORTION`. Compatibility may also accept a frame
that has already been undistorted through the existing bridge contract, but the
automatic solver never estimates distortion. The preset does not claim
identical height, distance, FOV, crop, or lens characteristics.

## Compatibility

Compatibility is checked before fitting and returns `COMPATIBLE`,
`COMPATIBLE_WITH_WARNINGS`, or `INCOMPATIBLE` with structured reasons. Checks
cover native pixel dimensions, orientation, long/short aspect ratio, rotation
metadata, distortion, trusted camera end, both wickets, setup/supporting frames,
observation validity, severe clipping, nested false-wicket evidence, and
unsupported crop or rotation.

An error makes input incompatible and fitting must not run. Warnings permit
bounded fitting but remain visible. Missing setup evidence, required wickets,
unsupported distortion, severe clipping, or unsupported crop/rotation must not
be hidden by preset priors.

## Bounded fitting and evidence

The optimiser may vary only camera height, distance behind the near wicket,
lateral offset, yaw, pitch, roll, horizontal FOV, and a tightly bounded
principal-point correction when separately justified. Pitch geometry, wicket
dimensions, image dimensions, camera end, lateral mapping, and distortion policy
remain fixed.

Candidate ordering must be deterministic and may reuse nominal/perturbed preset
poses and existing PnP/refined candidates. A robust objective should combine
confidence-weighted exact or pointlike reprojection, soft wicket envelopes and
lines, temporal agreement, camera/focal priors, and physical plausibility.
Preset priors guide weak evidence but cannot override contradictory observations.
Coarse detector intersections, lines, and repeated envelopes are correlated soft
evidence. V1 applies semantic uncertainty floors and correlation normalization;
the solver uses normalized `[-1, 1]` variables and the central preset/OpenCV
camera conversion utility.

## Temporal, physical, and uncertainty validation

One fixed camera is evaluated across supporting frames. Diagnostics report
near/far IoU, centre/width/height residuals, scale consistency, the worst frame,
and temporal stability. Physical checks cover camera bounds, angle/FOV bounds,
positive depth, pitch-facing direction, perspective ordering, projected scale,
scene containment, critical bound hits, competing solutions, and uncertainty.

Deterministic perturbation varies wicket coordinates and boxes within their
uncertainty plus bounded preset starts. It reports camera-position, rotation, and
FOV spread; projected wicket and pitch-corner movement; future bounce-location
sensitivity; and candidate-order stability. Low residual error with unstable
perturbations is downgraded.

## Readiness, fallback, and rendering

Automatic results never write an accepted snapshot. Physics and metric analytics
remain locked. `AUTO_REGISTRATION_READY` requires successful automatic fitting,
adequate temporal/physical/uncertainty checks, and no manual anchors.
`VISUAL_OVERLAY_READY` permits a useful visual overlay without metric trust.
`NEEDS_ASSISTANCE` may reveal the existing six-anchor workflow without modifying
its evidence.

Rendering is backend camera result -> existing `CameraBridgeInput` -> calibrated
camera controller -> existing Virtual Pitch renderer. The lab is the first
consumer; normal Video Analysis and Live Analysis remain unchanged. Future
integration may reuse a compatible fixed-camera preset, but current-video
observation, bounded refinement, validation, and explicit acceptance remain
required.

## Limitations

V1 is constrained to fixed rear-wicket practice-net cameras with both wickets
visible. It does not optimise distortion, support arbitrary crops or rotations,
infer missing wickets, accept production calibration, or activate trajectory and
physics metrics.

## Wicket landmark evidence upgrade

The developer-only V1 landmark upgrade is documented in
`wicket_landmark_evidence_upgrade_v1.md`. It adds native multi-frame physical
line evidence behind an explicit solver mode while preserving this solver,
legacy coarse mode, readiness thresholds, production locks, and camera bridge.
The strongest evaluation recovered six shaft axes but no supported endpoints or
scene lines; axis-only fitting remained ambiguous and did not improve readiness.
