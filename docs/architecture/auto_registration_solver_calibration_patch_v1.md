y
# Auto-registration Solver Calibration and Build Stability Patch V1

## Scope and checkpoint

The Preset-Constrained One-Click Auto Registration V1 implementation was
checkpointed on `main` as `5bfb0f230b25e75096b15ec6fefa954575a82581`
(`feat: add preset constrained auto registration`). This patch remains
uncommitted and does not change Video Analysis, Live Analysis, detector
behaviour, production acceptance, or metric locking.

## Root cause

The height-bound failure had two independent causes.

1. The persisted assisted camera reported a `45 deg` FOV label after focal
   refinement, while its final `K` (`fx=fy=1855.422582 px` at width `720`) has
   an effective horizontal FOV of `21.9608616 deg`. The automatic seed trusted
   the stale label and reconstructed a different camera.
2. Coarse detector-derived intersections, lines, and repeated wicket envelopes
   were correlated observations of the same regions. The legacy objective
   treated their small nominal uncertainties as independent exact evidence.
   At the assisted camera, coarse points contributed `5106.594` loss and lines
   `1702.871`; together they were 99.4% of the data loss and pulled all fits to
   the height ceiling.

The corrected solver derives FOV from `K`, applies semantic uncertainty floors
to coarse evidence, normalizes repeated evidence by correlation count, and
optimizes every physical variable in normalized `[-1, 1]` coordinates. It does
not increase bounds or reduce readiness thresholds.

## Camera parameter semantics

CricVision world coordinates use `+x` pitch-right, `+y` bowler-to-striker, and
`+z` up. OpenCV extrinsics use `Xc = R Xw + t`; camera coordinates are x-right,
y-down, z-forward. The bowler wicket is world `y=0`; the striker wicket is at
the official pitch length.

| Parameter | Exact meaning |
| --- | --- |
| `camera_height_m` | Camera centre world `z`, metres above the pitch plane. |
| `distance_behind_wicket_m` | Positive distance from the selected camera-end wicket away from the pitch. Bowler: `-camera_y`; striker: `camera_y-pitch_length`. |
| `lateral_offset_m` | Camera centre world `x`; positive is pitch-right, independent of image mapping. |
| `yaw_deg` | Azimuth of the optical axis from the selected end toward the pitch; positive turns toward world `+x`. |
| `pitch_deg` | Optical-axis elevation from horizontal; positive is upward and normal rear-camera pitch is negative. |
| `roll_deg` | Proper rotation about the optical axis after yaw/pitch; positive rotates camera-right toward reference camera-down. |
| `horizontal_fov_deg` | `2*atan(image_width/(2*fx))`; final `K`, not a candidate label, is authoritative. |
| principal-point offsets | `cx-width/2` and `cy-height/2`, in pixels. |
| `camera_end` | Chooses the reference wicket and longitudinal direction; it does not reflect world geometry. |
| `image_left_mapping` | Chooses the proper camera right/down basis. Alternate mapping is a 180-degree optical-axis rotation, never a reflection. |

Rotation construction is `yaw`, then `pitch`, then optical-axis `roll`.
Decomposition uses `C=-R^T t`, the third row of `R` as world optical direction,
and projections onto the deterministic reference right/down basis. Vertical
optical axes are rejected because yaw is undefined.

## Central conversion and round trip

`camera_preset_parameterization.py` is the sole preset/OpenCV conversion
utility. It owns construction, decomposition, validation, pack/unpack,
normalization, bound comparison, known-camera diagnostics, and projection
round-trip diagnostics. The optimizer calls this reconstruction utility.

The persisted assisted candidate
`A:image_left_to_world_left:fov_45` decomposes to:

| Parameter | Value | Preset bound | Result |
| --- | ---: | ---: | --- |
| lateral | `0.052879 m` | `-2.5..2.5` | inside |
| distance | `7.737437 m` | `3..15` | inside |
| height | `1.390532 m` | `0.75..3` | inside |
| yaw | `-0.430147 deg` | `-15..15` | inside |
| pitch | `-1.328444 deg` | `-18..8` | inside |
| roll | `0.389549 deg` | `-8..8` | inside |
| effective HFOV | `21.960862 deg` | `25..80` | outside by `3.039138 deg` |

No bound changed. The reference is an unaccepted diagnostic candidate, while
the corrected automatic solution is inside all existing bounds.

OpenCV -> preset -> OpenCV over all 36 Virtual Pitch landmarks gives RMSE
`7.81e-14 px`, median `0`, maximum `2.27e-13 px`, camera-position difference
`0 m`, rotation difference `0 deg`, no mirror/reversal, and zero positive-depth
mismatches.

## Objective and eligibility diagnostics

The public diagnostic exposes every point correspondence, line family,
near/far envelope centre/width/height/IoU, temporal metrics, each parameter
prior, physical checks, and aggregate losses. At the assisted camera:

| Objective | Legacy loss | Corrected loss |
| --- | ---: | ---: |
| coarse points | `5106.594` | `2.992` |
| lines | `1702.871` | `1.505` |
| setup envelopes | `36.928` | `1.929` |
| temporal envelopes | `3.595` | `0.149` |
| preset priors | `0.018` | `0.073` |
| final | `6850.006` | `6.649` |

Eligibility replay passes finite pose, height, distance, lateral, yaw, pitch,
roll, positive depth (`7.768 m` minimum), facing (`0.9985`), perspective order
(`3.583`), scene sizing, and broad focal plausibility. It fails only aggregate
preset/FOV eligibility because the effective `21.9609 deg` FOV is below the
development preset.

## Ablation result

Point-only fitting drives distance to `15 m`; envelope-only and data-without-
priors drive HFOV to `25 deg`. These weak, correlated observations do not
independently identify camera scale. Normal corrected priors produce an
eligible interior solution. Stronger priors remain bounded but reduce temporal
agreement. Assisted and existing-PnP initial candidates converge to essentially
the same corrected basin. This isolates the old drift to stale FOV semantics,
unscaled coarse evidence, and unit-conditioned optimization rather than a
physical need for a taller camera.

## Corrected strongest result

For `analysis_20260728_120858_762989`, persisted observations were reused and no
detector ran. The selected interior solution is lateral `-0.067617 m`, distance
`12.347984 m`, height `1.919701 m`, yaw `-0.416834 deg`, pitch `-1.684506 deg`,
roll `-6.762245 deg`, and HFOV `27.462076 deg`. It has no active bounds,
anchor RMSE `18.9362 px`, median error `18.3170 px`, near/far IoU
`0.4598/0.2498`, temporal score `0.2526`, stable uncertainty, and score
`0.3321`.

The camera bridge and projected pitch are produced, but status remains
`NEEDS_ASSISTANCE / VISUAL_ONLY`. `VISUAL_OVERLAY_READY` is not claimed because
the unchanged score threshold is `0.48`; coarse anchor error and weak far/
temporal evidence are the proven blocker. Ambiguity is also high (`0.99999`)
because several bounded starts converge to nearly equal scores.

Compared with the assisted reference, camera-position difference is `4.642 m`,
height `+0.529 m`, distance `+4.611 m`, lateral `-0.120 m`, yaw `+0.013 deg`,
pitch `-0.356 deg`, roll `-7.152 deg`, FOV `+5.501 deg`, and 36-landmark
projection RMSE `105.026 px`. The assisted camera is not treated as ground
truth, but this difference confirms that automatic evidence is not yet strong
enough for readiness.

## Weak clips and build stability

The three weak analyses remain `INSUFFICIENT_WICKETS`, attempt zero fits, reuse
persisted reports, and do not fabricate a second wicket or rerun detection.

The production-build delay was an ignored `.next` directory that was not
writable from the restricted Windows execution context. Direct launch exposed
`EPERM ... mkdir apps/web/.next`. After removing only that ignored output from
the same normal-user context that runs Node, the unchanged real workspace build
completed in `88.685 s`, including compile, lint/type checks, 9/9 static pages,
route optimization, and trace collection. No source/config workaround, API
fetch suppression, SSR change, or disabled check was needed.

## Limits

The available wickets are coarse correlated observations, not exact landmarks;
camera scale and roll remain ambiguous. This patch makes the solver stable and
honest but cannot create missing visual evidence. Automatic acceptance, metric
analytics, Video/Live integration, trajectory, replay, and physics activation
remain outside scope.

## Follow-on evidence result

`wicket_landmark_evidence_upgrade_v1.md` records the next evidence experiment.
The corrected parameterization and objective remain the sole optimization path.
Native multi-frame extraction recovered three axes per wicket, but no defensible
top/base points or transverse scene constraints. The resulting axis-only camera
remained `NEEDS_ASSISTANCE / VISUAL_ONLY` and more ambiguous than required,
confirming that mathematical stability cannot replace missing scale evidence.
