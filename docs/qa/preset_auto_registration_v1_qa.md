# Preset-Constrained Auto Registration V1 QA

## Scope

This plan validates the development-lab milestone only. It does not validate or
authorize Video Analysis integration, Live Analysis integration, accepted
calibration snapshots, physics execution, or metric analytics.

## Architecture Audit

| Concern | Existing authority | Auto-registration requirement |
| --- | --- | --- |
| Pitch dimensions and landmarks | `virtual_pitch_service.py` | Import the V1 specification; define no dimensions locally. |
| Wicket detection and persisted evidence | `wicket_observation_service.py` | Load `wicket_observations_v1.json` by default; redetect only on an explicit request. |
| solvePnP and pose refinement | `real_pitch_registration_service.py` | Reuse candidate/correspondence/projection mechanics; do not call OpenCV solvePnP directly. |
| Assisted/manual calibration | `scene_calibration_service.py` | Preserve as a distinct fallback; never mutate manual anchors automatically. |
| OpenCV/Three.js conversion | `camera_bridge_service.py` and the existing frontend bridge | Return the existing bridge contract; define no second conversion. |
| Renderer camera ownership | calibrated camera controller | Consume the bridge result; create no Canvas, camera, or renderer in auto-registration utilities. |

The integrated workspace contains the schema, service, routes, and lab UI. The
focused test module executes without skips.

## Public Test Assumptions

The backend tests expect:

- `services.api.schemas.preset_auto_registration.CameraSetupPreset`
- `services.api.schemas.preset_auto_registration.PresetAutoRegistrationResult`
- optional `PresetCompatibilityInput`; a mapping is used when absent
- `get/check` compatibility through `check_preset_compatibility` or
  `evaluate_preset_compatibility`
- fitting through `fit_bounded_camera` or `fit_preset_registration`
- classification through `classify_auto_registration` or
  `classify_registration_result`
- orchestration through `run_preset_auto_registration`
- persistence through `persist_preset_auto_registration` or `_persist_result`

The fit result must expose fitted parameters, attempted candidate ordering,
robust-loss identity, temporal metrics, outlier frames, uncertainty, and evidence
about preset-prior influence. These are diagnostics required by the milestone,
not implementation details to hide.

## Deterministic Coverage

- Contract: version, explicit units, all parameter bounds, invalid inverted
  bounds, required orientation, and supported distortion policy.
- Compatibility: portrait, landscape, rotation, aspect ratio, setup/supporting
  frames, both wickets, clipping, and distortion.
- Evidence reuse: persisted load by default, no detector import, and exactly one
  observation run after explicit redetection.
- Fitting: exact, noisy, weak-prior, strong-evidence, robust single-frame outlier,
  deterministic candidate order, and no escape from preset bounds.
- Temporal: one fixed camera over five supporting frames, one rejected bad frame,
  and stability downgrade.
- Safety: physical rejection, excessive uncertainty downgrade, metrics locked,
  `production_accepted=false`, report-only persistence, and no accepted snapshot.
- Architecture: no local solvePnP, official geometry constants, detector calls,
  or frontend bridge math in the new service.

Synthetic perturbations use NumPy generator seed `20260801`. Tests must not use
wall-clock time, unordered set iteration, platform-dependent random seeds, model
inference, network access, or generated analysis media.

## Integration Matrix

| Case | Expected result |
| --- | --- |
| Strongest persisted two-wicket clip | Honest `AUTO_REGISTRATION_READY`, `VISUAL_OVERLAY_READY`, or `NEEDS_ASSISTANCE`; record all evidence. |
| Weak or one-wicket clip | `INSUFFICIENT_WICKETS` or `INSUFFICIENT_EVIDENCE`; no manufactured evidence. |
| Unsupported rotation/crop/distortion | `PRESET_INCOMPATIBLE`; fitting not invoked. |
| Stable five-frame observations | One camera pose and strong temporal score. |
| One robust outlier | Bad frame reported; stable camera retained when remaining support is sufficient. |
| Inconsistent sequence | Downgraded to assistance/failure; never auto-ready. |
| Low residual, high perturbation spread | Downgraded for uncertainty. |
| Physically invalid attractive fit | Rejected. |

## Manual Browser QA

Run at `1440 x 900` and `390 x 844`:

1. Select the strongest analysis and `STANDARD_REAR_WICKET_NET_V1`.
2. Run Auto Detect and Align with persisted evidence reuse.
3. Verify progress order, readable status, one Canvas, one calibrated camera,
   no camera flash, no giant-stump regression, and no console errors.
4. Resize repeatedly and confirm camera UUID/projection checksum stability.
5. Confirm successful automatic status hides manual anchors.
6. Confirm `NEEDS_ASSISTANCE` reveals the advanced fallback without modifying
   existing manual anchors.
7. Verify no horizontal overflow on mobile and Advanced Calibration is collapsed.
8. Clear the automatic result and verify no accepted snapshot or metric state is
   changed.

## Commands

```powershell
python -m pytest tests/test_preset_auto_registration.py -q
python -m pytest -q
cd apps/web
npx tsc --noEmit --incremental false
npm run build
```

Any module-level skip in `test_preset_auto_registration.py` is a milestone failure
after implementation integration.

## Execution Results - 2026-08-01

- Focused automatic-registration tests: 28 passed.
- Complete backend suite: 405 passed with one existing Starlette deprecation
  warning.
- Five deterministic Virtual Pitch camera/bridge test scripts: passed after TypeScript
  transpilation to an ignored temporary directory.
- Strict TypeScript check: passed.
- FastAPI route workflow: preset list 200, run 200, load 200, clear 204.
- Strong clip `analysis_20260728_120858_762989`: persisted observations reused;
  16 deterministic candidates converged, but none was physically eligible
  because every fit reached a critical preset bound. Final status is
  `NEEDS_ASSISTANCE`; production acceptance and metrics remain locked.
- Weak clips `analysis_20260718_005833_bbaf9d`,
  `analysis_20260718_065149_af258b`, and
  `analysis_20260718_090651_051cea`: `INSUFFICIENT_WICKETS`, zero candidates,
  and no detector rerun.
- Browser QA at 1440 x 900 and 390 x 844: no horizontal overflow, assistance
  fallback visible, no console warnings/errors, and no duplicate canvas.
- Production build: `next build` remained CPU-active but exceeded 600 seconds
  both with and without the dev server. No compiler error was emitted before
  timeout; this check is not recorded as passed.

## Solver Calibration Patch Results - 2026-08-01

- Safety checkpoint: `5bfb0f230b25e75096b15ec6fefa954575a82581`.
- Camera parameterization tests: 18 passed; all 36 landmarks round trip at
  `7.81e-14 px` RMSE with no mirror, end reversal, or depth mismatch.
- Corrected strongest fit: no active bounds; height `1.9197 m`, distance
  `12.3480 m`, HFOV `27.4621 deg`; persisted observations reused.
- Strongest evidence: anchor RMSE `18.9362 px`, near/far IoU `0.4598/0.2498`,
  temporal score `0.2526`, stable uncertainty, score `0.3321`.
- Status remains honest `NEEDS_ASSISTANCE / VISUAL_ONLY`; a bridge camera and
  projected pitch exist, but the unchanged `0.48` visual-readiness threshold is
  not met. Production acceptance is false and metrics remain locked.
- All three weak clips remain `INSUFFICIENT_WICKETS`, with zero candidates and
  no detector rerun.
- Focused camera/auto/real-registration tests: 68 passed.
- Complete backend suite after integration: 426 passed.
- Production build passed in `88.685 s` after clearing only the ignored,
  permission-blocked `.next`; final lead verification passed in `39.421 s`.
  No frontend source or configuration changed.
- Strict TypeScript, five deterministic camera/bridge scripts, desktop
  `1440x900`, and mobile `390x844` checks passed with one canvas, no overflow,
  and no console errors.

See `docs/architecture/auto_registration_solver_calibration_patch_v1.md` and
`docs/qa/auto_registration_build_patch_v1.md` for the measured diagnostics.

## Risks Found

1. `scene_calibration_service.py` can run wicket observation and can accept a
   calibration with downstream physics effects. Auto-registration must not reuse
   those acceptance paths.
2. solvePnP currently exists in both the authoritative real-registration service
   and the synthetic recovery helper in `virtual_pitch_service.py`. A third copy
   in auto-registration would create divergent pose semantics.
3. `wicket_observation_service.py` imports the detector directly by design. The
   new orchestrator must depend on its load/run API, never on detector internals.
4. Official geometry is readily available from `build_virtual_pitch_specification`;
   copying numeric pitch or wicket constants would be unnecessary duplication.
5. Camera bridge precedence includes accepted calibration state. Automatic V1
   results must use an explicit non-accepted source and must not masquerade as an
   accepted snapshot.
6. Soft detector envelopes and coarse pointlike anchors are materially different
   evidence. Tests must not allow boxes to become exact landmark correspondences.
