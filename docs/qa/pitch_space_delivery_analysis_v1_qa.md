# Pitch-Space Delivery Analysis Lab V1 QA

## Status

This document is the deterministic QA record for the isolated
`/pitch-space-analysis` development page. Results below were measured on the
local Windows development environment on 2026-08-01/02.

## Contract And Safety Gates

| Gate | Expected | Result |
| --- | --- | --- |
| Versioned strict result | `PitchSpaceDeliveryAnalysisV1`, V1, no extra fields | PASS |
| Partial failures | Earlier valid stages survive a missing later metric | PASS |
| Production acceptance | False; development estimates only | PASS |
| Airborne 3D | Explicitly unavailable | PASS |
| Manual calibration | No anchors, camera height, FOV, or orientation prompt | PASS |
| Virtual Pitch | Existing V1 backend specification reused | PASS |
| Detector/tracker | Existing stump and ball pipelines reused | PASS |
| Frame choice | Frame 0 first; deterministic early fallback only | PASS |
| Camera fit | One transform reused while camera is stable | PASS |
| Persistence | Atomic analysis-owned report; generated media ignored | PASS |

## Deterministic Backend Matrix

Frame selection must cover Frame 0 pass, Frame 0 failure with Frame 5, ranked
early fallback, repeated identical input, blur, clipping, missing far wicket,
portrait, and landscape. Wicket stabilization must cover confidence weighting,
jitter reduction, outlier/clipping rejection, role preservation, and retention
of the selected setup reference.

Homography QA must cover exact synthetic fit, finite inverse, round trip,
portrait/landscape images, left/right hypotheses, reflection handling,
non-collapse, and invalid boxes. Track and metric QA must preserve provenance
and unavailable states while exercising bounce alternatives, line/length zones,
multi-point robust speed, lateral movement, and insufficient evidence.

Camera checks use deterministic intervals and synthetic stable boxes:

| Scenario | Expected status | Metric behavior | Result |
| --- | --- | --- | --- |
| Fixed boxes | `FIXED_CAMERA` | estimates retained | PASS |
| Small coherent drift | `MINOR_DRIFT` | confidence reduced | PASS |
| Sudden zoom | `UNSTABLE_CAMERA` | metric claims stop at drift | PASS |
| Large translation | `UNSTABLE_CAMERA` | image tracking retained | PASS |
| Missing check evidence | explicit warning/unavailable | never assumed fixed | PASS |

## Required Real-Analysis Evaluation

| Analysis | Frame 0 | Fallback / selected frame | Wickets / source | Stable boxes | Fit / camera | Image / pitch points | Bounce / line / length | Speed / movement | Unavailable | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `analysis_20260728_120858_762989` | Pass | none / 0 | both / persisted | near 2, far 2 | READY / FIXED_CAMERA | 45 / 45 | frame 100 / pitch-left / full | unavailable / 0.057 m pitch-left | airborne height, speed | 236.5 ms |
| `analysis_20260718_005833_bbaf9d` | Fail | none / none | insufficient / persisted | none | failed / unavailable | 0 / 0 | unavailable | unavailable | all delivery metrics | 10.9 ms |
| `analysis_20260718_065149_af258b` | Fail | none / none | insufficient / persisted | none | failed / unavailable | 0 / 0 | unavailable | unavailable | all delivery metrics | 12.2 ms |
| `analysis_20260718_090651_051cea` | Fail | none / none | insufficient / persisted | none | failed / unavailable | 0 / 0 | unavailable | unavailable | all delivery metrics | 9.3 ms |
| New upload `analysis_20260801_190145_3e2c65` | Fail | none / none | insufficient / newly generated | none | failed / unavailable | 0 / 0 | unavailable | unavailable | all delivery metrics | 893.2 ms analysis; 1194.9 ms HTTP |

Weak clips must remain partial or unavailable when evidence is insufficient.
Coverage is not a reason to fabricate a fit, bounce, speed, or movement result.

## Timing Record

Record measured milliseconds for upload, Frame 0 decode and stump detection,
fallback, stabilization, pitch fit, reused/new ball tracking, pitch conversion,
bounce, speed, movement, replay preparation, and total. A reused stage must be
identified as reused rather than assigned a synthetic runtime.

## Frontend And Regression QA

Required checks are the focused frontend tests, strict TypeScript, production
build, and regression suites for Wicket Observation V1, Complete Delivery
Tracking V2, and Virtual Pitch V1. Confirm Video Analysis and Live Analysis are
unchanged.

Visual QA viewports:

- desktop: `1440 x 900`;
- mobile: `390 x 844`.

At each viewport verify upload/load-existing controls, Frame 0 status, real
video, pitch replay, synchronized timeline, readable metric/unavailable cards,
preserved media aspect ratio, no horizontal overflow, no duplicate canvas, no
giant-stump regression, and no console errors.

## Measured Verification

- Focused pitch-space backend: 54 passed.
- Full backend regression: 532 passed in 92.6 seconds.
- Strict TypeScript: passed.
- Focused replay test: passed.
- Focused ESLint and `git diff --check`: passed.
- Development route/API: HTTP 200 for result, recent, source-video and reused
  Virtual Pitch endpoints; upload returned HTTP 201.
- Desktop 1440 x 900 and mobile 390 x 844 entry-state QA: passed with no
  overflow or console errors in the specialist run.
- Live result-state browser QA and page-driven upload: not rerun by the lead;
  the current in-app browser policy blocked both localhost development ports.
- Production build: not verified. The restricted run reached `next build` and
  timed out after 300 seconds; the required worker-process rerun could not be
  approved because the execution account had reached its usage limit.

The focused backend command is:

```text
python -m pytest -q tests/test_pitch_space_setup.py tests/test_pitch_space_fit.py tests/test_pitch_space_track.py tests/test_pitch_space_metrics.py tests/test_pitch_space_analysis_contract.py tests/test_pitch_space_camera_stability.py
```
