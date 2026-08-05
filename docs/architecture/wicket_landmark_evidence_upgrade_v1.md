# Wicket Landmark and Scene Evidence Upgrade V1

## Purpose

Preset Auto Registration V1 was stable but underconstrained because one detector box generated correlated centres, edges, widths, heights, and line approximations. This milestone adds an independent, versioned evidence layer that measures visible wicket structure from original video frames. It does not replace the stump detector, camera solver, camera bridge, or Virtual Pitch geometry.

The feature is development-only. It cannot accept calibration or unlock metrics, and it is not mounted in Video Analysis or Live Analysis.

## Data flow

1. Load the persisted `wicket_observations_v1` report.
2. Decode the analysis-owned original video in its native dimensions and apply known orientation once.
3. Rank a bounded deterministic set of supporting frames.
4. Map persisted detector regions into native pixels and create role-specific padded crops.
5. Align crops with bounded translation and optional bounded affine refinement.
6. Build temporal median, edge, and stability evidence.
7. Extract line segments, cluster plausible stump axes, and seek independently supported endpoints and transverse lines.
8. Fuse evidence across frames with confidence, uncertainty, support IDs, and rejection reasons.
9. Persist `reports/wicket_landmark_evidence_v1.json` atomically.
10. Optionally adapt the evidence into the existing corrected preset solver for a development comparison.

The detector is reused as an ROI provider by default. Redetection is an explicit request only.

## Coordinate ownership

All public landmark coordinates use `NATIVE_ORIENTED_PIXELS`. Each crop retains a crop-to-native transform. Detector coordinates are scaled only when persisted detector dimensions differ from decoded native dimensions. Crop bounds are clamped, clipping is measured, and mismatched decoded/setup dimensions fail rather than being silently converted.

Debug media uses validated analysis-owned URLs under `/static/video-analysis/<analysis_id>/calibration/wicket_landmarks_v1/`; filesystem paths are never exposed.

## Frame and ROI strategy

Frame ranking combines persisted detector confidence and stability with decoded sharpness, brightness, local contrast, clipping, obstruction, and crop completeness. Ordering and tie-breaking are deterministic. The selected set is bounded, with at least three useful frames required for temporal evidence.

Near crops use wider horizontal padding plus extra top and base context. Far crops are tighter to preserve effective resolution. Invalid dimensions, severe clipping, unstable boxes, and implausible regions are rejected. Alignment uses a deterministic reference, phase-correlation translation, bounded optional ECC affine refinement, and residual/transform rejection; unconstrained warping is not allowed.

## Classical extraction

Preprocessing uses grayscale contrast normalization, light denoising, gradients, thresholded edges, and Hough line candidates. Candidate axes are filtered by length, orientation, ROI position, local edge support, temporal stability, and spacing. Clusters reject boundary lines, net poles, player legs, mat seams, duplicate shafts, and implausible spacing.

Top/base points require supported shaft termination or a credible transverse-line intersection. Bail and base lines must be near shared shaft endpoints, span a plausible portion of the shaft cluster, and intersect multiple shaft positions. Detector-box edges are never promoted to physical endpoints. Unsupported points and lines are explicitly `UNAVAILABLE` with null coordinates.

Near wickets attempt three, then two, then one axis. Far wickets apply stronger temporal and spacing requirements because their pixel separation is small. The extractor never fills a missing third shaft.

## Temporal consensus and uncertainty

Per-frame evidence is fused in native coordinates using robust medians, grouping tolerances, support ratios, and spread. Confidence combines spatial support, temporal support, alignment quality, and structural plausibility. Perpendicular and angular line uncertainty come from observed spread and support; point uncertainty is axis/intersection spread. Correlated evidence carries a `correlation_family`, and independent-constraint counts use unique families rather than algebraically derived fields.

## Contract

`WicketLandmarkEvidenceResult` is strict Pydantic schema version `v1` and contains native dimensions, rotation, near/far sets, supporting frames, frame selection, temporal alignment, diagnostics, warnings, failures, detector reuse, and immutable production locks.

Points preserve semantic ID, native coordinates, confidence, x/y uncertainty, support frames, extraction method, semantic type, status, correlation family, and rejection reason. Lines preserve endpoints, normalized equation, confidence, angular/perpendicular uncertainty, support frames, extraction method, semantics, status, family, and rejection reason. Semantic types are `EXACT`, `POINTLIKE`, `LINE`, `SOFT`, and `UNAVAILABLE`; classical output is never upgraded to `EXACT` without exact physical evidence.

## Solver integration

Development modes are `LEGACY_COARSE`, `WICKET_LANDMARKS`, and `WICKET_LANDMARKS_WITH_SCENE`. All call the same preset optimizer. Physical stump axes become `STUMP_AXIS` soft correspondences and use projected endpoint-to-infinite-line perpendicular distances plus projected/observed axis-angle residual. Endpoint residuals are used only when endpoints exist. Confidence and measured uncertainty set weight; coarse box families retain their existing semantic floors and correlation normalization.

The adapter does not modify readiness thresholds. Automatic results remain candidates, `production_accepted=false`, and `metrics_unlocked=[]`.

## Evaluation

On `analysis_20260728_120858_762989`, 8 of 11 ranked frames were selected at native `720 x 1280`. Near ROI was `91 x 126`; far ROI was `43 x 76`. All 11 role alignments were accepted with median normalized residual `0.026145`.

Three near axes were supported by frames `0,16,29` at confidence `0.64545` and median perpendicular uncertainty `1.24093 px`. Three far axes were supported by 8 frames at confidence `0.67702` and uncertainty `1.69563 px`. Endpoints, bail lines, and base lines were unavailable because temporal transverse support was insufficient. Both wickets are therefore `PARTIAL`, with three independent axis families each.

This axis-only evidence did not meet the numerical improvement target. The improved and optional-scene modes both remain `NEEDS_ASSISTANCE / VISUAL_ONLY`; near/far IoU is `0.41009/0.13372`, temporal stability is `0.13521`, ambiguity is `0.99999978`, and the 36-landmark RMSE from the assisted reference is `50.1681 px`. The old coarse result remains better on IoU (`0.45981/0.24982`) and temporal stability (`0.25259`). This proves that shaft direction and spacing without supported vertical endpoints or scene ground constraints do not identify absolute scale, height, distance, or FOV.

The three weak analyses return `INSUFFICIENT_EVIDENCE`; their unchanged auto-registration result is `INSUFFICIENT_WICKETS`. No detector was rerun and no landmarks were fabricated.

## Performance

The strongest evidence pass measured `671.341 ms` for composite frame loading/scoring/ROI/alignment/preprocessing, `42.407 ms` for landmark extraction and consensus, and `4.060 ms` for first-pass serialization. The corrected solver rerun measured `24,532.982 ms`; total orchestration was `25,261.051 ms`. Internal frame-preparation sub-stage timings are explicitly null because that service currently exposes one composite measurement.

## Limitations and next step

The current strongest recording resolves shaft axes but not trustworthy physical top/base contacts. Optional scene lines are unavailable. Axis-only observations are scale-ambiguous and can pull the optimizer toward a different but still plausible basin. Debug output shows accepted evidence only; raw/rejected candidate visualization remains numerical unless later exposed safely.

The next milestone should be **Trained Wicket Keypoint Model or Additional Scene Landmarks V1**, evaluated behind this same evidence contract. It should target reliable top/base keypoints or independently semantic crease/pitch intersections before any Video Analysis integration or metric activation.
